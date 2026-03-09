def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # ---- Table 3 genre variables (order and labels) ----
    genres = [
        ("Latin/Salsa", "latin"),
        ("Jazz", "jazz"),
        ("Blues/R&B", "blues"),
        ("Show Tunes", "musicals"),
        ("Oldies", "oldies"),
        ("Classical/Chamber", "classicl"),
        ("Reggae", "reggae"),
        ("Swing/Big Band", "bigband"),
        ("New Age/Space", "newage"),
        ("Opera", "opera"),
        ("Bluegrass", "blugrass"),
        ("Folk", "folk"),
        ("Pop/Easy Listening", "moodeasy"),
        ("Contemporary Rock", "conrock"),
        ("Rap", "rap"),
        ("Heavy Metal", "hvymetal"),
        ("Country/Western", "country"),
        ("Gospel", "gospel"),
    ]
    missing_vars = [v for _, v in genres if v not in colmap]
    if missing_vars:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing_vars}")

    # ---- Rows (exact order) ----
    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        "(M) Don’t know much about it",
        "(M) No answer",
        "Mean",
    ]
    VALID = {1, 2, 3, 4, 5}

    # ---- Helpers: robust typed-missing detection for numeric-only extracts ----
    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _as_string(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    def _typed_missing_from_strings(raw: pd.Series):
        """
        Detect typed missing from textual exports (e.g., 'NA(d)' / 'NA(n)' / 'dk' / 'no answer').
        Returns boolean masks (dk, na).
        """
        st = _as_string(raw)
        st_norm = st.str.strip().str.lower()

        numeric_like = st_norm.str.fullmatch(r"[-+]?\d+(\.\d+)?").fillna(False)

        dk_tokens = {
            "na(d)", "na(dk)", "dk",
            "dont know", "don't know",
            "dont know much", "don't know much",
            "dont know much about it", "don't know much about it",
            "dontknow", "don'tknow",
        }
        na_tokens = {"na(n)", "no answer", "noanswer"}

        dk = (st_norm.isin(dk_tokens) & ~numeric_like).fillna(False)
        na = (st_norm.isin(na_tokens) & ~numeric_like & ~dk).fillna(False)
        return dk, na

    def _infer_dk_na_codes(series_num: pd.Series):
        """
        Infer which numeric codes in this dataset correspond to DK vs NA.
        For these GSS music items, extracts commonly use:
          DK: 8/98/-1, NA: 9/99/-2 (or vice-versa in some recodes).
        We infer by choosing the mapping that yields (a) both present across items and
        (b) DK generally larger than NA (typical for 'don't know much about it' vs 'no answer').
        """
        candidates_a = {"dk": {8, 98, -1}, "na": {9, 99, -2}}
        candidates_b = {"dk": {9, 99, -2}, "na": {8, 98, -1}}

        def score(mapping):
            dk_total = 0
            na_total = 0
            dk_items_present = 0
            na_items_present = 0
            for _, vlow in genres:
                sn = _as_numeric(df[colmap[vlow]])
                dk_ct = int(sn.isin(list(mapping["dk"])).sum())
                na_ct = int(sn.isin(list(mapping["na"])).sum())
                dk_total += dk_ct
                na_total += na_ct
                if dk_ct > 0:
                    dk_items_present += 1
                if na_ct > 0:
                    na_items_present += 1

            # Prefer mapping where both appear across many items and dk > na
            # Penalty if one side never appears.
            presence = dk_items_present + na_items_present
            balance = (dk_total - na_total)
            both_nonzero = int(dk_total > 0) + int(na_total > 0)
            # Strongly discourage mapping where NA dwarfs DK (usually wrong for these items)
            wrong_direction_penalty = 0
            if dk_total == 0 and na_total > 0:
                wrong_direction_penalty = 10**9
            if na_total == 0 and dk_total > 0:
                wrong_direction_penalty = 10**8  # still possible but less ideal
            if balance < 0:
                wrong_direction_penalty += abs(balance) * 1000

            return presence * 10_000 + balance * 10 + both_nonzero * 1_000_000 - wrong_direction_penalty

        s_a = score(candidates_a)
        s_b = score(candidates_b)
        return candidates_a if s_a >= s_b else candidates_b

    inferred_mapping = _infer_dk_na_codes(pd.Series(dtype=float))
    DK_NUM_CODES = inferred_mapping["dk"]
    NA_NUM_CODES = inferred_mapping["na"]

    def _classify_music_item(raw: pd.Series):
        """
        Classify responses into:
          - valid 1..5
          - dk (don't know much about it)
          - na (no answer)
        using inferred numeric typed-missing mapping plus string tokens if present.
        Remaining missing/other codes -> NA.
        """
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)

        dk_str, na_str = _typed_missing_from_strings(raw)

        dk_mask = (dk_num | dk_str) & ~valid_mask
        na_mask = (na_num | na_str) & ~valid_mask & ~dk_mask

        # Remaining NaN/blank (generic missing) -> NA (no answer)
        if pd.api.types.is_numeric_dtype(raw):
            generic_missing = raw.isna()
        else:
            st = _as_string(raw)
            generic_missing = st.isna() | (st.str.strip() == "")
        generic_missing = generic_missing & ~valid_mask & ~dk_mask & ~na_mask
        na_mask = na_mask | generic_missing

        # Any other non-valid numeric codes (refused/other) -> NA
        other_nonvalid = (~sn.isna()) & (~valid_mask) & (~dk_mask) & (~na_mask)
        na_mask = na_mask | other_nonvalid

        return sn, valid_mask, dk_mask, na_mask

    # ---- Build the output table ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, vmask, dk_mask, na_mask = _classify_music_item(raw)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(vmask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # ---- Format (counts as integers; mean to 2 decimals) ----
    formatted = table.copy()
    for r in row_labels:
        for c in formatted.columns:
            if c == "Attitude":
                continue
            v = formatted.loc[r, c]
            if r == "Mean":
                formatted.loc[r, c] = "" if pd.isna(v) else f"{float(v):.2f}"
            else:
                formatted.loc[r, c] = "" if pd.isna(v) else str(int(v))

    # ---- Save as human-readable text in 3 panels (6 genres each) ----
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    panels = [
        [g[0] for g in genres[0:6]],
        [g[0] for g in genres[6:12]],
        [g[0] for g in genres[12:18]],
    ]

    def pad(text, width, align="left"):
        text = "" if text is None else str(text)
        if len(text) >= width:
            return text
        if align == "right":
            return " " * (width - len(text)) + text
        if align == "center":
            left = (width - len(text)) // 2
            right = width - len(text) - left
            return " " * left + text + " " * right
        return text + " " * (width - len(text))

    att_col = "Attitude"
    row_w = max(len(att_col), int(formatted[att_col].astype(str).map(len).max())) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write("Frequencies are counts only (no percentages).\n")
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n")
        f.write(f"Typed-missing mapping inferred as DK={sorted(list(DK_NUM_CODES))} and NA={sorted(list(NA_NUM_CODES))} when numeric codes are present; blanks/other codes treated as No answer.\n\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            header = pad(att_col, row_w, "left") + "".join(pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = pad(formatted.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[r, c]
                    line += pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")
            f.write("\n")

    return formatted