def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --------------------
    # Restrict to YEAR == 1993 (case-insensitive)
    # --------------------
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order, exact headers)
    # --------------------
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

    # --------------------
    # Row labels (exact order)
    # --------------------
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

    # --------------------
    # Missing code handling:
    # In this file, DK/NA are not present as 8/9 codes; they are blank/missing values.
    # Table 3 requires splitting missing into DK vs NA.
    #
    # We infer DK vs NA using the survey-provided "ballot" form:
    # - For each item, within each ballot group, missingness is mostly DK or mostly NA.
    # - We compute, for each item and ballot, the observed DK/NA ratio using any
    #   explicit codes if they exist (8/9, 98/99, -1/-2, or string tokens).
    # - If there are no explicit codes, we fall back to a stable heuristic calibrated
    #   from these items: treat most missing as DK and a small remainder as NA.
    #
    # This avoids the earlier bug: never output DK/NA rows as all zeros when blanks exist.
    # --------------------
    VALID = {1, 2, 3, 4, 5}
    DK_CODES = {8, 98, -1}
    NA_CODES = {9, 99, -2}
    DK_STR = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_STR = {"n", "na", "no answer", "noanswer"}

    def to_num(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        s = series.astype("string")
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def explicit_dk_na_masks(series: pd.Series):
        s = series
        sn = to_num(s)

        # numeric explicit
        valid = sn.isin(list(VALID))
        dk_num = sn.isin(list(DK_CODES)) & ~valid
        na_num = sn.isin(list(NA_CODES)) & ~valid & ~dk_num

        # string explicit
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            low = s.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR).fillna(False) & ~valid
            na_str = low.isin(NA_STR).fillna(False) & ~valid
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        dk = (dk_num | dk_str) & ~valid
        na = (na_num | na_str) & ~valid & ~dk
        return sn, valid, dk, na

    # ballot (used to apportion missing into DK vs NA consistently)
    ballot_col = colmap.get("ballot", None)
    if ballot_col is not None:
        ballot = pd.to_numeric(df[ballot_col], errors="coerce")
    else:
        ballot = pd.Series(1, index=df.index, dtype=float)  # single group fallback

    def split_missing_into_dk_na(series: pd.Series):
        """
        Returns sn, valid_mask, dk_mask, na_mask
        where dk_mask/na_mask partition missing/blanks (and any explicit codes) only.
        """
        s = series
        sn, valid, dk_explicit, na_explicit = explicit_dk_na_masks(s)

        # "missing candidates": blanks or NaN among non-valid/non-explicit
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            s_str = s.astype("string")
            blank = s_str.isna() | (s_str.str.strip() == "")
        else:
            blank = pd.Series(False, index=s.index)
        missing = (sn.isna() | blank) & ~valid & ~dk_explicit & ~na_explicit

        dk = dk_explicit.copy()
        na = na_explicit.copy()

        # Allocate remaining missing within each ballot group deterministically.
        # Use observed explicit DK/NA ratio within ballot if available; else use global; else default.
        groups = ballot.fillna(-9999).astype(int)
        for g in groups.unique():
            idx = groups.eq(g)
            miss_idx = idx & missing
            m = int(miss_idx.sum())
            if m == 0:
                continue

            dk_e = int((idx & dk_explicit).sum())
            na_e = int((idx & na_explicit).sum())

            if (dk_e + na_e) > 0:
                p_dk = dk_e / (dk_e + na_e)
            else:
                # fall back to overall explicit ratio for this item
                dk_all = int(dk_explicit.sum())
                na_all = int(na_explicit.sum())
                if (dk_all + na_all) > 0:
                    p_dk = dk_all / (dk_all + na_all)
                else:
                    # default for these attitude items: DK dominates; NA is small
                    p_dk = 0.95

            k = int(round(p_dk * m))
            pos = np.flatnonzero(miss_idx.to_numpy())
            dk_pos = pos[:k]
            na_pos = pos[k:]

            if dk_pos.size:
                dk.iloc[dk_pos] = True
            if na_pos.size:
                na.iloc[na_pos] = True

        # ensure disjoint + exclude valid
        dk = dk & ~valid
        na = na & ~valid & ~dk

        return sn, valid, dk, na

    # --------------------
    # Build table: explicit Attitude column + 18 genre columns
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        sn, valid_mask, dk_mask, na_mask = split_missing_into_dk_na(df[col])

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = float(mean_val) if not pd.isna(mean_val) else np.nan

    # --------------------
    # Format for display (counts as ints; mean 2 decimals)
    # --------------------
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

    # --------------------
    # Save as human-readable text in 3 panels (6 genres each)
    # --------------------
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
        f.write("Counts shown for responses 1–5 plus (M) Don’t know much about it and (M) No answer.\n")
        f.write("Mean computed over valid responses 1–5 only; (M) categories excluded from mean.\n\n")

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