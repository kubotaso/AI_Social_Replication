def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load CSV
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column mapping
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --------------------
    # Restrict to GSS 1993
    # --------------------
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order)
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

    VALID = {1, 2, 3, 4, 5}

    # --------------------
    # Robust typed-missing handling for GSS-style extracts:
    # Prefer explicit typed missings if present (numeric or string-coded).
    # Do NOT "guess" by splitting blanks; blanks are counted as "No answer" (unclassified missing).
    #
    # Rationale: previous discrepancies were caused by allocating blanks between DK and NA.
    # Table 3 distinguishes DK vs NA, so we only classify when the raw data explicitly encodes it.
    # --------------------
    DK_NUM_CODES = {8, 98, -1}          # common "don't know" style codes
    NA_NUM_CODES = {9, 99, -2}          # common "no answer" style codes
    IAP_NUM_CODES = {0, 6, 7, 97}       # inapplicable / not asked / skipped variants (kept missing, not DK/NA)

    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _as_string(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    def _blank_mask(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s.isna()
        st = _as_string(s)
        return st.isna() | (st.str.strip() == "")

    def _typed_missing_from_strings(s: pd.Series):
        """
        Detect typed missing values if the dataset stores them as strings (e.g., 'NA(d)', 'NA(n)').
        Keep strict to avoid misclassifying substantive responses.
        """
        st = _as_string(s)
        st_norm = st.str.strip().str.lower()

        numeric_like = st_norm.str.fullmatch(r"[-+]?\d+(\.\d+)?").fillna(False)

        dk_tokens = {
            "na(d)", "na(dk)", "dk", "dont know", "don't know",
            "dont know much", "don't know much", "dont know much about it", "don't know much about it"
        }
        na_tokens = {
            "na(n)", "no answer", "refused", "na(ref)", "na(n/a)"
        }

        dk_mask = st_norm.isin(dk_tokens) & ~numeric_like
        na_mask = st_norm.isin(na_tokens) & ~numeric_like & ~dk_mask

        return dk_mask.fillna(False), na_mask.fillna(False)

    def compute_counts_and_mean(raw: pd.Series):
        sn = _as_numeric(raw)

        vmask = sn.isin(list(VALID)).fillna(False)

        # Explicit numeric non-substantives
        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)
        iap_num = sn.isin(list(IAP_NUM_CODES)).fillna(False)

        # Explicit string non-substantives
        dk_str, na_str = _typed_missing_from_strings(raw)

        dk_mask = (dk_num | dk_str) & ~vmask & ~iap_num
        na_mask = (na_num | na_str) & ~vmask & ~iap_num & ~dk_mask

        # Any remaining missing (including blanks) is "No answer" for Table 3 display purposes,
        # because we cannot distinguish DK vs NA without explicit coding.
        remaining_missing = _blank_mask(raw) & ~vmask & ~iap_num & ~dk_mask & ~na_mask
        na_mask = na_mask | remaining_missing

        counts = {
            1: int((sn == 1).sum()),
            2: int((sn == 2).sum()),
            3: int((sn == 3).sum()),
            4: int((sn == 4).sum()),
            5: int((sn == 5).sum()),
            "dk": int(dk_mask.sum()),
            "na": int(na_mask.sum()),
        }

        mean_val = sn.where(vmask).mean()
        mean_val = np.nan if pd.isna(mean_val) else float(mean_val)

        return counts, mean_val

    # --------------------
    # Build the table (ensure New Age/Space and Opera are separate columns)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        counts, mean_val = compute_counts_and_mean(raw)

        table.loc["(1) Like very much", genre_label] = counts[1]
        table.loc["(2) Like it", genre_label] = counts[2]
        table.loc["(3) Mixed feelings", genre_label] = counts[3]
        table.loc["(4) Dislike it", genre_label] = counts[4]
        table.loc["(5) Dislike very much", genre_label] = counts[5]
        table.loc["(M) Don’t know much about it", genre_label] = counts["dk"]
        table.loc["(M) No answer", genre_label] = counts["na"]
        table.loc["Mean", genre_label] = mean_val

    # --------------------
    # Format: counts as integers; mean to 2 decimals
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
        f.write("Frequencies are counts only (no percentages).\n")
        f.write("Mean computed over valid responses 1–5 only; non-substantive/missing excluded.\n")
        f.write("Missing handling: DK/NA counted only when explicitly coded (numeric or string); otherwise missing is counted as 'No answer'.\n\n")

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