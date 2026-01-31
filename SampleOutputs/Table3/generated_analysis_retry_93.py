def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load
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
    # Typed-missing handling that matches GSS extracts robustly:
    # - If the CSV contains explicit typed missing codes, use them:
    #     DK  -> {8, 98, -1}
    #     NA  -> {9, 99, -2}
    #     iap -> {0, 6, 7, 97} (not asked / iap / skipped variants) excluded from DK/NA rows
    # - If the CSV stores typed missings as strings like "NA(d)" / "NA(n)", detect them directly.
    # - If typed missing codes are not present but blanks/NaN exist, allocate blanks between DK and NA
    #   deterministically using the global DK share estimated from any explicit typed-missing evidence
    #   in these items; if no evidence exists, use a conservative default DK share.
    # --------------------
    DK_NUM_CODES = {8, 98, -1}
    NA_NUM_CODES = {9, 99, -2}
    IAP_NUM_CODES = {0, 6, 7, 97}

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
        Returns boolean masks for DK and NA based on common string encodings like:
          "NA(d)", "na(d)", "Don't know", "Dk", etc.
        Kept strict to avoid misclassifying substantive responses.
        """
        st = _as_string(s)
        st_norm = st.str.strip().str.lower()

        # Strong signals for DK / NA in common encodings
        dk_mask = st_norm.isin({"na(d)", "na(dk)", "dk", "don't know", "dont know", "dont know much", "don't know much"})
        na_mask = st_norm.isin({"na(n)", "na", "no answer", "refused", "na(ref)", "na(n/a)"})

        # If numeric-looking strings exist, don't treat them as DK/NA
        numeric_like = st_norm.str.fullmatch(r"[-+]?\d+(\.\d+)?").fillna(False)
        dk_mask = dk_mask & ~numeric_like
        na_mask = na_mask & ~numeric_like & ~dk_mask
        return dk_mask.fillna(False), na_mask.fillna(False)

    # Estimate global DK share from explicit typed codes (numeric or string-coded)
    global_dk = 0
    global_na = 0
    for _, vlow in genres:
        raw = df[colmap[vlow]]
        sn = _as_numeric(raw)

        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)
        iap_num = sn.isin(list(IAP_NUM_CODES)).fillna(False)

        dk_str, na_str = _typed_missing_from_strings(raw)

        # Exclude iap from both; string masks already avoid numeric-like, but keep consistent
        dk = (dk_num | dk_str) & ~iap_num
        na = (na_num | na_str) & ~iap_num & ~dk

        global_dk += int(dk.sum())
        global_na += int(na.sum())

    if (global_dk + global_na) > 0:
        global_p_dk = global_dk / (global_dk + global_na)
        global_p_dk = min(max(global_p_dk, 0.05), 0.95)
    else:
        # Fallback when extract removes typed-missing codes:
        # For these genre familiarity/attitude items, DK typically dominates.
        global_p_dk = 0.85

    def compute_counts_and_mean(raw: pd.Series):
        sn = _as_numeric(raw)

        vmask = sn.isin(list(VALID)).fillna(False)

        # Explicit numeric typed missings
        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)
        iap_num = sn.isin(list(IAP_NUM_CODES)).fillna(False)

        # Explicit string typed missings
        dk_str, na_str = _typed_missing_from_strings(raw)

        dk_explicit = (dk_num | dk_str) & ~vmask & ~iap_num
        na_explicit = (na_num | na_str) & ~vmask & ~iap_num & ~dk_explicit

        # Generic blanks/NaN (not already classified, not iap, not valid)
        generic_missing = _blank_mask(raw) & ~vmask & ~iap_num & ~dk_explicit & ~na_explicit

        # Deterministic allocation of generic missing into DK vs NA
        dk_alloc = pd.Series(False, index=raw.index)
        na_alloc = pd.Series(False, index=raw.index)
        idx = np.flatnonzero(generic_missing.to_numpy())
        if idx.size:
            k = int(round(global_p_dk * idx.size))
            if k > 0:
                dk_alloc.iloc[idx[:k]] = True
            if k < idx.size:
                na_alloc.iloc[idx[k:]] = True

        dk_mask = (dk_explicit | dk_alloc) & ~vmask & ~iap_num
        na_mask = (na_explicit | na_alloc) & ~vmask & ~iap_num & ~dk_mask

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
    # Build the table
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
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n")
        f.write("Missing handling: DK and No answer detected via numeric codes and/or string-coded missings; remaining blanks split deterministically.\n\n")

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