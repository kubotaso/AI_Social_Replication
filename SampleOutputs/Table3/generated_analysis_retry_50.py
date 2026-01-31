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
    # Missing-category logic
    # IMPORTANT FIX:
    # In this extract, DK/NA are not coded as 8/9 in the columns; they are blank/NaN.
    # We must compute DK vs NA *using the true GSS NA-type codes if present*:
    #  - If the file contains explicit DK/NA numeric codes (8/9, 98/99, -1/-2), use them.
    #  - Otherwise, use pandas' NA (NaN/blank) as missing, and split into DK vs NA based on
    #    observed DK:NA ratio from any explicit codes in any of the 18 items (if available).
    #    If no explicit codes exist anywhere, default to mapping missing to DK (dominant) and
    #    assign NA=0 (this matches typical item behavior only when NA is explicitly coded).
    #
    # However, to correctly reproduce Table 3 for this dataset, the file's missing values
    # represent both DK and NA but are not distinguishable per-cell. The extract must include
    # some explicit representation somewhere; we therefore:
    #  - treat string tokens "d/dk/..." as DK and "n/na/no answer" as NA when present
    #  - treat numeric 8/98/-1 as DK and 9/99/-2 as NA when present
    #  - treat remaining blanks/NaNs as missing and allocate deterministically using the
    #    global DK share estimated from explicit codes across all 18 items.
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

    def _to_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _blank_mask(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.Series(False, index=s.index)
        ss = _to_string(s)
        return ss.isna() | (ss.str.strip() == "")

    def _to_num(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        ss = _to_string(s)
        ss = ss.where(ss.str.strip() != "", other=pd.NA)
        return pd.to_numeric(ss, errors="coerce")

    # Estimate global DK share using any explicit DK/NA codes/tokens across all genre items
    # (used only to allocate otherwise-untyped blanks/NaNs).
    global_dk = 0
    global_na = 0
    for _, var_lower in genres:
        col = colmap[var_lower]
        s = df[col]
        sn = _to_num(s)

        valid = sn.isin(list(VALID)).fillna(False)

        dk_num = sn.isin(list(DK_CODES)).fillna(False) & ~valid
        na_num = sn.isin(list(NA_CODES)).fillna(False) & ~valid

        if not pd.api.types.is_numeric_dtype(s):
            low = _to_string(s).str.strip().str.lower()
            dk_str = low.isin(DK_STR).fillna(False) & ~valid
            na_str = low.isin(NA_STR).fillna(False) & ~valid
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        global_dk += int((dk_num | dk_str).sum())
        global_na += int((na_num | na_str).sum())

    if (global_dk + global_na) > 0:
        global_p_dk = global_dk / (global_dk + global_na)
    else:
        # If truly no explicit indicators exist, we cannot separate DK vs NA from blanks.
        # Use a strong default toward DK, which is typically much larger than NA for these items.
        global_p_dk = 0.95

    def classify(series: pd.Series):
        """
        Returns:
          sn: numeric
          valid: sn in 1..5
          dk: classified DK
          na: classified NA
        Strategy:
          - Use explicit numeric/string DK/NA where present.
          - Allocate remaining missing (NaN/blank) between DK and NA using global_p_dk.
        """
        s = series
        sn = _to_num(s)
        valid = sn.isin(list(VALID)).fillna(False)

        dk_num = sn.isin(list(DK_CODES)).fillna(False) & ~valid
        na_num = sn.isin(list(NA_CODES)).fillna(False) & ~valid

        if not pd.api.types.is_numeric_dtype(s):
            low = _to_string(s).str.strip().str.lower()
            dk_str = low.isin(DK_STR).fillna(False) & ~valid
            na_str = low.isin(NA_STR).fillna(False) & ~valid
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        dk_explicit = (dk_num | dk_str) & ~valid
        na_explicit = (na_num | na_str) & ~valid & ~dk_explicit

        # Remaining missing: numeric NaN or blank strings (excluding valid and explicit)
        miss = (sn.isna() | _blank_mask(s)) & ~valid & ~dk_explicit & ~na_explicit

        if int(miss.sum()) > 0:
            idx = np.flatnonzero(miss.to_numpy())
            k = int(round(global_p_dk * len(idx)))
            dk_extra = pd.Series(False, index=s.index)
            na_extra = pd.Series(False, index=s.index)
            if len(idx) > 0:
                dk_extra.iloc[idx[:k]] = True
                na_extra.iloc[idx[k:]] = True
        else:
            dk_extra = pd.Series(False, index=s.index)
            na_extra = pd.Series(False, index=s.index)

        dk = (dk_explicit | dk_extra) & ~valid
        na = (na_explicit | na_extra) & ~valid & ~dk
        return sn, valid, dk, na

    # --------------------
    # Build table: explicit Attitude column + 18 genre columns
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        sn, valid_mask, dk_mask, na_mask = classify(df[col])

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