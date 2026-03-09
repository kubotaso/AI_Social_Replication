def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Resolve columns case-insensitively ---
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --- Restrict to GSS 1993 ---
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # --- Table 3 genre variables (exact set/order) ---
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

    # --- Row labels (exact order) ---
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

    # --- Robust NA(d)/NA(n) handling WITHOUT guessing/splitting ---
    # For this GSS battery, missing-type categories are typically coded:
    #   8 = Don't know much about it
    #   9 = No answer
    # We also defensively recognize common variants (e.g., 98/99, -1/-2) and string tokens.
    DK_NUM_CODES = {8, 98, -1}
    NA_NUM_CODES = {9, 99, -2}

    DK_STR_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_STR_TOKENS = {"n", "na", "no answer", "noanswer"}

    def _blank_str_mask(s: pd.Series) -> pd.Series:
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            ss = s.astype("string")
            return ss.isna() | (ss.str.strip() == "")
        return pd.Series(False, index=s.index)

    def _compute_item_counts(series: pd.Series):
        """
        Compute counts for 1..5, DK, NA, and mean over 1..5.

        Important: Do NOT reallocate unknown missings between DK and NA.
        Only count DK/NA where explicitly coded (numeric or string).
        Any other missing/special codes are treated as missing for the mean but NOT displayed
        in DK/NA rows (since Table 3 only shows those two missing categories).
        """
        s = series

        # numeric parsing for main categories and numeric DK/NA codes
        sn = pd.to_numeric(s, errors="coerce")

        # substantive counts
        c = {k: int((sn == k).sum()) for k in [1, 2, 3, 4, 5]}

        # explicit DK/NA masks (numeric)
        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)

        # explicit DK/NA masks (string tokens)
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            low = s.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR_TOKENS).fillna(False)
            na_str = low.isin(NA_STR_TOKENS).fillna(False)
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        # Ensure disjointness and exclude substantive responses
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)
        dk_mask = (dk_num | dk_str) & ~valid_mask
        na_mask = (na_num | na_str) & ~valid_mask & ~dk_mask

        dk_cnt = int(dk_mask.sum())
        na_cnt = int(na_mask.sum())

        # mean over 1..5 only
        mean_val = sn.where(valid_mask).mean()

        # Note: other missing/special codes and blank/NaN are excluded from mean, but not displayed
        # in either DK/NA rows to avoid misallocation.
        return c[1], c[2], c[3], c[4], c[5], dk_cnt, na_cnt, mean_val

    # --- Build table with Attitude as first column ---
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk_cnt, na_cnt, mean_val = _compute_item_counts(s)

        table.loc["(1) Like very much", genre_label] = c1
        table.loc["(2) Like it", genre_label] = c2
        table.loc["(3) Mixed feelings", genre_label] = c3
        table.loc["(4) Dislike it", genre_label] = c4
        table.loc["(5) Dislike very much", genre_label] = c5
        table.loc["(M) Don’t know much about it", genre_label] = dk_cnt
        table.loc["(M) No answer", genre_label] = na_cnt
        table.loc["Mean", genre_label] = mean_val

    # --- Format for output (counts as integers; mean to 2 decimals) ---
    formatted = table.copy()
    for r in row_labels:
        if r == "Mean":
            for c in formatted.columns:
                if c == "Attitude":
                    continue
                v = formatted.loc[r, c]
                formatted.loc[r, c] = "" if pd.isna(v) else f"{float(v):.2f}"
        else:
            for c in formatted.columns:
                if c == "Attitude":
                    continue
                v = formatted.loc[r, c]
                formatted.loc[r, c] = "" if pd.isna(v) else str(int(v))

    # --- Save as human-readable text file with 3 panels (6 genres each) ---
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    panels = [
        [g[0] for g in genres[0:6]],
        [g[0] for g in genres[6:12]],
        [g[0] for g in genres[12:18]],
    ]

    def _pad(text, width, align="left"):
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
    row_w = max(len(att_col), max(len(str(x)) for x in formatted[att_col].tolist())) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("Notes: Counts shown for 1–5 plus two explicit missing categories. Mean computed over 1–5 only.\n")
        f.write("Missing categories counted only when explicitly coded (numeric: DK in {8,98,-1}; NA in {9,99,-2}; or matching common string tokens).\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            header = _pad(att_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = _pad(formatted.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[r, c]
                    line += _pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")

    return formatted