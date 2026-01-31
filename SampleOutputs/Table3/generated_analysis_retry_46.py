def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column resolver
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --------------------
    # Restrict to YEAR == 1993
    # --------------------
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year_num = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year_num == 1993].copy()

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

    # --------------------
    # Robust classification of responses
    #   Valid: 1..5
    #   DK: explicit DK codes if present; otherwise cannot be inferred from plain NaN
    #   NA: explicit NA codes if present; otherwise plain NaN treated as NA
    #
    # IMPORTANT: Do NOT "redistribute" missing between DK and NA. Only count DK/NA when
    # they are explicitly encoded; otherwise all remaining missing are "No answer".
    # --------------------
    VALID = {1, 2, 3, 4, 5}

    # Candidate numeric codes that GSS-style extracts often use
    DK_NUM_CODES = {8, 98, -1}        # don't know
    NA_NUM_CODES = {9, 99, -2}        # no answer
    # Sometimes special missing sets appear in some extracts; treat as NA-type unless DK-known
    OTHER_NA_CODES = {0, 97, 998, 999, -3, -4, -5, -6, -7, -8, -9}

    DK_STR_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_STR_TOKENS = {
        "n", "na", "no answer", "noanswer", "refused", "refuse",
        "missing", "miss", "blank"
    }

    def _to_num(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        s = series.astype("string")
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def classify(series: pd.Series):
        s = series
        sn = _to_num(s)

        valid_mask = sn.isin(list(VALID))

        # explicit numeric DK/NA
        dk_num = sn.isin(list(DK_NUM_CODES))
        na_num = sn.isin(list(NA_NUM_CODES))
        other_na_num = sn.isin(list(OTHER_NA_CODES))

        # explicit string DK/NA
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            low = s.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR_TOKENS).fillna(False)
            na_str = low.isin(NA_STR_TOKENS).fillna(False)
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        dk_mask = (dk_num | dk_str) & ~valid_mask
        na_mask = (na_num | na_str | other_na_num) & ~valid_mask & ~dk_mask

        # Remaining missings (NaN / blank / non-numeric) counted as No answer (cannot distinguish from DK)
        blank = pd.Series(False, index=s.index)
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            low2 = s.astype("string")
            blank = low2.isna() | (low2.str.strip() == "")
        remainder_missing = (sn.isna() | blank) & ~valid_mask & ~dk_mask & ~na_mask
        na_mask = na_mask | remainder_missing

        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table with an explicit Attitude column
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

            header = _pad(att_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = _pad(formatted.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[r, c]
                    line += _pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")
            f.write("\n")

    return formatted