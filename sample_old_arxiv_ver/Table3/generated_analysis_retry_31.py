def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Resolve columns case-insensitively ----
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # ---- Restrict to GSS 1993 ----
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---- Genre variables (exact order and labels) ----
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

    # ---- Row labels (exact order) ----
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

    # ---- Missing categories (GSS-style) ----
    # For these music items, DK and NA are typically distinct labeled missing types.
    # In numeric extracts they may appear as 8/9, 98/99, or as negative codes.
    VALID = {1, 2, 3, 4, 5}
    DK_CODES = {8, 98, -1}
    NA_CODES = {9, 99, -2}

    DK_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_TOKENS = {"n", "na", "no answer", "noanswer"}

    def _is_blank_str(series: pd.Series) -> pd.Series:
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            s = series.astype("string")
            return s.isna() | (s.str.strip() == "")
        return pd.Series(False, index=series.index)

    def _counts_for_item(series: pd.Series):
        # Numeric representation
        s_num = pd.to_numeric(series, errors="coerce")

        # Substantive counts
        c1 = int((s_num == 1).sum())
        c2 = int((s_num == 2).sum())
        c3 = int((s_num == 3).sum())
        c4 = int((s_num == 4).sum())
        c5 = int((s_num == 5).sum())

        # Explicit DK/NA numeric codes
        dk = int(s_num.isin(list(DK_CODES)).sum())
        na = int(s_num.isin(list(NA_CODES)).sum())

        # Explicit DK/NA string tokens (only when not numeric)
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            low = series.astype("string").str.strip().str.lower()
            numeric_nan = s_num.isna()
            dk += int((numeric_nan & low.isin(DK_TOKENS)).sum())
            na += int((numeric_nan & low.isin(NA_TOKENS)).sum())

        # IMPORTANT: Do NOT "redistribute" plain NaN between DK and NA.
        # If the dataset preserves DK/NA as labeled missing, they will appear as explicit codes
        # (8/9, 98/99, -1/-2, or tokens). Unlabeled NaN remain unclassified missing and are
        # excluded from DK/NA counts (as they cannot be split without metadata).
        # This avoids systematic swapping/redistribution errors.
        mean_val = s_num.where(s_num.isin(list(VALID))).mean()

        return c1, c2, c3, c4, c5, dk, na, mean_val

    # ---- Build table with Attitude as first column ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]

        # If a column is entirely NaN (or blank strings), still produce zeros/NaN mean
        if s.isna().all() or _is_blank_str(s).all():
            c1 = c2 = c3 = c4 = c5 = dk = na = 0
            mean_val = np.nan
        else:
            c1, c2, c3, c4, c5, dk, na, mean_val = _counts_for_item(s)

        table.loc["(1) Like very much", genre_label] = c1
        table.loc["(2) Like it", genre_label] = c2
        table.loc["(3) Mixed feelings", genre_label] = c3
        table.loc["(4) Dislike it", genre_label] = c4
        table.loc["(5) Dislike very much", genre_label] = c5
        table.loc["(M) Don’t know much about it", genre_label] = dk
        table.loc["(M) No answer", genre_label] = na
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for display (counts ints; mean 2 decimals) ----
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

    # ---- Save human-readable text file with 3 panels (6 genres each) ----
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
        f.write(title + "\n\n")
        f.write("Counts shown for 1–5 plus (M) Don’t know much about it and (M) No answer.\n")
        f.write("Mean computed over valid responses 1–5 only; missing categories excluded from mean.\n")
        f.write("DK/NA counts are computed from explicit codes (8/9, 98/99, -1/-2) and/or string tokens when present.\n")
        f.write("Unlabeled blank/NaN values are treated as missing but are not split into DK vs NA.\n")

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