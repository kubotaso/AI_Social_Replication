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
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --- Genre variables (Table 3 columns; exact order) ---
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

    # --- Table rows (exact order) ---
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

    # --- Missing code handling for GSS extracts: use explicit codes only (no inference) ---
    # Common codes for these items: DK=8, NA=9 (sometimes DK=98/-1, NA=99/-2)
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

    def _as_clean_string(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip()

    def _count_item(series: pd.Series):
        s = series

        # Numeric parsing for all values; strings like "8" become 8
        sn = pd.to_numeric(s, errors="coerce")

        # Substantive counts
        c1 = int((sn == 1).sum())
        c2 = int((sn == 2).sum())
        c3 = int((sn == 3).sum())
        c4 = int((sn == 4).sum())
        c5 = int((sn == 5).sum())

        # Missing-category counts based on explicit codes/tokens
        dk = int(sn.isin(DK_NUM_CODES).sum())
        na = int(sn.isin(NA_NUM_CODES).sum())

        # If some values are non-numeric tokens, count them too (without double-counting numerics)
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            low = _as_clean_string(s).str.lower()
            nonnum = pd.to_numeric(low, errors="coerce").isna() & low.notna() & (low != "")
            dk += int((low.isin(DK_STR_TOKENS) & nonnum).sum())
            na += int((low.isin(NA_STR_TOKENS) & nonnum).sum())

        # Mean on valid 1..5 only
        mean_val = sn.where(sn.isin([1, 2, 3, 4, 5])).mean()

        return c1, c2, c3, c4, c5, dk, na, mean_val

    # --- Build output table with explicit Attitude label column ---
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk, na, mean_val = _count_item(s)

        table.loc["(1) Like very much", genre_label] = c1
        table.loc["(2) Like it", genre_label] = c2
        table.loc["(3) Mixed feelings", genre_label] = c3
        table.loc["(4) Dislike it", genre_label] = c4
        table.loc["(5) Dislike very much", genre_label] = c5
        table.loc["(M) Don’t know much about it", genre_label] = dk
        table.loc["(M) No answer", genre_label] = na
        table.loc["Mean", genre_label] = mean_val

    # --- Format for display (counts as ints, means to 2 decimals) ---
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

    # --- Save human-readable text file in 3 panels (6 genres each) ---
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