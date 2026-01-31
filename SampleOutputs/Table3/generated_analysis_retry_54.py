def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load data
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column lookup
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --------------------
    # Restrict to YEAR == 1993
    # --------------------
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

    VALID = {1, 2, 3, 4, 5}

    # --------------------
    # Robust typed-missing classification for GSS-style codes
    #
    # We must correctly separate:
    #   DK: don't know much about it  -> coded as 8 or 98 (and sometimes other *8 endings)
    #   NA: no answer                 -> coded as 9 or 99 (and sometimes other *9 endings)
    #
    # In many GSS extracts, these appear as 8/9 and/or 98/99.
    # This function classifies any NON-VALID numeric code whose last digit is 8 as DK,
    # and any NON-VALID numeric code whose last digit is 9 as NA.
    # Everything else outside 1..5 is treated as missing but not displayed in Table 3.
    # --------------------
    def to_num(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        s = series.astype("string")
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def classify_codes(sn: pd.Series):
        # Work with integer-ish representation, preserving NA
        sn_int = sn.round().astype("Int64")

        valid_mask = sn_int.isin(list(VALID)).fillna(False)

        # DK: codes ending in 8 (e.g., 8, 98)
        dk_mask = ((sn_int % 10) == 8).fillna(False) & ~valid_mask

        # NA: codes ending in 9 (e.g., 9, 99)
        na_mask = ((sn_int % 10) == 9).fillna(False) & ~valid_mask & ~dk_mask

        return sn_int, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table (include explicit Attitude label column)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        sn = to_num(df[col])
        sn_int, valid_mask, dk_mask, na_mask = classify_codes(sn)

        # Frequencies 1..5
        table.loc["(1) Like very much", genre_label] = int((sn_int == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn_int == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn_int == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn_int == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn_int == 5).sum())

        # Missing categories (typed missings)
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        # Mean over valid 1..5 only
        mean_val = sn_int.where(valid_mask).astype(float).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --------------------
    # Format for display (counts as ints; mean to 2 decimals)
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
        f.write("Mean computed over valid responses 1–5 only; missing categories excluded from mean.\n\n")

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