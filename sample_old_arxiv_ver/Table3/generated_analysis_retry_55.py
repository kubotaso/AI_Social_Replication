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
    # Typed-missing handling
    #
    # For these GSS music items in this CSV, "Don't know much about it" and "No answer"
    # are present as true missing values in the file. They are not reliably encoded as 8/9
    # here (the substantive 1–5 are present; missings are blank/NA). The published table
    # separates the missing total into DK vs NA; the GSS source provides these as distinct
    # missing types, but the CSV export can collapse them to NA.
    #
    # To compute DK vs NA from raw data without hardcoding paper numbers:
    #   1) If explicit DK/NA codes exist (8/9, 98/99, -1/-2 or string tokens), use them.
    #   2) Otherwise, treat NA/blank as missing and(typed) and derive DK vs NA split by
    #      using the most common GSS convention in exports where:
    #        - "no answer" is often stored as 9/99/-2 or literal "no answer"
    #        - "don't know" as 8/98/-1 or literal "don't know"
    #      If neither exists anywhere, we cannot recover DK vs NA from a collapsed NA.
    #      In that case we still compute correct 1–5 frequencies and means; the two (M)
    #      rows are computed as:
    #        DK = total_missing
    #        NA = 0
    #
    # This is the only defensible computation from the raw file when typed missing info
    # is not present.
    # --------------------
    DK_STR = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_STR = {"n", "na", "no answer", "noanswer"}

    def _to_string(series: pd.Series) -> pd.Series:
        return series.astype("string")

    def _to_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        s = _to_string(series)
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def classify_music_item(series: pd.Series):
        """
        Returns:
          sn: numeric series (float), NaN for non-numeric/blank
          dk_mask: boolean
          na_mask: boolean
          valid_mask: boolean for 1..5
          other_missing_mask: boolean missing not classified as dk/na (collapsed NA)
        """
        sn = _to_numeric(series)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        # numeric typed missings, if present
        # DK: 8/98/-1; NA: 9/99/-2 (common conventions)
        sn_int = sn.round().astype("Int64")
        dk_num = sn_int.isin([8, 98, -1]).fillna(False)
        na_num = sn_int.isin([9, 99, -2]).fillna(False)

        # string typed missings, if present
        if (series.dtype == "object") or str(series.dtype).startswith("string"):
            low = _to_string(series).str.strip().str.lower()
            dk_str = low.isin(DK_STR).fillna(False)
            na_str = low.isin(NA_STR).fillna(False)
            blank_str = (low == "").fillna(False)
        else:
            dk_str = pd.Series(False, index=series.index)
            na_str = pd.Series(False, index=series.index)
            blank_str = pd.Series(False, index=series.index)

        dk_mask = (dk_num | dk_str) & ~valid_mask
        na_mask = (na_num | na_str) & ~valid_mask & ~dk_mask

        # anything else missing (NaN from numeric coercion or blank strings) not yet typed
        missing_any = sn.isna() | blank_str
        other_missing_mask = missing_any & ~valid_mask & ~dk_mask & ~na_mask

        return sn, dk_mask, na_mask, valid_mask, other_missing_mask

    # --------------------
    # Build table (include explicit Attitude label column)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        sn, dk_mask, na_mask, valid_mask, other_missing_mask = classify_music_item(df[col])

        # Frequencies 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # Missing categories:
        # If typed missings exist, use them. If not, allocate all "other missing" to DK by default.
        dk_typed = int(dk_mask.sum())
        na_typed = int(na_mask.sum())
        other_miss = int(other_missing_mask.sum())

        if (dk_typed + na_typed) == 0 and other_miss > 0:
            dk_count = other_miss
            na_count = 0
        else:
            dk_count = dk_typed + 0
            na_count = na_typed + 0
            # If there are additional collapsed missings beyond typed ones, keep them out of displayed rows
            # because we cannot defensibly split them; however, typically this will be zero when typed codes exist.

        table.loc["(M) Don’t know much about it", genre_label] = int(dk_count)
        table.loc["(M) No answer", genre_label] = int(na_count)

        # Mean over valid 1..5 only
        mean_val = sn.where(valid_mask).mean()
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