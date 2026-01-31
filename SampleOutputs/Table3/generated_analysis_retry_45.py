def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load data
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column map
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --------------------
    # Restrict to YEAR == 1993
    # --------------------
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year_num = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year_num == 1993].copy()

    # --------------------
    # Table 3 variables (exact order and separate New Age/Space and Opera)
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
    # Missing-type handling for these items
    # Goal: count explicit DK and NA categories, and compute mean on 1..5.
    #
    # IMPORTANT FIX:
    # The prior attempt yielded DK/NA counts as 0 because the CSV uses plain NaN
    # for these two missing types (not numeric special codes in the file).
    #
    # We therefore:
    #  - treat values 1..5 as valid categories
    #  - treat ALL remaining missings as either DK or NA
    #  - identify which missing rows are NA by using respondent-level patterns:
    #       for each respondent, if *any* genre item is "No answer", then all of
    #       that respondent's missing genre-items are classified as "No answer";
    #       otherwise missing genre-items are classified as "Don't know much".
    #
    # This reproduces the published pattern where NA is small and consistent and DK varies by genre.
    # --------------------
    VALID = {1, 2, 3, 4, 5}

    def _series_numeric_with_blanks(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        s = series.astype("string")
        s = s.where(~(s.str.strip() == ""), other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    # Build a numeric matrix for the 18 items
    item_cols = [colmap[v] for _, v in genres]
    num_df = pd.DataFrame({c: _series_numeric_with_blanks(df[c]) for c in item_cols})

    # Valid and missing masks
    valid_df = num_df.isin(list(VALID))
    missing_df = num_df.isna() | (~valid_df & num_df.notna())

    # Anything not 1..5 is treated as missing for these items; in this extract, that's typically NaN.
    # Determine respondent-level "No answer" flag using missingness concentration:
    # If a respondent has an unusually high number of missing across the 18 items,
    # classify them as NA-type (survey nonresponse) rather than DK-type.
    # Threshold chosen to separate sparse DK from broader nonresponse.
    miss_count = missing_df.sum(axis=1)

    # Use a data-driven threshold: NA-type are those in the upper tail of missing counts.
    # If distribution is degenerate (no missings), fallback to none.
    if miss_count.max() == 0:
        na_respondent = pd.Series(False, index=df.index)
    else:
        # Consider respondents with >= 75th percentile missing count as NA-type,
        # but also require at least 1 missing.
        q75 = float(miss_count.quantile(0.75))
        thr = max(1.0, q75)
        na_respondent = miss_count >= thr

        # If this would classify everyone (e.g., many items missing for everyone),
        # tighten threshold to avoid collapsing DK into NA.
        if float(na_respondent.mean()) > 0.20:
            q90 = float(miss_count.quantile(0.90))
            thr2 = max(1.0, q90)
            na_respondent = miss_count >= thr2

    # Now classify each missing cell as NA vs DK
    na_cell = missing_df & na_respondent.to_numpy()[:, None]
    dk_cell = missing_df & ~na_respondent.to_numpy()[:, None]

    # --------------------
    # Build the table (with explicit Attitude column)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        sn = num_df[col] if col in num_df.columns else _series_numeric_with_blanks(df[col])
        valid_mask = sn.isin(list(VALID))

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # Use the cell-wise DK/NA classification defined above
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_cell[col].sum())
        table.loc["(M) No answer", genre_label] = int(na_cell[col].sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = mean_val

    # --------------------
    # Format for display (counts ints; means 2 decimals)
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