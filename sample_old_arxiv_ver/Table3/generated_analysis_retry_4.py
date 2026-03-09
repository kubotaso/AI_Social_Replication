def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 ---
    year_col = None
    for c in ("YEAR", "year", "Year"):
        if c in df.columns:
            year_col = c
            break
    if year_col is None:
        raise KeyError("YEAR/year column not found in dataset.")
    df = df.loc[df[year_col] == 1993].copy()

    # --- Genre variables (Table 3 columns) ---
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

    # Resolve actual column names case-insensitively
    colmap = {c.lower(): c for c in df.columns}
    missing = [var for _, var in genres if var not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # --- Table 3 rows (response categories + mean) ---
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

    # --- Missing-code classification ---
    # This dataset likely uses standard GSS numeric missing codes:
    # DK: 8 or 98; No answer: 9 or 99; sometimes -1/-2; sometimes string tokens.
    DK_TOKENS = {
        "d", "dk", "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_TOKENS = {"n", "na", "no answer", "noanswer"}
    DK_NUM_CODES = {-1, 8, 98}
    NA_NUM_CODES = {-2, 9, 99}

    def classify_missing(series: pd.Series):
        """
        Return (dk_mask, na_mask, valid_mask) where valid is 1..5.
        Treat pandas NaN as neither DK nor NA (not displayed separately in Table 3).
        """
        # numeric parse for checking 1..5 and numeric missing sentinels
        sn = pd.to_numeric(series, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        # numeric missing
        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)

        # string missing tokens (if applicable)
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            ss = series.astype("string")
            low = ss.str.strip().str.lower()
            dk_str = low.isin(DK_TOKENS).fillna(False)
            na_str = low.isin(NA_TOKENS).fillna(False)
            dk_mask = dk_num | dk_str
            na_mask = na_num | na_str
        else:
            dk_mask = dk_num
            na_mask = na_num

        return dk_mask, na_mask, valid_mask

    # --- Build table (counts + mean) ---
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=object)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]
        sn = pd.to_numeric(s, errors="coerce")

        # Counts for categories 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # Missing categories shown in Table 3
        dk_mask, na_mask, valid_mask = classify_missing(s)
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        # Mean on valid 1..5 only
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --- Format output: counts as ints, mean to 2 decimals ---
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )
        else:
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else str(int(x))
            )

    # --- Save as human-readable text (avoid merged headers) ---
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    # Use a generous fixed column width so adjacent headers remain distinct (e.g., New Age/Space vs Opera)
    col_width = 16
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(formatted.to_string(col_space=col_width, justify="center"))
        f.write("\n")

    return formatted