def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to 1993 ----
    year_candidates = ["YEAR", "year", "Year"]
    year_col = next((c for c in year_candidates if c in df.columns), None)
    if year_col is None:
        raise KeyError("YEAR/year column not found in dataset.")
    df = df.loc[df[year_col] == 1993].copy()

    # ---- Genres (columns) ----
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

    # resolve actual column names case-insensitively
    colmap = {c.lower(): c for c in df.columns}
    missing_cols = [v for _, v in genres if v not in colmap]
    if missing_cols:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing_cols}")

    # ---- Table 3 rows ----
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

    # ---- Missing handling ----
    # GSS extracts commonly use:
    # - for DK: 8 or 98 (sometimes -1 or 'd'/'dk')
    # - for No answer: 9 or 99 (sometimes -2 or 'n')
    DK_TOKENS = {
        "d", "dk", "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_TOKENS = {"n", "no answer", "noanswer"}
    DK_NUM_CODES = {-1, 8, 98}
    NA_NUM_CODES = {-2, 9, 99}

    def classify_missing(series: pd.Series):
        """
        Returns dk_mask, na_mask, valid_mask (valid == 1..5).
        Treat pandas NaN as neither DK nor NA for Table 3 display.
        """
        s = series
        sn = pd.to_numeric(s, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        if s.dtype == "object" or str(s.dtype).startswith("string"):
            ss = s.astype("string")
            low = ss.str.strip().str.lower()
            dk_mask = low.isin(DK_TOKENS).fillna(False) | sn.isin(list(DK_NUM_CODES)).fillna(False)
            na_mask = low.isin(NA_TOKENS).fillna(False) | sn.isin(list(NA_NUM_CODES)).fillna(False)
            return dk_mask, na_mask, valid_mask

        dk_mask = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_mask = sn.isin(list(NA_NUM_CODES)).fillna(False)
        return dk_mask, na_mask, valid_mask

    # ---- Build table (counts + mean) ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=object)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]
        sn = pd.to_numeric(s, errors="coerce")

        # counts 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # DK and No answer counts (computed from raw codes, not hardcoded)
        dk_mask, na_mask, valid_mask = classify_missing(s)
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        # mean on valid 1..5 only
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # format mean to 2 decimals, counts as ints
    formatted = table.copy()
    for r in row_labels:
        if r == "Mean":
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )
        else:
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else str(int(x))
            )

    # ---- Save to text (ensure columns do not visually merge) ----
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    # Use fixed column width so "New Age/Space" and "Opera" stay distinct
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(formatted.to_string(col_space=14, justify="center"))
        f.write("\n")

    return formatted