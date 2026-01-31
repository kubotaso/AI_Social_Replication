def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 ---
    year_col = next((c for c in ["YEAR", "year", "Year"] if c in df.columns), None)
    if year_col is None:
        raise KeyError("YEAR/year column not found in dataset.")
    df = df.loc[df[year_col] == 1993].copy()

    # --- Genre variables (Table 3 columns): (display label, dataset field) ---
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

    # --- Table 3 rows ---
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

    # --- Missing code handling (GSS conventions): "Don't know" vs "No answer" ---
    DK_STR_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
        "don't know much about", "don’t know much about",
    }
    NA_STR_TOKENS = {"n", "na", "no answer", "noanswer"}

    def classify_missing(series: pd.Series):
        """
        Return (dk_mask, na_mask, valid_mask) where:
          - valid_mask indicates substantive responses 1..5
          - dk_mask indicates "Don't know much about it"
          - na_mask indicates "No answer"
        Detects numeric codes using common GSS schemes: 8/9, 98/99, -1/-2,
        plus string tokens if the column is stored as text.
        """
        sn = pd.to_numeric(series, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        present = set(sn.dropna().unique().tolist())

        dk_codes, na_codes = [], []

        # Prefer the common GSS pairings if present
        if 8 in present or 9 in present:
            if 8 in present:
                dk_codes.append(8)
            if 9 in present:
                na_codes.append(9)
        if 98 in present or 99 in present:
            # If 8/9 not present, 98/99 are likely the missing pair
            if not dk_codes and 98 in present:
                dk_codes.append(98)
            if not na_codes and 99 in present:
                na_codes.append(99)
        if -1 in present or -2 in present:
            # If neither 8/9 nor 98/99 used, -1/-2 are common in some extracts
            if not dk_codes and -1 in present:
                dk_codes.append(-1)
            if not na_codes and -2 in present:
                na_codes.append(-2)

        dk_num = sn.isin(dk_codes).fillna(False) if dk_codes else pd.Series(False, index=sn.index)
        na_num = sn.isin(na_codes).fillna(False) if na_codes else pd.Series(False, index=sn.index)

        # Add string token detection (defensive)
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            low = series.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR_TOKENS).fillna(False)
            na_str = low.isin(NA_STR_TOKENS).fillna(False)
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

        # Substantive counts 1..5
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

    # --- Format for clean text export ---
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

    # --- Save as human-readable text file (avoid merged headers by ensuring spacing) ---
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    # Use a generous column spacing so adjacent long headers don't visually run together
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(formatted.to_string(col_space=22, justify="center"))
        f.write("\n")

    return formatted