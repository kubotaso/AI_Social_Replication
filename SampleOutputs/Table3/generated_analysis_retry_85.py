def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + filter to 1993
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)

    # normalize column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    df = df.loc[df["YEAR"].eq(1993)].copy()

    # -----------------------
    # Variables (Table 3)
    # -----------------------
    genre_map = [
        ("Latin/Salsa", "LATIN"),
        ("Jazz", "JAZZ"),
        ("Blues/R&B", "BLUES"),
        ("Show Tunes", "MUSICALS"),
        ("Oldies", "OLDIES"),
        ("Classical/Chamber", "CLASSICL"),
        ("Reggae", "REGGAE"),
        ("Swing/Big Band", "BIGBAND"),
        ("New Age/Space", "NEWAGE"),
        ("Opera", "OPERA"),
        ("Bluegrass", "BLUGRASS"),
        ("Folk", "FOLK"),
        ("Pop/Easy Listening", "MOODEASY"),
        ("Contemporary Rock", "CONROCK"),
        ("Rap", "RAP"),
        ("Heavy Metal", "HVYMETAL"),
        ("Country/Western", "COUNTRY"),
        ("Gospel", "GOSPEL"),
    ]

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        "(M) Don't know much about it",
        "(M) No answer",
        "Mean",
    ]

    # Ensure all required columns exist (case-insensitive already handled)
    missing_vars = [v for _, v in genre_map if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required variables in CSV: {missing_vars}")

    # -----------------------
    # Missing-category detection
    # -----------------------
    # Goal: compute separate counts for:
    #   - DK: "(M) Don't know much about it"
    #   - NA: "(M) No answer"
    #
    # We DO NOT guess/split missing values. If DK/NA are not distinguishable in the export,
    # we fail with a clear error telling the user to re-export with preserved codes.
    #
    # We support multiple encodings commonly seen in GSS extracts:
    #   - strings like "[NA(d)]", "NA(d)", "DON'T KNOW", "DONT KNOW", "DK"
    #   - strings like "[NA(n)]", "NA(n)", "NO ANSWER", "NA"
    #   - numeric codes sometimes used for DK/NA (e.g., 8/9 or 98/99)
    #
    # If the column is purely numeric with NaN for missing, DK vs NA cannot be separated.

    def _as_clean_str(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _mask_tokens(s_str: pd.Series, tokens):
        mask = pd.Series(False, index=s_str.index)
        for t in tokens:
            t_up = str(t).upper()
            # literal contains, no regex needed
            mask = mask | s_str.str.contains(t_up, regex=False, na=False)
        return mask

    DK_TOKENS = [
        "[NA(D)]", "NA(D)", "NAD",
        "DON'T KNOW", "DONT KNOW", "DK",
        "DON'T KNOW MUCH", "DONT KNOW MUCH",
        "DON'T KNOW MUCH ABOUT IT", "DONT KNOW MUCH ABOUT IT",
    ]
    NA_TOKENS = [
        "[NA(N)]", "NA(N)", "NAN",
        "NO ANSWER",
    ]

    # If numeric codes exist for DK/NA, add them here.
    # These are only applied when the series is numeric-like (or string numeric).
    DK_NUM_CODES = {8, 98}
    NA_NUM_CODES = {9, 99}

    def _extract_counts_and_mean(raw: pd.Series, varname: str):
        # Parse numeric values
        x_num = pd.to_numeric(raw, errors="coerce")

        # Valid substantive responses 1..5
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        # Attempt to detect DK/NA explicitly, first via strings
        s_str = _as_clean_str(raw)

        dk_mask_str = _mask_tokens(s_str, DK_TOKENS)
        na_mask_str = _mask_tokens(s_str, NA_TOKENS)

        # Additionally detect numeric DK/NA codes
        dk_mask_num = x_num.isin(list(DK_NUM_CODES))
        na_mask_num = x_num.isin(list(NA_NUM_CODES))

        dk_mask = dk_mask_str | dk_mask_num
        na_mask = na_mask_str | na_mask_num

        # If a value is classified as DK/NA, it is missing for mean and not in 1..5.
        # (No need to remove from valid: valid is already only 1..5)

        # Identify other missing/unclassified non-1..5 values
        non_1_5 = ~x_num.isin([1, 2, 3, 4, 5]) | x_num.isna()
        unclassified = non_1_5 & ~(dk_mask | na_mask)

        # If there are any unclassified missing/non-1..5, we cannot split DK vs NA reliably
        # unless unclassified count is zero. (We refuse to guess.)
        if int(unclassified.sum()) > 0:
            # Provide a small diagnostic: example raw values and counts
            ex_vals = raw.loc[unclassified].head(10).tolist()
            unclassified_n = int(unclassified.sum())
            dk_n = int(dk_mask.sum())
            na_n = int(na_mask.sum())
            non1_5_n = int((~x_num.isin([1, 2, 3, 4, 5])).sum())
            nan_n = int(x_num.isna().sum())

            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                f"this CSV export does not preserve distinguishable DK/NA categories for all missing/non-1..5 values. "
                f"Unclassified missing/non-1..5 count={unclassified_n} (of total rows={len(raw)}). "
                f"Detected DK={dk_n}, NA={na_n}. "
                f"Numeric diagnostics: non-1..5={non1_5_n}, NaN={nan_n}. "
                f"Example unclassified raw values: {ex_vals}. "
                f"Re-export data preserving DK vs No-answer codes (e.g., '[NA(d)]' and '[NA(n)]' or distinct numeric codes)."
            )

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table (return value)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _extract_counts_and_mean(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Save human-readable text file in 3 blocks (6 columns each)
    # -----------------------
    formatted = table.copy()

    # Format integer rows as ints; mean as 2 decimals
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table