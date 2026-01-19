def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names to match mapping (upper-case)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Keep YEAR == 1993 (exclude missing YEAR implicitly)
    df = df.loc[pd.to_numeric(df["YEAR"], errors="coerce").eq(1993)].copy()

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
        "(M) Don’t know much about it",
        "(M) No answer",
        "Mean",
    ]

    def _series_to_clean_string(s):
        # Keep NA as <NA> so .str ops behave; avoid turning NaN into "nan" text
        return s.astype("string")

    def _count_dk_na_and_valid(raw_series):
        """
        Returns:
          counts_1_5: Series indexed [1..5] of ints
          dk_count: int ("don't know much about it" / NA(d))
          na_count: int ("no answer" / NA(n))
          mean_val: float (mean over valid 1..5)
        Robust to:
          - numeric-coded responses (1..5)
          - NaN blanks
          - bracket-coded missings like "[NA(d)]", "[NA(n)]"
          - (fallback) infers DK/NA when non-1..5 numeric codes exist and no bracket codes exist
        """
        s_str = _series_to_clean_string(raw_series)
        s_up = s_str.str.upper()

        # Detect explicit bracket-coded missing categories (common in some GSS extracts)
        dk_mask_str = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask_str = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Parse numeric values where possible
        x = pd.to_numeric(raw_series, errors="coerce")

        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid_num = x.where(valid_mask, np.nan)

        # Counts for 1..5
        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Initialize DK/NA counts from explicit string detection
        dk_count = int(dk_mask_str.sum())
        na_count = int((na_mask_str & ~dk_mask_str).sum())

        # If no explicit bracket-coded DK/NA observed, try numeric inference on non-1..5 codes
        # (only for cases where the extract uses numeric special codes rather than bracket strings).
        any_explicit = (dk_count + na_count) > 0
        non_valid_numeric = x.notna() & (~valid_mask)

        if (not any_explicit) and non_valid_numeric.any():
            vc = x.loc[non_valid_numeric].value_counts()

            # Heuristic: two most frequent non-1..5 numeric codes correspond to DK and NA,
            # where DK tends to be more common than NA.
            top_codes = list(vc.index[:2])

            dk_code = None
            na_code = None
            if len(top_codes) >= 1:
                if len(top_codes) == 1:
                    dk_code = top_codes[0]
                else:
                    c1, c2 = top_codes[0], top_codes[1]
                    if vc.loc[c1] >= vc.loc[c2]:
                        dk_code, na_code = c1, c2
                    else:
                        dk_code, na_code = c2, c1

            if dk_code is not None:
                dk_count = int((x == dk_code).sum())
            if na_code is not None:
                na_count = int((x == na_code).sum())

        # Mean computed only on valid 1..5
        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        return counts_1_5, dk_count, na_count, mean_val

    # Build numeric table
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        counts_1_5, dk_count, na_count, mean_val = _count_dk_na_and_valid(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # Format for output: counts as ints, Mean as 2 decimals
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    # Save as three 6-column blocks to match presentation
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = formatted.loc[:, cols].copy()
            block_df.index.name = "Attitude"
            f.write(block_df.to_string())
            f.write("\n\n")

    return table