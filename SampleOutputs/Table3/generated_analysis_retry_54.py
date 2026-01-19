def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)

    # Normalize column names (input file uses lowercase in sample)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Keep YEAR==1993 (exclude missing automatically by the equality check)
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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required variable not found in data: {v}")

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

    # -----------------------
    # Helpers
    # -----------------------
    def _series_as_str_upper(s: pd.Series) -> pd.Series:
        # robust string view for detecting "[NA(d)]" style tokens if present
        return s.astype("string").str.strip().str.upper()

    def _find_missing_masks(raw: pd.Series):
        """
        Attempt to separate:
          - DK: [NA(d)] or equivalent label
          - NA: [NA(n)] or equivalent label
        across several common encodings.

        Returns:
          valid_numeric: float series with only 1..5 retained, else NaN
          dk_mask: boolean mask (same index)
          na_mask: boolean mask (same index)

        If the data do not preserve separable DK vs NA (e.g., both are just blank/NaN),
        this function will raise, because we must *compute* those rows from raw data.
        """
        # Numeric parse (keeps NaN for non-numeric)
        x = pd.to_numeric(raw, errors="coerce")
        valid_numeric = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # First: detect explicit string tokens like "[NA(d)]", "NA(d)", etc.
        s_up = _series_as_str_upper(raw)

        dk_token = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b|\bDON'?T KNOW\b|\bDONT KNOW\b", regex=True, na=False)
        na_token = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b|\bNO ANSWER\b", regex=True, na=False)

        # Second: detect common numeric special codes if present (some exports keep these)
        # We only consider codes that are NOT 1..5.
        # (These are conservative "usual suspects"; if they don't exist, they contribute 0.)
        dk_num = x.isin([8, 98, 998])   # often DK
        na_num = x.isin([9, 99, 999])   # often NA/refused/NA

        dk_mask = dk_token | dk_num
        na_mask = na_token | na_num

        # Anything outside 1..5 that isn't classified is "other missing"
        nonvalid_nonnull = x.notna() & ~x.isin([1, 2, 3, 4, 5])
        other_missing = raw.isna() | nonvalid_nonnull | (s_up.eq("") if hasattr(s_up, "eq") else False)

        # If we have explicit DK/NA encodings, we can split.
        # But if DK and NA are not distinguishable and there exist missing values, we must error.
        classified_any = int(dk_mask.sum()) + int(na_mask.sum())
        total_missing_pool = int((valid_numeric.isna()).sum())

        # Note: valid_numeric.isna includes both true missing and special codes; that is OK.
        # To be "separable", either:
        #   (a) we observed any explicit DK/NA classifications; OR
        #   (b) there are no missing at all.
        if total_missing_pool > 0 and classified_any == 0:
            # Provide a helpful diagnostic: show unique non-1..5 numeric codes (if any)
            uniq_nonvalid = (
                pd.Series(pd.unique(x[nonvalid_nonnull].dropna()))
                .sort_values()
                .tolist()
            )
            raise ValueError(
                "Dataset does not preserve separable missing categories for this item. "
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts "
                "from this CSV export because all missings are collapsed (e.g., blank/NA). "
                f"Non-1..5 numeric codes observed (if any): {uniq_nonvalid}. "
                "Re-export data preserving distinct missing codes (e.g., '[NA(d)]' and '[NA(n)]' "
                "or distinct numeric codes for DK vs NA)."
            )

        # If there are missing values beyond the classified masks, keep them out of both DK and NA
        # (they are "other missing" not shown in Table 3; we will not include them in DK/NA rows).
        # This is consistent with the table spec that only shows DK and NA rows.
        # (Means always computed only on 1..5.)
        return valid_numeric, dk_mask, na_mask

    # -----------------------
    # Build Table 3 (counts + mean)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        valid, dk_mask, na_mask = _find_missing_masks(df[var])

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # -----------------------
    # Save human-readable text output (3 blocks of 6 genres)
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    display = table.copy()

    # Format counts as integers; mean as 2 decimals
    for r in display.index:
        if r == "Mean":
            display.loc[r] = display.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            display.loc[r] = display.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display.insert(0, "Attitude", display.index)
    display = display.reset_index(drop=True)

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table