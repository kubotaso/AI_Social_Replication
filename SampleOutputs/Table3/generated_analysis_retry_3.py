def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Filter to YEAR == 1993
    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")
    df = df.loc[df["YEAR"].eq(1993)].copy()

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

    # ---- Missing-code handling ----
    # In this release, missing may be stored as:
    #   - NaN (blank)
    #   - bracketed strings like "[NA(d)]", "[NA(n)]", etc.
    #   - other non-1..5 numeric codes (e.g., 8/9) depending on extraction
    #
    # For Table 3, we must split missing into:
    #   DK = "don't know much about it"  (NA(d) or equivalent)
    #   NA = "no answer"                (NA(n) or equivalent)
    #
    # If the dataset uses numeric DK/NA codes, we detect them by looking at the
    # most common non-1..5 codes among non-missing values and map the top two
    # to DK and NA (DK typically much larger than NA in this instrument).
    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def _extract_missing_buckets(raw_series):
        """
        Returns:
          valid_num: numeric series with only 1..5, else NaN
          dk_mask: boolean mask for DK/NA(d)
          na_mask: boolean mask for No answer/NA(n)
        """
        s = raw_series

        # Start with string-based detection (covers "[NA(d)]" style exports)
        s_str = s.astype("string")
        s_up = s_str.str.upper()

        dk_mask = s_up.str.contains(r"\bNA\(D\)\b|\[NA\(D\)\]", regex=True, na=False)
        na_mask = s_up.str.contains(r"\bNA\(N\)\b|\[NA\(N\)\]", regex=True, na=False)

        # Numeric parsing
        x = _to_num(s)

        # Valid substantive responses
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If we already found explicit DK/NA strings, great; otherwise attempt numeric inference
        # for observations that are not 1..5 and not NaN.
        not_valid_numeric = x.notna() & (~x.isin([1, 2, 3, 4, 5]))

        # Only infer if we have no explicit bracketed codes at all in this variable
        if (dk_mask.sum() + na_mask.sum()) == 0 and not_valid_numeric.any():
            # Candidate codes and their frequencies
            vc = x.loc[not_valid_numeric].value_counts(dropna=True)

            # If there is only one extra code, treat it as DK (common) and NA as 0
            # If 2+ codes, take two most common; the larger frequency is DK, smaller is NA.
            top_codes = list(vc.index[:2])

            if len(top_codes) == 1:
                dk_code = top_codes[0]
                na_code = None
            else:
                c1, c2 = top_codes[0], top_codes[1]
                if vc.loc[c1] >= vc.loc[c2]:
                    dk_code, na_code = c1, c2
                else:
                    dk_code, na_code = c2, c1

            dk_mask = dk_mask | (x.eq(dk_code) & not_valid_numeric)
            if na_code is not None:
                na_mask = na_mask | (x.eq(na_code) & not_valid_numeric)

        # Any remaining missing-like observations (NaN or other codes) are not split;
        # they are excluded from the mean and not counted in the two displayed M rows.
        return valid_num, dk_mask, na_mask

    # ---- Build table ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _extract_missing_buckets(raw)

        # Counts for 1..5
        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Counts for DK/NA
        dk_count = int(dk_mask.sum())
        # Ensure NA doesn't double-count DK if a value matched both patterns
        na_count = int((na_mask & ~dk_mask).sum())

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for display (counts as ints; mean to 2 decimals) ----
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # ---- Save as three 6-column blocks (presentation similar to published table) ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = formatted.loc[:, cols]
            # Ensure row labels are visible as 'Attitude' stub
            block_df = block_df.copy()
            block_df.index.name = "Attitude"
            f.write(block_df.to_string())
            f.write("\n\n")

    return table