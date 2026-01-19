def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to 1993 only (exclude missing YEAR automatically)
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

    def _as_string_series(s):
        # robust string conversion preserving missing
        return s.astype("string")

    def _to_numeric(s):
        return pd.to_numeric(s, errors="coerce")

    def _dk_na_masks(raw_series):
        """
        Detect DK/NA using bracketed GSS missing tags if present, else fall back to
        numeric code inference (most common invalid code = DK, second most common = NA).

        Returns:
          valid_num : numeric series with only 1..5, else NaN
          dk_mask   : boolean mask for "Don't know much about it"
          na_mask   : boolean mask for "No answer"
        """
        s_str = _as_string_series(raw_series)
        s_up = s_str.str.upper()

        # Primary: explicit tagged missings (common in some GSS extracts)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        x = _to_numeric(raw_series)

        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If no explicit tags were found, infer DK/NA from numeric out-of-range codes
        # among observations that are numeric but not 1..5.
        has_any_tagged = bool(dk_mask.any() or na_mask.any())
        invalid_numeric = x.notna() & (~x.isin([1, 2, 3, 4, 5]))

        if (not has_any_tagged) and invalid_numeric.any():
            vc = x.loc[invalid_numeric].value_counts(dropna=True)

            # Choose two most frequent invalid numeric codes, if present.
            top = list(vc.index[:2])

            if len(top) >= 1:
                dk_code = top[0]
                dk_mask = x.eq(dk_code) & invalid_numeric
            else:
                dk_mask = pd.Series(False, index=raw_series.index)

            if len(top) >= 2:
                na_code = top[1]
                na_mask = x.eq(na_code) & invalid_numeric
            else:
                na_mask = pd.Series(False, index=raw_series.index)

        # Ensure no overlap
        na_mask = na_mask & (~dk_mask)

        return valid_num, dk_mask, na_mask

    # Build numeric table (counts + mean)
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _dk_na_masks(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # Format for display: counts as integers; mean to 2 decimals with trailing zeros
    formatted = table.copy()

    for idx in formatted.index:
        if idx == "Mean":
            formatted.loc[idx] = formatted.loc[idx].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[idx] = formatted.loc[idx].map(
                lambda v: "" if pd.isna(v) else str(int(v))
            )

    # Write as 3 blocks of 6 columns (layout similar to the published table)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = formatted.loc[:, cols].copy()
            block_df.index.name = "Attitude"
            f.write(block_df.to_string())
            f.write("\n\n")

    return table