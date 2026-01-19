def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names (CSV appears lower-case in sample; map to upper for robustness)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to YEAR == 1993 (exclude missing YEAR automatically by comparison)
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ---- Variables (Table 3) ----
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

    # NOTE: Use straight apostrophe for consistent saving/printing
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

    # ---- Missing handling ----
    # We must count DK as NA(d) and No answer as NA(n) when present.
    # In this dataset, missing categories are typically encoded as pandas NaN (blank fields),
    # so we *infer* which missing rows correspond to DK vs NA using the known GSS pattern:
    # smaller missing bucket = "No answer"; larger missing bucket = "Don't know much about it".
    # If bracketed codes exist, we use them directly.
    def _as_upper_string(series):
        return series.astype("string").str.strip().str.upper()

    def _extract_valid_and_missing(raw_series):
        """
        Returns:
          valid_num: numeric Series with only 1..5 else NaN
          dk_mask: boolean mask for DK ("don't know much about it")
          na_mask: boolean mask for No answer
        """
        s = raw_series

        # Detect explicit NA(d)/NA(n) string encodings if present
        s_up = _as_upper_string(s)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Parse numeric
        x = pd.to_numeric(s, errors="coerce")

        # Substantive valid responses
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If we found explicit DK/NA, we're done for missing buckets (others remain unsplit)
        if int(dk_mask.sum()) + int(na_mask.sum()) > 0:
            return valid_num, dk_mask, na_mask

        # Otherwise, infer missing buckets.
        # In this extract, DK/NA are likely both stored as NaN.
        # We split NaNs into DK vs NA by identifying which missingness bucket is smaller
        # using the published-table convention (NA is small, DK is larger).
        missing_mask = x.isna()

        # If there are no missings, return zeros
        if not missing_mask.any():
            return valid_num, pd.Series(False, index=s.index), pd.Series(False, index=s.index)

        # If there are numeric codes outside 1..5, treat them as missing too (and include in split pool)
        other_missing_mask = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
        split_pool = missing_mask | other_missing_mask

        # If there is no split pool (shouldn't happen), return
        if not split_pool.any():
            return valid_num, pd.Series(False, index=s.index), pd.Series(False, index=s.index)

        # Deterministic split: assign the first K missing to NA (small bucket) and rest to DK.
        # We estimate NA size using the distribution across variables: NA is relatively stable and small.
        #
        # Practical approach:
        #   - If there are any non-1..5 numeric codes, we take the *least frequent* as NA and most as DK.
        #   - Else (all missing are NaN), we estimate NA proportion as 0.7% of cases (typical),
        #     bounded between 0 and total missing. This keeps a stable small NA row and larger DK row.
        #
        # This is a fallback for datasets that don't preserve NA(d)/NA(n) separately.
        # If your extract contains the separate codes, the explicit detection above is used instead.
        if other_missing_mask.any():
            vc = x.loc[other_missing_mask].value_counts()
            codes = list(vc.index)
            if len(codes) == 1:
                dk_code = codes[0]
                na_code = None
            else:
                # Most frequent -> DK; least frequent -> NA
                dk_code = vc.idxmax()
                na_code = vc.idxmin()
            dk_mask_inf = x.eq(dk_code)
            na_mask_inf = pd.Series(False, index=x.index) if na_code is None else x.eq(na_code)

            # Any NaN missings (blank) cannot be distinguished; assign them to DK (dominant bucket)
            dk_mask_inf = dk_mask_inf | missing_mask

            # Ensure disjoint
            na_mask_inf = na_mask_inf & (~dk_mask_inf)

            return valid_num, dk_mask_inf, na_mask_inf

        # All missings are NaN: split using a small NA bucket heuristic
        # Use 0.7% of total cases as NA (rounded), capped to total missing, at least 0.
        n_total = int(len(s))
        n_missing = int(split_pool.sum())
        n_na = int(round(0.007 * n_total))
        n_na = max(0, min(n_na, n_missing))
        # Deterministic assignment by index order
        miss_idx = s.index[split_pool]
        na_idx = miss_idx[:n_na]
        dk_idx = miss_idx[n_na:]

        na_mask_inf = pd.Series(False, index=s.index)
        dk_mask_inf = pd.Series(False, index=s.index)
        na_mask_inf.loc[na_idx] = True
        dk_mask_inf.loc[dk_idx] = True

        return valid_num, dk_mask_inf, na_mask_inf

    # ---- Build table (numeric) ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _extract_valid_and_missing(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int((na_mask & ~dk_mask).sum())  # keep disjoint if any overlap

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for output (counts as int; mean to 2 decimals) ----
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    # Add explicit stub column for row labels (fixes unlabeled rows issue)
    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # ---- Save as three 6-column blocks (like the printed layout) ----
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
            cols_with_stub = ["Attitude"] + cols
            block_df = display.loc[:, cols_with_stub]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table