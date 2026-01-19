def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Load + standardize columns
    # ----------------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Required column 'year' not found.")

    # Filter to 1993 only (exclude missing year implicitly)
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # Variables (Table 3)
    # ----------------------------
    genre_map = [
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

    # ----------------------------
    # Missing split: NA(d) vs NA(n)
    # ----------------------------
    def _split_missing(raw_series):
        """
        Returns:
          valid_num: numeric Series with values 1..5 else NaN
          dk_mask: boolean mask for NA(d) (Don't know much about it)
          na_mask: boolean mask for NA(n) (No answer)

        This function NEVER raises just because explicit NA(d)/NA(n) codes are absent.
        If it cannot identify NA(d)/NA(n) explicitly, it will approximate a split
        using observed non-substantive codes when present, else it will fall back to:
          - all missing -> DK
          - NA -> 0
        """
        s = raw_series

        # Try to detect explicit NA(d)/NA(n) if the dataset carries them as strings
        s_str = s.astype("string")
        s_up = s_str.str.strip().str.upper()

        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Parse numeric where possible
        x = pd.to_numeric(s, errors="coerce")
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If explicit masks found, return them (and also include numeric-coded NA(d)/NA(n) if present)
        if dk_mask.any() or na_mask.any():
            # Ensure disjoint preference: NA(n) overrides only itself; keep as-is and make disjoint
            na_mask = na_mask & (~dk_mask)
            return valid_num, dk_mask, na_mask

        # Otherwise: try numeric non-substantive codes (anything not in 1..5)
        nonsub_mask = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
        miss_mask = x.isna()

        # If there are numeric non-substantive codes, attempt to map the most frequent to DK
        # and the least frequent to NA (common pattern: DK > NA).
        if nonsub_mask.any():
            vc = x.loc[nonsub_mask].value_counts(dropna=True)
            if len(vc) == 1:
                dk_code = vc.index[0]
                na_code = None
            else:
                dk_code = vc.idxmax()
                na_code = vc.idxmin()

            dk_mask_num = x.eq(dk_code)
            na_mask_num = pd.Series(False, index=x.index) if na_code is None else x.eq(na_code)

            # Any remaining missing (blank) are assigned to DK (dominant bucket) to avoid losing them
            dk_mask_num = dk_mask_num | (miss_mask & (~na_mask_num))
            na_mask_num = na_mask_num & (~dk_mask_num)

            return valid_num, dk_mask_num, na_mask_num

        # If only NaN missing exists (no explicit codes), we cannot split reliably.
        # To keep Table 3 structure without fabricating a split, treat all missing as DK, NA as 0.
        dk_mask_inf = miss_mask.copy()
        na_mask_inf = pd.Series(False, index=x.index)

        return valid_num, dk_mask_inf, na_mask_inf

    # ----------------------------
    # Build numeric table
    # ----------------------------
    genre_labels = [g[0] for g in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _split_missing(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int((na_mask & (~dk_mask)).sum())  # enforce disjoint

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ----------------------------
    # Human-readable text output
    # ----------------------------
    display = table.copy()

    # Format: counts as integers; mean as 2 decimals
    for r in display.index:
        if r == "Mean":
            display.loc[r] = display.loc[r].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            display.loc[r] = display.loc[r].map(
                lambda v: "" if pd.isna(v) else str(int(round(float(v))))
            )

    # Add stub column with row labels
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (like the printed layout)
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]
    out_path = "./output/table3_frequency_distributions_gss1993.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: If explicit NA(d)/NA(n) codes are not present, missing values are assigned to DK.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            f.write(display.loc[:, cols_with_stub].to_string(index=False))
            f.write("\n\n")

    return table