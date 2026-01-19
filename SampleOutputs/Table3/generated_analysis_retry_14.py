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

    def _series_to_string_upper(s):
        # robust conversion: keep NAs, normalize whitespace/case
        return (
            s.astype("string")
            .str.strip()
            .str.upper()
        )

    def _parse_numeric(s):
        # Handles floats/ints/strings; non-numeric -> NaN
        return pd.to_numeric(s, errors="coerce")

    def _compute_valid_and_missing_masks(raw_series):
        """
        Returns:
          valid_num: numeric series containing only 1..5 (else NaN)
          dk_mask: boolean mask for "Don't know much about it"
          na_mask: boolean mask for "No answer"
        """
        s_str_up = _series_to_string_upper(raw_series)

        # Detect explicit GSS-style missing tokens like "[NA(d)]", "NA(d)", etc.
        # Also allow variants with spaces.
        dk_mask = s_str_up.str.contains(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", regex=True, na=False)
        na_mask = s_str_up.str.contains(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", regex=True, na=False)

        x = _parse_numeric(raw_series)

        # Valid substantive responses: 1..5
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If explicit NA(d)/NA(n) tokens were found, use them.
        if dk_mask.any() or na_mask.any():
            return valid_num, dk_mask, na_mask

        # Otherwise infer DK/NA from common GSS numeric codes for these items (typically 8=DK, 9=NA).
        # Keep logic conservative: only map codes that clearly appear.
        non_substantive = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
        if non_substantive.any():
            # Commonly used:
            # 8 = don't know, 9 = no answer; sometimes 0/98/99 appear in other extracts.
            # We'll prioritize (8,9), else fall back to the two most frequent non-1..5 codes.
            present_codes = set(pd.unique(x.loc[non_substantive].dropna()).tolist())

            if 8 in present_codes:
                dk_mask = x.eq(8)
            if 9 in present_codes:
                na_mask = x.eq(9)

            # If still not found, fall back to frequency-based inference:
            if (not dk_mask.any()) and (not na_mask.any()):
                vc = x.loc[non_substantive].value_counts(dropna=True)
                top_codes = list(vc.index[:2])

                if len(top_codes) >= 1:
                    # DK is usually more frequent than NA
                    dk_code = top_codes[0]
                    dk_mask = x.eq(dk_code) & non_substantive
                if len(top_codes) >= 2:
                    na_code = top_codes[1]
                    na_mask = x.eq(na_code) & non_substantive

        return valid_num, dk_mask, na_mask

    # Build numeric table first (counts as ints, mean as float)
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _compute_valid_and_missing_masks(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        # prevent double counting if any odd encoding triggers both masks
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

    # Format for display: counts as integers, mean as 2 decimals
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    # Save as three 6-column blocks (like published layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

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