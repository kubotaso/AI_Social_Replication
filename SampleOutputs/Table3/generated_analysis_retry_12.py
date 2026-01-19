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

    # Filter to YEAR == 1993 (drop missing YEAR automatically)
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

    def _as_string_upper(s):
        return s.astype("string").str.strip().str.upper()

    def _extract_dk_na_masks(raw_series):
        """
        Returns:
          valid_num: numeric series with only 1..5, else NaN
          dk_mask: "Don't know much about it" (prefers explicit NA(d); otherwise inferred)
          na_mask: "No answer" (prefers explicit NA(n); otherwise inferred)
        """
        # Parse numeric
        x = pd.to_numeric(raw_series, errors="coerce")

        # Valid substantive responses
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # String-based missing detection (e.g., "[NA(d)]")
        s_up = _as_string_upper(raw_series)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # If explicit bracket codes exist, trust them and do not infer numeric codes
        if (dk_mask | na_mask).any():
            # Ensure NA doesn't double-count DK if any weird overlap occurs
            na_mask = na_mask & (~dk_mask)
            return valid_num, dk_mask.fillna(False), na_mask.fillna(False)

        # Otherwise infer numeric codes for DK/NA from non-1..5 values (if present)
        nonvalid = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
        if not nonvalid.any():
            return valid_num, dk_mask.fillna(False), na_mask.fillna(False)

        vc = x.loc[nonvalid].value_counts(dropna=True)

        # Heuristic: in this instrument DK is much more common than NA.
        # Take the most frequent nonvalid code as DK, second as NA (if any).
        dk_code = vc.index[0]
        na_code = vc.index[1] if len(vc.index) > 1 else None

        dk_mask = x.eq(dk_code) & nonvalid
        na_mask = x.eq(na_code) & nonvalid if na_code is not None else pd.Series(False, index=raw_series.index)

        return valid_num, dk_mask.fillna(False), na_mask.fillna(False)

    # Build numeric table
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        valid_num, dk_mask, na_mask = _extract_dk_na_masks(raw)

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

    # Format for display: counts as integers; mean to 2 decimals
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # Save as three 6-column blocks (published-style layout)
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