def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names (file uses lowercase; analysis spec uses uppercase)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter: YEAR == 1993 (exclude missing automatically via comparison)
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

    # --- Missing handling ---
    # This dataset appears numeric with blanks as NaN, but we also support string-coded [NA(...)].
    # Table 3 requires separate rows for:
    #   DK: [NA(d)]  ("Don't know much about it")
    #   NA: [NA(n)]  ("No answer")
    # If the file is numeric-only (DK/NA already converted to NaN), we cannot recover DK vs NA from NaN.
    # In that case, we infer DK/NA numeric codes among non-1..5 values if present; otherwise DK/NA will be 0.
    # (In the provided task, the correct behavior is that DK/NA are encoded as non-1..5, not as plain NaN.)
    def _as_string_upper(s):
        return s.astype("string").str.strip().str.upper()

    def _extract_buckets(raw_series):
        """
        Returns:
            valid_num: numeric series with values 1..5, else NaN
            dk_mask: boolean mask for DK
            na_mask: boolean mask for No answer
        """
        s = raw_series

        # Detect explicit bracket-coded missing
        s_up = _as_string_upper(s)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", na=False, regex=True)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", na=False, regex=True)

        # Numeric parse
        x = pd.to_numeric(s, errors="coerce")

        # Substantive
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If no explicit DK/NA markers, try to infer numeric codes for DK and NA
        # among non-1..5 numeric values. (Common in GSS extracts.)
        non_substantive = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
        if (dk_mask.sum() + na_mask.sum()) == 0 and non_substantive.any():
            vc = x.loc[non_substantive].value_counts()
            codes = list(vc.index[:2])

            dk_code = None
            na_code = None
            if len(codes) == 1:
                dk_code = codes[0]
            elif len(codes) >= 2:
                # larger frequency -> DK; smaller -> NA
                c1, c2 = codes[0], codes[1]
                if vc.loc[c1] >= vc.loc[c2]:
                    dk_code, na_code = c1, c2
                else:
                    dk_code, na_code = c2, c1

            if dk_code is not None:
                dk_mask = dk_mask | x.eq(dk_code)
            if na_code is not None:
                na_mask = na_mask | x.eq(na_code)

        # Ensure NA doesn't double-count DK
        na_mask = na_mask & (~dk_mask)

        return valid_num, dk_mask, na_mask

    # --- Build numeric table (counts as ints, mean as float) ---
    cols = [label for label, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=cols, dtype="float64")

    for label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _extract_buckets(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", label] = counts_1_5.loc[1]
        table.loc["(2) Like it", label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", label] = int(dk_mask.sum())
        table.loc["(M) No answer", label] = int(na_mask.sum())

        table.loc["Mean", label] = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

    # --- Format for display: counts as integers; mean to 2 decimals ---
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else str(int(v))
            )

    # --- Save in three 6-column blocks (like published layout) ---
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [cols[i : i + 6] for i in range(0, len(cols), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, block_cols in enumerate(blocks, start=1):
            block_df = formatted.loc[:, block_cols].copy()
            block_df.index.name = "Attitude"
            f.write(f"Block {bi}:\n")
            f.write(block_df.to_string())
            f.write("\n\n")

    return table