def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Load + standardize columns
    # ----------------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

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

    # -------------------------------------------------------
    # Missing handling:
    # - Substantive: codes 1..5
    # - DK row: NA(d) (or equivalent strings)
    # - NA row: NA(n) (or equivalent strings)
    # - Other missing stays missing (excluded from mean; not shown in M rows)
    # -------------------------------------------------------
    def _as_string_upper(s: pd.Series) -> pd.Series:
        # Keep <NA> safely; uppercase for robust matching
        return s.astype("string").str.strip().str.upper()

    def _extract_valid_dk_na(raw_series: pd.Series):
        """
        Returns:
          valid_num: numeric series with only 1..5, else NaN
          dk_mask: boolean mask for [NA(d)] (don't know much about it)
          na_mask: boolean mask for [NA(n)] (no answer)
        """
        s_up = _as_string_upper(raw_series)

        # Detect bracketed NA codes in string exports
        # Accept forms: "[NA(d)]", "NA(d)", "NA(D)"
        dk_mask = s_up.str.contains(r"NA\(\s*D\s*\)", regex=True, na=False)
        na_mask = s_up.str.contains(r"NA\(\s*N\s*\)", regex=True, na=False)

        # Numeric parsing (handles floats and numeric strings)
        x = pd.to_numeric(raw_series, errors="coerce")

        # Substantive valid responses
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If data are numeric-only with special codes for DK/NA,
        # try to infer them from the most common non-1..5 numeric codes.
        # (Only do this when no explicit NA(d)/NA(n) strings are present.)
        if (dk_mask.sum() + na_mask.sum()) == 0:
            non_substantive = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
            if non_substantive.any():
                vc = x.loc[non_substantive].value_counts(dropna=True)
                top = list(vc.index[:2])

                dk_code = top[0] if len(top) >= 1 else None
                na_code = top[1] if len(top) >= 2 else None

                # Heuristic: if two codes exist, the more frequent is DK
                if dk_code is not None and na_code is not None:
                    if vc.loc[na_code] > vc.loc[dk_code]:
                        dk_code, na_code = na_code, dk_code

                if dk_code is not None:
                    dk_mask = dk_mask | x.eq(dk_code)
                if na_code is not None:
                    na_mask = na_mask | x.eq(na_code)

        # Ensure no overlap
        na_mask = na_mask & (~dk_mask)

        return valid_num, dk_mask, na_mask

    # ----------------------------
    # Build Table 3
    # ----------------------------
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        valid_num, dk_mask, na_mask = _extract_valid_dk_na(raw)

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

    # ----------------------------
    # Formatting for text output
    # - Counts as integers
    # - Mean to 2 decimals
    # ----------------------------
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else str(int(round(float(v))))
            )

    # ----------------------------
    # Save as three 6-column blocks
    # ----------------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            block_df = formatted.loc[:, cols].copy()
            block_df.index.name = "Attitude"
            f.write(f"Block {bi}:\n")
            f.write(block_df.to_string())
            f.write("\n\n")

    return table