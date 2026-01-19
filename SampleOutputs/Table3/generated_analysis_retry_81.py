def run_analysis(data_source):
    import os
    import re
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + filter to 1993
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

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

    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

    # -----------------------
    # Helpers: explicit NA(d) / NA(n) detection if present as strings
    # -----------------------
    def _norm_str_series(s):
        # Keep string values for token detection; leave NaN as <NA>
        return s.astype("string").str.strip()

    def _token_mask(s_str, patterns):
        # patterns are regexes; match case-insensitive
        mask = pd.Series(False, index=s_str.index)
        for pat in patterns:
            mask = mask | s_str.str.contains(pat, case=False, regex=True, na=False)
        return mask

    dk_patterns = [
        r"\[?\s*NA\s*\(\s*d\s*\)\s*\]?",  # [NA(d)] or NA(d)
        r"\bDK\b",
        r"DON'?T\s+KNOW",
        r"DONT\s+KNOW",
    ]
    na_patterns = [
        r"\[?\s*NA\s*\(\s*n\s*\)\s*\]?",  # [NA(n)] or NA(n)
        r"\bNO\s+ANSWER\b",
        r"\bNA\b",
    ]

    def _extract_counts_and_mean(raw):
        # raw: original series (may be numeric, may be strings, may have NaN)
        s_str = _norm_str_series(raw)

        # Explicit coded missing (only works if export preserved them)
        dk_mask_exp = _token_mask(s_str, dk_patterns)
        na_mask_exp = _token_mask(s_str, na_patterns)

        # Numeric parse for 1..5
        x = pd.to_numeric(s_str, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Anything that is not a valid 1..5 is "missing pool"
        missing_pool = valid.isna()

        # If explicit DK/NA are present, use them; otherwise, we cannot separate DK vs NA.
        exp_any = int((dk_mask_exp | na_mask_exp).sum())
        if exp_any > 0:
            # Any remaining missing (not labeled DK/NA) will be folded into "No answer"
            # (conservative; does not affect mean)
            other_missing = missing_pool & ~(dk_mask_exp | na_mask_exp)
            dk_n = int(dk_mask_exp.sum())
            na_n = int((na_mask_exp | other_missing).sum())
            mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
            return counts_1_5, dk_n, na_n, mean_val

        # No explicit categories preserved: attempt numeric-code based split if present
        # Common in some exports: DK=8, NA=9 (or 98/99)
        # We only apply this if those codes actually appear.
        dk_numeric_codes = {8, 98}
        na_numeric_codes = {9, 99}

        dk_mask_num = x.isin(list(dk_numeric_codes))
        na_mask_num = x.isin(list(na_numeric_codes))
        if int((dk_mask_num | na_mask_num).sum()) > 0:
            other_missing = missing_pool & ~(dk_mask_num | na_mask_num)
            dk_n = int(dk_mask_num.sum())
            na_n = int((na_mask_num | other_missing).sum())
            mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
            return counts_1_5, dk_n, na_n, mean_val

        # Still cannot separate: the CSV collapsed distinct missing types into generic NaN
        # => no way to compute the two rows from raw data alone.
        total_unclass = int(missing_pool.sum())
        if total_unclass > 0:
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                "this CSV export does not preserve distinguishable missing categories "
                "(e.g., '[NA(d)]'/'[NA(n)]', DK/No-answer strings, or numeric codes like 8/9 or 98/99). "
                f"Example missing/non-1..5 count in this variable: {total_unclass}. "
                "Re-export data preserving DK vs No-answer codes."
            )

        # No missing at all
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, 0, 0, mean_val

    # -----------------------
    # Build table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _extract_counts_and_mean(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Save human-readable text (3 blocks of 6, like the paper layout)
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
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