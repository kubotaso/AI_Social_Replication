def run_analysis(data_source):
    import os
    import re
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Normalize column names to match mapping
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to GSS 1993
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

    # Ensure required columns exist
    missing_cols = [var for _, var in genre_map if var not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required genre variables: {missing_cols}")

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

    # --- Missing-code parsing ---
    # We MUST compute DK vs NA from raw data, not hardcode.
    # This CSV may encode missing as:
    #  - explicit strings like "[NA(d)]", "[NA(n)]"
    #  - labeled strings containing "DON'T KNOW"/"NO ANSWER"
    #  - numeric codes (varies by extraction)
    # If the extract only has NaN for both DK and NA, it is not distinguishable and cannot be recovered.
    def _to_str_series(s):
        return s.astype("string")

    def _explicit_missing_masks(raw):
        s = _to_str_series(raw).str.strip()

        # tokens like [NA(d)] / NA(d) (case-insensitive)
        s_up = s.str.upper()

        dk = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Also accept common text labels if present
        # (kept conservative to avoid false positives)
        dk = dk | s_up.str.contains(r"\bDON'?T\s+KNOW\b", regex=True, na=False)
        na = na | s_up.str.contains(r"\bNO\s+ANSWER\b", regex=True, na=False)

        return dk, na

    def _detect_numeric_missing_scheme(series_num):
        """
        Try to infer distinct numeric codes for DK vs NA.
        Returns (dk_codes, na_codes) as sets of numbers (floats/ints), possibly empty.
        """
        # Candidate codes often used across GSS-like extracts (not paper values; these are code patterns)
        # We will only accept a candidate if it is present in the data and is outside 1..5.
        candidates = [0, 6, 7, 8, 9, 98, 99, 97, 96, -1, -2, -3]

        present = set(pd.unique(series_num.dropna()))
        present_non_1_5 = {v for v in present if v not in {1, 2, 3, 4, 5}}

        # Heuristic mapping:
        # - if 8/9 present => treat 8=DK, 9=NA
        # - if 98/99 present => treat 98=DK, 99=NA
        # - if 6/7 present => treat 6=DK, 7=NA (common in some recodes)
        # Otherwise, no reliable numeric split.
        def _has(x):
            return x in present_non_1_5

        if _has(8) and _has(9):
            return {8}, {9}
        if _has(98) and _has(99):
            return {98}, {99}
        if _has(6) and _has(7):
            return {6}, {7}

        # Sometimes 0 indicates NA; but DK has some other value (rare). We won't guess.
        _ = candidates  # keep for clarity; no further guessing
        return set(), set()

    def _tabulate_one(raw, varname):
        # Parse numeric values
        x = pd.to_numeric(raw, errors="coerce")

        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        counts_1_5 = (
            pd.Series(valid)
            .value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Try explicit string-coded missing
        dk_exp, na_exp = _explicit_missing_masks(raw)

        # Try numeric-coded missing
        dk_codes, na_codes = _detect_numeric_missing_scheme(x)
        dk_num = pd.Series(False, index=x.index)
        na_num = pd.Series(False, index=x.index)
        if dk_codes:
            dk_num = x.isin(list(dk_codes))
        if na_codes:
            na_num = x.isin(list(na_codes))

        dk_mask = dk_exp | dk_num
        na_mask = na_exp | na_num

        # Any other non-1..5 that isn't classified is "unclassified missing"
        nonvalid_mask = ~valid_mask
        # nonvalid_mask includes NaN too (because NaN is not in 1..5); that's ok for detectability
        unclassified = nonvalid_mask & ~(dk_mask | na_mask)

        # If there are unclassified missing values, we cannot split DK vs NA reliably.
        # This happens when the CSV collapsed both categories to NaN.
        unclassified_n = int(unclassified.sum())

        if unclassified_n > 0:
            # Provide a precise, actionable error (no fabricated counts).
            # Show a few example raw values (often just nan).
            examples = raw.loc[unclassified].head(10).tolist()
            # compact examples
            ex_str = ", ".join([("nan" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in examples])
            raise ValueError(
                "Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts "
                f"for {varname}: this CSV export does not preserve distinguishable missing categories. "
                f"Unclassified missing/non-1..5 count={unclassified_n}. "
                "Re-export data preserving DK vs No-answer codes (e.g., '[NA(d)]' and '[NA(n)]' or distinct numeric codes). "
                f"Example unclassified raw values: {ex_str}"
            )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # Build numeric table
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # Prepare formatted table for printing (counts as ints; mean rounded to 2 decimals)
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Output in 3 blocks of 6 genres (paper-like layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing (M) categories.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table