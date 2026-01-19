def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + standardize cols
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # 1993 only (exclude missing YEAR automatically via eq)
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # -----------------------
    # Variables / labels
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
        "(M) Don’t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Missing-code handling
    # -----------------------
    def _as_clean_string(s):
        return s.astype("string").str.strip()

    def _parse_na_bracket_code(s):
        """
        Detects GSS-style bracketed missing like "[NA(d)]", "[NA(n)]".
        Returns uppercased string series.
        """
        ss = _as_clean_string(s).str.upper()
        return ss

    def _detect_numeric_missing_codes(x_num):
        """
        Returns (dk_code, na_code) if likely present, else (None, None).
        Common in GSS extracts: 8=DK, 9=NA, but we do not hardcode.
        We infer:
          - consider non-1..5 integer-like codes (allow floats that are whole numbers)
          - pick the two most frequent; larger freq -> DK, smaller -> NA
        """
        if x_num is None or x_num.empty:
            return None, None

        # Candidate: numeric, not 1..5
        cand = x_num.dropna()
        cand = cand[~cand.isin([1, 2, 3, 4, 5])]

        if cand.empty:
            return None, None

        # Keep "integer-like" only (e.g., 8.0, 9.0) to avoid stray continuous values
        cand_intlike = cand[cand.apply(lambda v: float(v).is_integer())]
        if cand_intlike.empty:
            return None, None

        vc = cand_intlike.value_counts()
        top = list(vc.index[:2])

        if len(top) == 1:
            return top[0], None

        c1, c2 = top[0], top[1]
        if vc.loc[c1] >= vc.loc[c2]:
            return c1, c2
        return c2, c1

    def _extract_components(raw_series):
        """
        Returns:
          valid_num: numeric series with only 1..5, else NaN
          dk_mask: boolean for DK ("Don't know much about it")
          na_mask: boolean for NA ("No answer")
        """
        s_raw = raw_series

        # String detection for bracket codes
        s_up = _parse_na_bracket_code(s_raw)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]", regex=True, na=False) | s_up.str.contains(r"\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]", regex=True, na=False) | s_up.str.contains(r"\bNA\(N\)\b", regex=True, na=False)

        # Numeric parsing
        x = pd.to_numeric(s_raw, errors="coerce")

        # Valid 1..5 for mean + substantive rows
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If no bracket codes detected, try numeric inference (e.g., 8/9)
        if (dk_mask.sum() + na_mask.sum()) == 0:
            dk_code, na_code = _detect_numeric_missing_codes(x)

            if dk_code is not None:
                dk_mask = x.eq(dk_code)
            if na_code is not None:
                na_mask = x.eq(na_code)

        # Ensure NA does not double-count DK if something matches both (shouldn't, but safe)
        na_mask = na_mask & (~dk_mask)

        return valid_num, dk_mask, na_mask

    # -----------------------
    # Build the Table 3 frame
    # -----------------------
    genre_labels = [g[0] for g in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        valid_num, dk_mask, na_mask = _extract_components(df[var])

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

    # -----------------------
    # Format for file output
    # -----------------------
    formatted = table.copy()

    # counts as ints, mean as 2 decimals
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    # -----------------------
    # Save as 3 blocks of 6
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding all missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = formatted.loc[:, cols].copy()
            block_df.index.name = "Attitude"
            f.write(block_df.to_string())
            f.write("\n\n")

    return table