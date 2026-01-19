def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names to match mapping
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
        try:
            return s.astype("string")
        except Exception:
            return s.astype(str)

    def _extract_codes(raw_series):
        """
        Returns:
          valid_num: numeric series with only 1..5 else NaN
          dk_mask: boolean mask for 'Don't know much about it'
          na_mask: boolean mask for 'No answer'
        Notes:
          - Handles bracketed missing like [NA(d)], [NA(n)] (case-insensitive)
          - Handles common numeric encodings (typically 8=DK, 9=NA), plus heuristics if needed
          - Any other missing/refused/skipped/etc. remain excluded from mean and not shown in DK/NA rows
            (Table 3 displays only the two M rows).
        """
        s_str = _as_string_series(raw_series).str.strip()
        s_up = s_str.str.upper()

        # Explicit GSS-style bracket missing codes (most reliable when present)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Numeric parsing
        x = pd.to_numeric(s_str, errors="coerce")

        # Substantive valid responses
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If bracket codes exist, don't try to infer numeric DK/NA beyond that
        if (dk_mask.sum() + na_mask.sum()) > 0:
            return valid_num, dk_mask, na_mask

        # If no explicit bracket codes, handle common numeric missing codes
        # Most common for these items: 8 = DK, 9 = NA (when present).
        dk_mask = dk_mask | x.eq(8)
        na_mask = na_mask | x.eq(9)

        # If still none, attempt a conservative heuristic:
        # among non-1..5 and non-missing numeric codes, pick two most common as DK and NA.
        # Larger frequency -> DK; smaller -> NA.
        non_valid = x.notna() & (~x.isin([1, 2, 3, 4, 5, 8, 9]))
        if (dk_mask.sum() + na_mask.sum()) == 0 and non_valid.any():
            vc = x.loc[non_valid].value_counts(dropna=True)
            top = list(vc.index[:2])
            if len(top) == 1:
                dk_code, na_code = top[0], None
            else:
                c1, c2 = top[0], top[1]
                if vc.loc[c1] >= vc.loc[c2]:
                    dk_code, na_code = c1, c2
                else:
                    dk_code, na_code = c2, c1
            dk_mask = dk_mask | x.eq(dk_code)
            if na_code is not None:
                na_mask = na_mask | x.eq(na_code)

        return valid_num, dk_mask, na_mask

    # Build numeric table
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _extract_codes(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
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

    # Format for display: counts as ints, mean rounded to 2 decimals
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    # Save as three 6-column blocks (presentation similar to published table)
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