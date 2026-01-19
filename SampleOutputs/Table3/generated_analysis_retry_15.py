def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names to uppercase for robust referencing
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to GSS 1993 only
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

    def _normalize_missing_label(x):
        """Convert various missing label strings to a canonical form."""
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.upper() == "NAN":
            return None
        su = s.upper()
        # Common forms: "[NA(d)]", "NA(d)", "[NA(D)]", etc.
        if "NA(" in su:
            # try to extract NA(<code>)
            start = su.find("NA(")
            end = su.find(")", start)
            if end != -1:
                code = su[start + 3 : end].strip()
                return f"NA({code})"
        return s  # leave unchanged

    def _extract_buckets(raw_series):
        """
        Returns:
          valid_num: numeric series where only 1..5 retained else NaN
          dk_mask: boolean mask for Don't know much about it (NA(d))
          na_mask: boolean mask for No answer (NA(n))
        """
        s = raw_series.copy()

        # Build string-normalized series for explicit NA(...) detection
        s_str = s.astype("string")
        norm = s_str.map(_normalize_missing_label)

        dk_mask = norm.eq("NA(D)")
        na_mask = norm.eq("NA(N)")

        # Numeric parse (works if values are numeric strings/floats)
        x = pd.to_numeric(s, errors="coerce")

        # Valid substantive responses
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If explicit NA codes were not present, handle common numeric missing codes (if used)
        # Many GSS extracts use 8 = DK, 9 = NA for 1-5 scales.
        if (dk_mask.sum() + na_mask.sum()) == 0:
            dk_mask = x.eq(8)
            na_mask = x.eq(9)

        # Ensure NA does not double-count DK
        na_mask = na_mask & (~dk_mask)

        return valid_num, dk_mask, na_mask

    # Build numeric table first (counts as floats, mean as float)
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        valid_num, dk_mask, na_mask = _extract_buckets(df[var])

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        table.loc["Mean", genre_label] = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

    # Format for human-readable output (counts as ints; mean to 2 decimals)
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    # Save as three 6-column blocks (like the published layout)
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