def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to 1993 only
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ---- Variables (Table 3) ----
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

    # ---- Helpers: detect explicit GSS missing codes if present ----
    def _missing_type_masks(series: pd.Series):
        """
        Returns masks for:
          - dk: [NA(d)]
          - na: [NA(n)]
          - other_missing: any other [NA(...)] OR numeric codes outside 1..5 OR actual NaN
        """
        s = series

        # String view (keeps <NA>)
        s_str = s.astype("string").str.strip()

        # Any explicit [NA(x)] tags
        s_up = s_str.str.upper()

        dk = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        any_bracket_na = s_up.str.contains(r"\[NA\([A-Z]\)\]|\bNA\([A-Z]\)\b", regex=True, na=False)

        # Numeric parse
        x = pd.to_numeric(s_str, errors="coerce")
        valid = x.isin([1, 2, 3, 4, 5])

        # Other missing:
        # - explicit [NA(...)] not dk/na
        # - numeric codes outside 1..5
        # - true NaN / blanks
        other_bracket = any_bracket_na & (~dk) & (~na)
        other_numeric = x.notna() & (~valid)
        blanks = x.isna() & (~any_bracket_na)  # plain NaN/blank in file

        other_missing = other_bracket | other_numeric | blanks
        return dk, na, other_missing, x

    def _split_missing_or_raise(series: pd.Series, varname: str):
        """
        Compute DK and NA counts strictly from explicit codes.
        If explicit DK/NA codes are not present, raise a clear error.
        """
        dk_mask, na_mask, other_missing_mask, x = _missing_type_masks(series)

        # Explicit presence check: must have ability to separate DK vs NA
        if (dk_mask.sum() == 0) and (na_mask.sum() == 0):
            # If there are any missings, we cannot split them without explicit tags.
            # Per user requirement, we must compute from raw data, not hard-code.
            # So we fail fast with an informative error.
            if other_missing_mask.sum() > 0:
                raise ValueError(
                    f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                    f"dataset does not contain explicit NA(d)/NA(n) codes; missing values are present but unsplittable."
                )
            # No missings at all
            return dk_mask, na_mask, x

        # If one is present but the other is not, still proceed (counts for absent one are 0).
        # Any other missing remains unallocated (not shown in table per spec).
        return dk_mask, na_mask, x

    # ---- Build table (numeric) ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        dk_mask, na_mask, x = _split_missing_or_raise(raw, var)

        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

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
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for human-readable output: counts as ints, mean to 2 decimals ----
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # ---- Save as three 6-column blocks ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
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