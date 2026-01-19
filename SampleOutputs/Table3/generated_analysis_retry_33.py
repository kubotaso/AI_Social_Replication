def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + normalize columns
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Keep 1993 only (exclude missing YEAR implicitly)
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

    # Verify variables exist
    missing_vars = [v for _, v in genre_map if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required genre variables in data: {missing_vars}")

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

    # -----------------------
    # Helper: extract NA(d) / NA(n) if present
    # -----------------------
    def _as_upper_string(series: pd.Series) -> pd.Series:
        # Keep <NA> as <NA> in pandas StringDtype; avoid "nan" strings where possible
        return series.astype("string").str.strip().str.upper()

    def _count_explicit_missing(series: pd.Series):
        """
        Returns:
          dk_count, na_count, other_missing_count, valid_numeric_series (float with NaN for non-1..5)
        Notes:
          - Explicit missing detection looks for [NA(d)] and [NA(n)] (case-insensitive).
          - If explicit codes are absent, DK vs NA cannot be separated from the raw data.
        """
        s_up = _as_upper_string(series)

        # Detect explicit strings like "[NA(d)]", "NA(d)", etc.
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        x = pd.to_numeric(series, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Anything that is not valid (1..5) and not classified as explicit DK/NA is "other missing"
        other_missing = valid.isna() & ~(dk_mask | na_mask)

        return int(dk_mask.sum()), int(na_mask.sum()), int(other_missing.sum()), valid, dk_mask, na_mask, other_missing

    # -----------------------
    # Build counts + means
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    # Determine whether explicit NA(d)/NA(n) codes exist anywhere
    explicit_found_anywhere = False
    for _, var in genre_map:
        dk_c, na_c, _, _, _, _, _ = _count_explicit_missing(df[var])
        if (dk_c + na_c) > 0:
            explicit_found_anywhere = True
            break

    if not explicit_found_anywhere:
        raise ValueError(
            "Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts: "
            "dataset does not contain explicit NA(d)/NA(n) codes (or equivalent string labels). "
            "Provide a dataset extract that preserves these distinct missing codes."
        )

    for genre_label, var in genre_map:
        dk_c, na_c, other_miss_c, valid, dk_mask, na_mask, other_missing = _count_explicit_missing(df[var])

        # If some other missing codes exist besides explicit NA(d)/NA(n), we cannot allocate them to DK vs NA
        # without making up rules. Fail fast to avoid producing wrong rows.
        if other_miss_c > 0:
            raise ValueError(
                f"Variable {var} contains {other_miss_c} missing/non-1..5 values that are not explicitly "
                "coded as NA(d) or NA(n). Cannot allocate them to DK vs NA without assumptions."
            )

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_c
        table.loc["(M) No answer", genre_label] = na_c
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # -----------------------
    # Save human-readable text file (3 blocks of 6 genres)
    # -----------------------
    formatted = table.copy()

    # Format: counts as integers, mean as 2 decimals
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing categories.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table