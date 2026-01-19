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

    # Filter to 1993 only
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

    # Ensure required vars exist (case-insensitive already standardized)
    missing_vars = [v for _, v in genre_map if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Required genre variable(s) not found in data: {missing_vars}")

    # -----------------------
    # Missing-code detection
    # -----------------------
    def _as_clean_string(s: pd.Series) -> pd.Series:
        # Use pandas "string" dtype to preserve <NA>; normalize whitespace/case
        return s.astype("string").str.strip().str.upper()

    def _count_missing_buckets(series: pd.Series):
        """
        Returns:
          counts_1_5: pd.Series indexed 1..5
          dk_n: int  (NA(d) only)
          na_n: int  (NA(n) only)
          mean_val: float (mean on valid 1..5)
        Rules:
          - Frequencies for 1..5 counted on numeric values exactly in {1..5}
          - DK = explicit NA(d) encodings only (string-based)
          - No answer = explicit NA(n) encodings only (string-based)
          - Any other missing / invalid codes are treated as missing for mean,
            but not displayed (because Table 3 shows only those two M rows).
        """
        # Numeric parse for valid 1..5
        x = pd.to_numeric(series, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # Explicit DK/NA via string patterns (works if dataset preserved them)
        s_up = _as_clean_string(series)

        # Accept a few common representations
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())

        # If explicit DK/NA codes are not present, we cannot split missing correctly.
        # The paper table requires the split, so fail fast with a clear error.
        if dk_n == 0 and na_n == 0:
            # If there are no missing/invalid values at all, return zeros (rare).
            non_1_5 = int((~x.isin([1, 2, 3, 4, 5])).sum()) if x.notna().any() else int(series.isna().sum())
            if non_1_5 == 0:
                return counts_1_5, 0, 0, mean_val
            raise ValueError(
                "Cannot compute separate '(M) Don’t know much about it' vs '(M) No answer' counts: "
                "dataset does not contain explicit NA(d)/NA(n) codes (or equivalent string labels)."
            )

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table (numeric)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _count_missing_buckets(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable output
    # -----------------------
    formatted = table.copy()
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
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table