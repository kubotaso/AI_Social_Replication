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

    # Filter: 1993 only
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
        "(M) Don\u2019t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # --- Helpers to detect distinct missing codes if present in the CSV export ---
    # We will ONLY split DK vs NA if we can reliably identify them from raw values.
    def _as_clean_string(s: pd.Series) -> pd.Series:
        # Keep <NA> for missing; only clean non-missing to preserve NA-ness
        return s.astype("string").str.strip().str.upper()

    def _detect_missing_masks(raw: pd.Series):
        """
        Returns: valid_numeric (float series with 1..5 else NaN),
                 dk_mask (bool),
                 na_mask (bool),
                 other_missing_mask (bool)  # missing/unusable but not classified as DK/NA
        """
        # Numeric parse for valid codes
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # String-based detection for special NA encodings (if preserved)
        s_up = _as_clean_string(raw)

        # Common GSS extract encodings (strings)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b|\bDON['’]?T\s+KNOW\b|\bDONT\s+KNOW\b|\bDK\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b|\bNO\s+ANSWER\b|\bNA\b", regex=True, na=False)

        # Some exports may use explicit negative numeric codes; treat -1/-2 style as candidates
        # NOTE: We still cannot split DK vs NA unless two distinct codes are observable and mapped.
        # Here we only classify if raw contains recognizable strings; numeric special codes remain "other".
        other_missing = valid.isna() & ~(dk_mask | na_mask)

        return valid, dk_mask, na_mask, other_missing

    # Ensure required columns exist
    missing_vars = [var for _, var in genre_map if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required genre variables in data: {missing_vars}")

    # Build the table
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    # We will compute counts for 1..5 always.
    # For DK/NA rows: only fill if the dataset preserves distinguishable DK vs NA codes/labels.
    # Otherwise, we set them to 0 and report the unsplittable missing count separately in the output file.
    # (The mean is computed on 1..5 only, as required.)
    unsplittable_missing = {}

    for genre_label, var in genre_map:
        raw = df[var]

        valid, dk_mask, na_mask, other_missing = _detect_missing_masks(raw)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        other_n = int(other_missing.sum())

        # If DK/NA are not preserved, dk_n and na_n will be 0 while other_n > 0.
        # We do NOT guess/split; we record the unsplittable count for transparency.
        if (dk_n + na_n) == 0 and other_n > 0:
            unsplittable_missing[genre_label] = other_n

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don\u2019t know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        table.loc["Mean", genre_label] = mean_val

    # Format for human-readable text output
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (paper layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: '(M) Don’t know much about it' and '(M) No answer' are only counted if the CSV preserves distinct codes/labels.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

        if unsplittable_missing:
            f.write("Unsplit Missing Summary (CSV does not preserve DK vs No answer; counts below are missing/non-1..5 values not classifiable into the two M rows):\n")
            for g in genre_labels:
                if g in unsplittable_missing:
                    f.write(f"  {g}: {unsplittable_missing[g]}\n")

    return table