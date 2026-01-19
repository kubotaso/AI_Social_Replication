def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + standardize columns
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Keep YEAR==1993 only (drop missing YEAR implicitly)
    df = df.loc[df["YEAR"] == 1993].copy()

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

    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

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

    genre_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    # -----------------------
    # Missing-category parsing (robust to different exports)
    # -----------------------
    def _as_str_upper(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw: pd.Series):
        """
        Attempt to detect distinct DK vs NA in common GSS-style exports:
        - "[NA(d)]", "NA(d)", "DK" for don't-know
        - "[NA(n)]", "NA(n)", "NO ANSWER" for no-answer
        """
        s = _as_str_upper(raw)

        dk_tokens = [
            r"\[NA\(D\)\]",
            r"\bNA\(D\)\b",
            r"\bDON'?T\s*KNOW\b",
            r"\bDONT\s*KNOW\b",
            r"\bDK\b",
            r"\bDON'?T\s*KNOW\s*MUCH\b",
        ]
        na_tokens = [
            r"\[NA\(N\)\]",
            r"\bNA\(N\)\b",
            r"\bNO\s*ANSWER\b",
            r"\bNA\b",  # often used as literal "NA" for no-answer in some extracts
        ]

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        for pat in dk_tokens:
            dk_mask = dk_mask | s.str.contains(pat, regex=True, na=False)
        for pat in na_tokens:
            na_mask = na_mask | s.str.contains(pat, regex=True, na=False)

        # Avoid double-counting: if something matches both, treat as no-answer
        dk_mask = dk_mask & ~na_mask
        return dk_mask, na_mask

    def _tabulate_one(raw: pd.Series, varname: str):
        """
        Returns:
          counts_1_5 (Series indexed 1..5),
          dk_n (int),
          na_n (int),
          mean_val (float)
        """
        # Primary numeric parse
        x = pd.to_numeric(raw, errors="coerce")

        # Substantive valid codes
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # Missing pool: anything not 1..5 (including NaN after coercion)
        miss_pool = valid.isna()

        # Try explicit DK/NA detection from original raw values
        dk_mask_exp, na_mask_exp = _explicit_missing_masks(raw)

        # Also support common numeric missing codings if they exist (do NOT assume; only use if observed)
        # (Some GSS extracts use 8/9 for DK/NA; others use 0/8/9; we only classify if present.)
        # Note: this does not affect 1..5 counts because those are already separated.
        numeric_dk = x.isin([8])   # only counts if 8 is actually present in this column
        numeric_na = x.isin([9])   # only counts if 9 is actually present in this column

        dk_mask = (dk_mask_exp | numeric_dk) & miss_pool
        na_mask = (na_mask_exp | numeric_na) & miss_pool

        unclassified = miss_pool & ~(dk_mask | na_mask)

        # If we cannot classify any missing values, we must fail rather than invent a split.
        # This matches the requirement "Never directly copy-paste numbers from the paper".
        if int(unclassified.sum()) > 0:
            # Provide a compact diagnostic for debugging/re-export.
            examples_n = int(unclassified.sum())
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                f"this CSV export does not preserve distinguishable missing categories. "
                f"Found {examples_n} missing/non-1..5 values that are not classifiable. "
                f"Re-export data preserving DK vs No-answer codes (e.g., '[NA(d)]'/'[NA(n)]', string labels, or distinct numeric codes)."
            )

        return counts_1_5, int(dk_mask.sum()), int(na_mask.sum()), mean_val

    # -----------------------
    # Build table
    # -----------------------
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

    # -----------------------
    # Save human-readable text (3 blocks of 6 genres)
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

    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]
    out_path = "./output/table3_frequency_distributions_gss1993.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table