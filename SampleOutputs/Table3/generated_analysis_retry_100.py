def run_analysis(data_source):
    import os
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

    # -----------------------
    # Helpers: parse explicit DK/NA codes if present
    # -----------------------
    def _to_str_upper(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _detect_dk_na(raw: pd.Series):
        """
        Detect explicit DK vs NA encodings if present in the CSV.
        Supports common GSS-style encodings:
          - "[NA(d)]" / "NA(d)" / "NA(D)"
          - "[NA(n)]" / "NA(n)" / "NA(N)"
          - literal strings: "DON'T KNOW", "DONT KNOW", "NO ANSWER", "NA"
        Returns boolean masks (dk_mask, na_mask).
        """
        s = _to_str_upper(raw)

        dk_mask = (
            s.str.contains(r"\[NA\(D\)\]", regex=True, na=False)
            | s.str.fullmatch(r"NA\(D\)", na=False)
            | s.str.contains(r"\bDONT\s*KNOW\b|\bDON'?T\s*KNOW\b", regex=True, na=False)
        )

        na_mask = (
            s.str.contains(r"\[NA\(N\)\]", regex=True, na=False)
            | s.str.fullmatch(r"NA\(N\)", na=False)
            | s.str.contains(r"\bNO\s*ANSWER\b", regex=True, na=False)
        )

        return dk_mask, na_mask

    def _tabulate_one(raw: pd.Series, varname: str):
        """
        Tabulate counts for codes 1..5 plus separate DK and NA, and compute mean on 1..5.

        IMPORTANT:
        - If there are any missing/non-1..5 values that cannot be classified as DK or NA
          (e.g., plain NaN because the export collapsed missing categories), this function
          raises a clear error instead of guessing/splitting.
        """
        # numeric parse for substantive codes
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit DK/NA detection (string-based)
        dk_mask, na_mask = _detect_dk_na(raw)

        # Anything non-1..5 that isn't explicitly DK/NA is "unclassified missing"
        non_1_5 = ~x.isin([1, 2, 3, 4, 5])  # includes NaN in x as False; handle separately below
        nan_mask = x.isna()
        missing_or_non_1_5 = nan_mask | non_1_5

        unclassified = missing_or_non_1_5 & ~(dk_mask | na_mask)

        unclassified_n = int(unclassified.sum())
        if unclassified_n > 0:
            # show a few examples (raw values) to help re-export debugging
            examples = raw.loc[unclassified].head(10).tolist()
            raise ValueError(
                f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                f"this CSV export does not preserve distinguishable DK/NA categories. "
                f"Unclassified missing/non-1..5 count={unclassified_n} (of total rows={len(raw)}). "
                f"Detected DK={int(dk_mask.sum())}, NA={int(na_mask.sum())}. "
                f"Re-export data preserving '[NA(d)]' and '[NA(n)]' (or distinct numeric codes). "
                f"Example unclassified raw values: {examples}"
            )

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

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
    # Save as human-readable text (3 blocks of 6 genres)
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