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

    # Filter to 1993 only (exclude missing YEAR automatically by equality filter)
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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {v}")

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
    # Helpers: detect DK vs NA
    # -----------------------
    DK_TOKENS = {
        "[NA(D)]", "NA(D)", "NAD", "DK", "DON'T KNOW", "DONT KNOW", "DON’T KNOW",
        "DON'T KNOW MUCH ABOUT IT", "DONT KNOW MUCH ABOUT IT", "DON’T KNOW MUCH ABOUT IT",
        "I DON'T KNOW", "IDK"
    }
    NA_TOKENS = {
        "[NA(N)]", "NA(N)", "NAN", "NO ANSWER", "NOANSWER", "NA", "N/A", "MISSING",
        "NOT ANSWERED", "REFUSED", "[NA(R)]", "NA(R)", "SKIPPED", "[NA(S)]", "NA(S)"
    }

    def _normalize_str_series(s: pd.Series) -> pd.Series:
        # Keep as string; convert missing to <NA> then to np.nan-like handling via pandas ops
        out = s.astype("string")
        out = out.str.strip()
        out = out.str.replace("\u2019", "'", regex=False)  # curly apostrophe -> straight
        out = out.str.upper()
        return out

    def _classify_missing(raw: pd.Series):
        """
        Returns:
          x_valid: numeric series with only 1..5 kept else NaN
          dk_mask: boolean mask for DK/Don't-know
          na_mask: boolean mask for No-answer (includes explicit NA(n), refused, skipped, etc.)
          other_missing_mask: missing/unclassified (NaN or non-1..5 not classified)
        """
        # Numeric parse (handles floats like 1.0..5.0)
        x = pd.to_numeric(raw, errors="coerce")
        x_valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        s_up = _normalize_str_series(raw)

        # Detect explicit NA(d)/NA(n) patterns like "[NA(d)]"
        # Also support any bracketed NA codes: [NA(x)]
        dk_mask = s_up.str.contains(r"\[NA\(\s*D\s*\)\]", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(\s*N\s*\)\]", regex=True, na=False)

        # Also detect common textual labels if present
        dk_mask = dk_mask | s_up.isin(DK_TOKENS)
        na_mask = na_mask | s_up.isin(NA_TOKENS)

        # Numeric DK/NA common encodings (if present in some extracts)
        # We only apply these to values that are not valid 1..5
        nonvalid = ~x.isin([1, 2, 3, 4, 5]) & x.notna()
        # Typical GSS-style: 8/9 or 98/99; also sometimes 0
        dk_mask = dk_mask | (nonvalid & x.isin([8, 98]))
        na_mask = na_mask | (nonvalid & x.isin([9, 99, 0]))

        # Everything else that is nonvalid is "other missing"
        other_missing_mask = x_valid.isna() & ~(dk_mask | na_mask)

        return x_valid, dk_mask, na_mask, other_missing_mask

    def _tabulate_one(raw: pd.Series, varname: str):
        x_valid, dk_mask, na_mask, other_missing = _classify_missing(raw)

        # If there are any unclassified missing values, we cannot split DK vs NA reliably
        # with this CSV export. Fail loudly with actionable error.
        if int(other_missing.sum()) > 0:
            # Provide a small sample of raw problematic values for debugging
            sample_vals = raw.loc[other_missing].head(10).tolist()
            raise ValueError(
                f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                f"this CSV export does not preserve distinguishable DK/NA categories. "
                f"Unclassified missing/non-1..5 count={int(other_missing.sum())} (of total rows={len(raw)}). "
                f"Re-export preserving '[NA(d)]' and '[NA(n)]' (or distinct numeric codes like 8/9, 98/99). "
                f"Example unclassified raw values: {sample_vals}"
            )

        counts_1_5 = (
            x_valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(x_valid.mean(skipna=True)) if x_valid.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table
    # -----------------------
    col_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=col_labels, dtype="float64")

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
    # Save human-readable table (three 6-genre blocks)
    # -----------------------
    display = table.copy()

    # Format: counts as ints; mean rounded to 2 decimals
    for r in display.index:
        if r == "Mean":
            display.loc[r] = display.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            display.loc[r] = display.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display.insert(0, "Attitude", display.index)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [col_labels[i:i + 6] for i in range(0, len(col_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for i, cols in enumerate(blocks, start=1):
            f.write(f"Block {i}:\n")
            f.write(display.loc[:, ["Attitude"] + cols].to_string(index=False))
            f.write("\n\n")

    return table