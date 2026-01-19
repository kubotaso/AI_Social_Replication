def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + standardize
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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required variable not found: {v}")

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
    # Helpers: detect DK vs NA robustly
    # -----------------------
    def _to_str_series(x):
        # Keep NA as <NA> (pandas StringDtype)
        return x.astype("string").str.strip()

    def _explicit_missing_masks(raw):
        """
        Return boolean masks for DK and NA using common GSS encodings:
        - literal "[NA(d)]", "[NA(n)]" (any case)
        - "NA(d)" / "NA(n)" (any case)
        - textual labels containing 'DON'T KNOW', 'DONT KNOW', 'DK'
        - textual labels containing 'NO ANSWER'
        - numeric codes 8/9, 98/99 (if present)
        """
        s = _to_str_series(raw)
        s_up = s.str.upper()

        # token-based (strings)
        dk_mask = (
            s_up.str.contains(r"\[?NA\(D\)\]?", regex=True, na=False)
            | s_up.str.contains(r"\bDONT\s*KNOW\b|\bDON'T\s*KNOW\b|\bDON’T\s*KNOW\b", regex=True, na=False)
            | s_up.str.fullmatch(r"\s*DK\s*", na=False)
        )
        na_mask = (
            s_up.str.contains(r"\[?NA\(N\)\]?", regex=True, na=False)
            | s_up.str.contains(r"\bNO\s*ANSWER\b", regex=True, na=False)
        )

        # numeric-based (if raw is numeric or numeric-like)
        x = pd.to_numeric(raw, errors="coerce")
        dk_mask = dk_mask | x.isin([8, 98])
        na_mask = na_mask | x.isin([9, 99])

        return dk_mask, na_mask

    def _tabulate_one(raw, varname):
        """
        Compute counts for 1..5, DK, NA, and mean (1..5 only).

        Critical rule:
        - If the file does not preserve DK vs NA and there are non-1..5 missings,
          we cannot infer the split without copying from paper. In that case raise.
        """
        x = pd.to_numeric(raw, errors="coerce")
        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        counts_1_5 = (
            x.where(valid_mask)
            .value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_mask, na_mask = _explicit_missing_masks(raw)

        # Anything not 1..5 is "missing/non-1..5" pool
        non_1_5_mask = ~valid_mask

        # Count explicitly classified DK/NA among non-1..5
        dk_n = int((dk_mask & non_1_5_mask).sum())
        na_n = int((na_mask & non_1_5_mask).sum())

        # Unclassified missings: these prevent a valid DK vs NA split
        unclassified_n = int((non_1_5_mask & ~(dk_mask | na_mask)).sum())

        if unclassified_n > 0:
            # If there are NO non-1..5 values, it's fine; otherwise we must fail fast.
            # Here unclassified_n > 0 implies there are non-1..5 values we can't split.
            examples = raw[non_1_5_mask & ~(dk_mask | na_mask)].head(10).tolist()
            raise ValueError(
                f"Cannot compute separate DK vs No answer for {varname}: "
                f"found {unclassified_n} non-1..5 values with no distinguishable DK/NA coding. "
                f"Examples: {examples}. Re-export data preserving '[NA(d)]' and '[NA(n)]' "
                f"(or distinct numeric codes like 8/9 or 98/99)."
            )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build Table 3
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

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