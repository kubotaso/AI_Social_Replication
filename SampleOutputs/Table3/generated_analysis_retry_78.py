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
    # Missing code detection
    # -----------------------
    # We can only compute DK vs No-answer if the export preserves distinct codes.
    # Acceptable encodings:
    #   - string tokens containing NA(d) vs NA(n) (case-insensitive, with/without brackets)
    #   - numeric codes 8/9 (common in some GSS extracts) interpreted as DK/NA respectively
    #
    # If the CSV has only generic NaN for both, they are not distinguishable and we fail
    # (to avoid the prior bug where they were silently collapsed).
    def _series_tokens(s):
        s = s.astype("string")
        return s.str.strip().str.upper()

    def _explicit_masks(raw):
        tok = _series_tokens(raw)

        # String-coded missing
        dk_str = tok.str.contains(r"\[?NA\(D\)\]?", regex=True, na=False)
        na_str = tok.str.contains(r"\[?NA\(N\)\]?", regex=True, na=False)

        # Numeric-coded missing (8/9)
        x = pd.to_numeric(raw, errors="coerce")
        dk_num = x.eq(8)
        na_num = x.eq(9)

        dk = dk_str | dk_num
        na = na_str | na_num
        return dk, na

    def _tabulate_one(raw, varname):
        x = pd.to_numeric(raw, errors="coerce")

        # Substantive valid responses
        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        # Explicit DK/NA masks
        dk_mask, na_mask = _explicit_masks(raw)

        # Anything not valid and not explicitly DK/NA is "unknown missing"
        unknown_missing = (~valid_mask) & x.notna() & (~dk_mask) & (~na_mask)
        # Also consider true NaNs from parsing; they are unknown unless tagged by string tokens (already handled)
        unknown_missing = unknown_missing | (x.isna() & (~dk_mask) & (~na_mask) & (~valid_mask))

        # We allow unknown_missing only if it's zero; otherwise we cannot split DK vs NA
        unk_n = int(unknown_missing.sum())
        if unk_n != 0:
            # Provide compact diagnostics
            total_nonvalid = int((~valid_mask).sum())
            raise ValueError(
                f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                f"this CSV export does not preserve distinguishable DK/NA categories. "
                f"Found {unk_n} missing/non-1..5 values that are not classifiable (out of {total_nonvalid} non-1..5). "
                f"Re-export data preserving codes like '[NA(d)]' and '[NA(n)]' or numeric 8/9."
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
    # Build numeric table
    # -----------------------
    out_cols = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype="float64")

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
    # Format for text output (counts as ints, mean 2 decimals)
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

    # -----------------------
    # Save as 3 blocks of 6 genres (paper-style layout)
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [out_cols[i : i + 6] for i in range(0, len(out_cols), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table