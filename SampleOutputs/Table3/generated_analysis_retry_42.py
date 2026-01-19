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

    # -----------------------
    # Helpers: detect explicit NA(d)/NA(n) codes
    # -----------------------
    def _to_str_series(s: pd.Series) -> pd.Series:
        # robust string conversion that preserves NA as <NA>
        return s.astype("string")

    def _explicit_missing_masks(raw: pd.Series):
        """
        Return boolean masks for explicit NA(d) and NA(n) encodings, if present.
        Accepts variants like '[NA(d)]', 'NA(d)', case-insensitive, with optional whitespace.
        """
        s = _to_str_series(raw).str.strip().str.upper()

        # Handle values that are truly missing in pandas: these become <NA> and won't match
        dk = s.str.fullmatch(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", na=False)
        na = s.str.fullmatch(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", na=False)
        return dk, na

    def _tabulate_one(raw: pd.Series, varname: str):
        """
        Compute counts for 1..5, DK, NA, and mean (on 1..5 only).
        Requires explicit NA(d)/NA(n) codes to split DK vs NA among missing/non-1..5.
        """
        # numeric parse; keep original for explicit code detection
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]))

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_mask, na_mask = _explicit_missing_masks(raw)

        # Any other non-1..5 values (including blank strings coerced to NaN) are "unspecified missing".
        # Table 3 splits missing into DK vs NA; we can only do that if explicit codes exist.
        other_missing_mask = valid.isna() & ~(dk_mask | na_mask)

        if int(other_missing_mask.sum()) > 0:
            # If there are no explicit codes at all, we can't split.
            if int(dk_mask.sum()) == 0 and int(na_mask.sum()) == 0:
                raise ValueError(
                    f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                    f"dataset does not preserve explicit NA(d)/NA(n) codes. "
                    f"Found {int(other_missing_mask.sum())} missing/non-1..5 values."
                )
            # If some explicit codes exist, remaining unspecified missing are still unsplittable reliably.
            raise ValueError(
                f"Cannot fully split missing for {varname}: found {int(other_missing_mask.sum())} missing/non-1..5 values "
                f"that are neither NA(d) nor NA(n). Re-export data preserving distinct missing codes."
            )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build Table 3 structure
    # -----------------------
    genre_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

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
    # Write human-readable text file (3 blocks of 6 genres)
    # -----------------------
    formatted = table.copy()

    # cast counts to ints for display; mean to 2 decimals
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]
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