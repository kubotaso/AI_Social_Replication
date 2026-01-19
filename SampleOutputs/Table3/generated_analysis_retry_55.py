def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + harmonize columns
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
    # Missing-category parsing
    # -----------------------
    # We will ONLY report separate DK vs No answer when the CSV actually preserves them
    # via explicit string tokens (e.g., "[NA(d)]", "NA(d)", "[NA(n)]", "NA(n)").
    # If the export collapses these to blank/NaN, separation is not identifiable.
    dk_pat = r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?"
    na_pat = r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?"

    def _as_clean_string(s: pd.Series) -> pd.Series:
        # keep pandas NA as <NA> in string dtype
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw: pd.Series):
        s = _as_clean_string(raw)
        dk_mask = s.str.contains(dk_pat, regex=True, na=False)
        na_mask = s.str.contains(na_pat, regex=True, na=False)
        return dk_mask, na_mask

    def _tabulate_one(raw: pd.Series, varname: str):
        # Parse numeric substantive codes
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Try to find explicit DK vs NA tokens in the raw series
        dk_mask, na_mask = _explicit_missing_masks(raw)

        # If explicit tokens exist anywhere, use them and treat remaining non-1..5 as other missing
        if int(dk_mask.sum()) + int(na_mask.sum()) > 0:
            # Any remaining non-1..5 (including blank/NaN or other NA codes) cannot be
            # assigned to DK vs NA reliably; fold them into "No answer" (conservative).
            other_missing = valid.isna() & ~(dk_mask | na_mask)
            na_mask = na_mask | other_missing
            dk_n = int(dk_mask.sum())
            na_n = int(na_mask.sum())
            mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
            return counts_1_5, dk_n, na_n, mean_val

        # No explicit tokens: dataset collapsed missing categories; cannot separate DK vs NA.
        # But Table 3 requires these as distinct rows -> raise with a clear error.
        total_missing = int(valid.isna().sum())
        if total_missing > 0:
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts "
                f"for {varname}: dataset does not contain explicit NA(d)/NA(n) codes (or equivalent string labels). "
                f"Found {total_missing} missing/non-1..5 values but they are unsplittable. "
                f"Re-export data preserving distinct missing codes (e.g., '[NA(d)]' and '[NA(n)]')."
            )

        # No missing at all
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, 0, 0, mean_val

    # -----------------------
    # Build numeric table
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
    # Human-readable text output (3 blocks of 6 genres)
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else str(int(round(float(v))))
            )

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
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table