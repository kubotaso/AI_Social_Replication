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

    # -----------------------
    # Missing-code handling
    # -----------------------
    # We must compute DK vs NA from raw data (not hard-coded from paper).
    # The CSV may encode missing as:
    #   - actual NaN
    #   - strings like "[NA(d)]" and "[NA(n)]"
    #   - other "[NA(...)]" strings
    #
    # Rule:
    #   - Count DK as explicit NA(d)
    #   - Count No answer as explicit NA(n)
    #   - If those explicit codes are not present anywhere for a variable and there exist
    #     missing/non-1..5 values, we cannot split. In that case, raise a clear error
    #     (instead of silently fabricating a split).
    #
    # Means are computed using only valid codes 1..5.

    def _to_str_upper(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _dk_na_masks(raw: pd.Series):
        s = _to_str_upper(raw)
        dk = s.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)
        other_na_tag = s.str.contains(r"\[NA\([A-Z]\)\]|\bNA\([A-Z]\)\b", regex=True, na=False)
        return dk, na, other_na_tag

    def _compute_for_var(raw: pd.Series, varname: str):
        # numeric values for 1..5
        x_num = pd.to_numeric(raw, errors="coerce")
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit NA(d)/NA(n) if present as strings
        dk_mask, na_mask, other_na_tag = _dk_na_masks(raw)

        # missing/non-1..5 pool includes:
        # - numeric NaN (from coercion or existing)
        # - numeric values not in 1..5
        # - explicit NA(...) tags (which become NaN when coerced)
        miss_pool = valid.isna()

        # Determine whether split is possible
        # We consider split possible iff we see any explicit NA(d) or NA(n)
        # OR there are no missing at all.
        if miss_pool.sum() > 0 and (dk_mask.sum() + na_mask.sum()) == 0:
            # Sometimes the dataset might already have separate numeric codes for DK/NA.
            # If so, they would appear as non-1..5 numeric values. We can detect common patterns:
            # - DK often coded as 8 or 9, NA as 0, 98, 99, etc. But we must not guess.
            # Therefore, we require explicit NA(d)/NA(n) tags to split.
            # Provide diagnostics to help user fix extract.
            non_1_5_vals = pd.to_numeric(raw, errors="coerce")
            non_1_5_vals = non_1_5_vals[~non_1_5_vals.isna() & ~non_1_5_vals.isin([1, 2, 3, 4, 5])]
            examples = non_1_5_vals.value_counts().head(5).to_dict()
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts "
                f"for {varname}: dataset does not preserve explicit NA(d)/NA(n) codes. "
                f"Found {int(miss_pool.sum())} missing/non-1..5 values. "
                f"Non-1..5 numeric examples (top 5): {examples}. "
                f"Re-export data with distinct missing codes (e.g., '[NA(d)]' and '[NA(n)]')."
            )

        # If explicit codes exist, count them; any remaining missing tagged as other NA(...) or NaN
        # are treated as No answer (generic missing in Table 3 beyond DK)
        # This matches table spec: it only shows two M rows; everything else should fall into "No answer".
        other_missing = miss_pool & ~(dk_mask | na_mask)
        na_mask = na_mask | other_missing

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
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _compute_for_var(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Save human-readable text
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