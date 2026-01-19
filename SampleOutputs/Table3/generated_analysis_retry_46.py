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

    missing_label_map = {
        "D": "(M) Don't know much about it",
        "N": "(M) No answer",
    }

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
    # Helpers: detect NA(d)/NA(n) in raw exports
    # -----------------------
    def _as_clean_string(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip()

    def _extract_missing_letter(series: pd.Series) -> pd.Series:
        """
        Return a Series with missing-code letter ('d','n', etc.) when values look like:
          '[NA(d)]', 'NA(d)', '[na(n)]', etc. Otherwise <NA>.
        """
        s = _as_clean_string(series).str.upper()
        extracted = s.str.extract(r"NA\(([A-Z])\)", expand=False)
        return extracted

    def _tabulate_one(series: pd.Series, varname: str):
        """
        Compute counts for 1..5 plus separate DK vs No answer, and mean on 1..5.
        Requires explicit NA(d) / NA(n) markers to separate the two missing buckets.
        """
        # numeric valid
        x_num = pd.to_numeric(series, errors="coerce")
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit missing letters
        miss_letter = _extract_missing_letter(series)  # 'D', 'N', etc.
        dk_n = int((miss_letter == "D").sum())
        na_n = int((miss_letter == "N").sum())

        # All non-1..5 that are not explicitly tagged are "unknown missing".
        # The paper separates only DK vs NA; without explicit tags we cannot reproduce.
        non_1_5_nonnull = series.notna() & ~x_num.isin([1, 2, 3, 4, 5])
        tagged_any = miss_letter.notna()
        unknown_missing = int((valid.isna() & (series.notna() | non_1_5_nonnull) & ~tagged_any).sum())

        if unknown_missing > 0:
            examples = (
                series.loc[valid.isna() & series.notna() & ~tagged_any]
                .astype("string")
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                f"found {unknown_missing} missing/non-1..5 values without explicit NA(d)/NA(n) tags. "
                f"Re-export the data preserving distinct missing codes. Examples: {examples}"
            )

        counts_1_5 = (
            pd.Series(valid)
            .value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        mean_val = float(np.nanmean(valid)) if np.isfinite(np.nanmean(valid)) else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # Validate required columns exist
    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required variable not found in data: {var}")

    # Quick check whether any explicit NA(d)/NA(n) exist at all
    any_tagged = False
    for _, var in genre_map:
        letters = _extract_missing_letter(df[var])
        if int((letters == "D").sum() + (letters == "N").sum()) > 0:
            any_tagged = True
            break
    if not any_tagged:
        raise ValueError(
            "Dataset does not preserve explicit NA(d)/NA(n) codes (or equivalent string labels), "
            "so '(M) Don't know much about it' vs '(M) No answer' cannot be separated. "
            "Re-export the data preserving these missing categories (e.g., '[NA(d)]' and '[NA(n)]')."
        )

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
    # Save human-readable text (3 blocks of 6)
    # -----------------------
    # Format: counts as integers, mean rounded to 2 decimals
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
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