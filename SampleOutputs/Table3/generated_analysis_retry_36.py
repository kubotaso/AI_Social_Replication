def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + normalize columns
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Keep YEAR == 1993; drop missing YEAR
    year_num = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.loc[year_num.eq(1993)].copy()

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

    missing_label_dk = "(M) Don't know much about it"
    missing_label_na = "(M) No answer"

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        missing_label_dk,
        missing_label_na,
        "Mean",
    ]

    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

    # -----------------------
    # Helpers for missing parsing
    # -----------------------
    def _as_upper_str(s: pd.Series) -> pd.Series:
        # Use pandas StringDtype to keep NA, then normalize
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw: pd.Series):
        """
        Detect explicit missing codes if they survived in the CSV as strings.
        We accept any of these encodings:
          - "[NA(d)]" or "NA(d)" or "NA(D)"
          - "[NA(n)]" or "NA(n)" or "NA(N)"
        """
        s_up = _as_upper_str(raw)
        dk = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)
        return dk, na

    def _compute_counts_and_mean(raw: pd.Series, varname: str):
        """
        Compute counts for 1..5, counts for DK/NA, and mean on 1..5.

        IMPORTANT:
        If the dataset extract does not preserve distinct DK vs NA codes,
        we cannot truthfully split missing into the two required rows.
        In that case, raise with a clear error message (instead of inventing numbers).
        """
        # Parse explicit DK/NA string codes if present
        dk_mask, na_mask = _explicit_missing_masks(raw)
        has_explicit = bool(dk_mask.any() or na_mask.any())

        # Numeric parse for substantive codes
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # "Non-1..5" can include: explicit DK/NA strings, other NA strings, blanks, etc.
        non_1_5 = ~x.isin([1, 2, 3, 4, 5])

        # If there are missing/non-1..5 values but no explicit DK/NA labels, we cannot split.
        # (Counts/means for 1..5 can still be computed, but the table requires both rows.)
        total_non_1_5 = int(non_1_5.sum())

        if total_non_1_5 > 0 and not has_explicit:
            raise ValueError(
                f"Cannot compute separate '{missing_label_dk}' vs '{missing_label_na}' counts for {varname}: "
                "dataset does not contain explicit NA(d)/NA(n) codes (or equivalent string labels)."
            )

        # With explicit codes: count DK and NA, and treat any other non-1..5 as additional DK
        # only if it is *not* explicitly NA(n). This keeps totals consistent without losing cases.
        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())

        # Assign any remaining non-1..5 (e.g., other NA flavors) into DK bucket by default
        # (Table 3 only prints DK and No answer; it does not show refused/other separately.)
        other_missing = non_1_5 & ~(dk_mask | na_mask)
        dk_n += int(other_missing.sum())

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _compute_counts_and_mean(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc[missing_label_dk, genre_label] = dk_n
        table.loc[missing_label_na, genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable output (3 blocks of 6 genres)
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