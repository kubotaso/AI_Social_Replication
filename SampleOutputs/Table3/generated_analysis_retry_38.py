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

    # Filter to 1993
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

    missing_other_codes = {
        "I", "M", "R", "S", "U", "X", "Y", "Z", "P", "J"
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

    def _as_clean_string(s: pd.Series) -> pd.Series:
        # Keep pandas NA; normalize strings for parsing
        return (
            s.astype("string")
            .str.strip()
            .str.upper()
        )

    def _extract_missing_letter(s_up: pd.Series) -> pd.Series:
        """
        Extract missing letter from strings like:
        '[NA(D)]', 'NA(D)', '[NA(d)]', etc.
        Returns pandas StringDtype series with values like 'D','N', or <NA>.
        """
        extracted = s_up.str.extract(r"\[?\s*NA\(\s*([A-Z])\s*\)\s*\]?", expand=False)
        return extracted.astype("string")

    def _compute_counts_for_var(raw: pd.Series, varname: str):
        """
        Returns:
          counts_1_5 (Series indexed 1..5),
          dk_count (int),
          na_count (int),
          mean_val (float)
        """
        # Numeric valid values 1..5
        x_num = pd.to_numeric(raw, errors="coerce")
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # Missing splits require explicit NA(d) / NA(n) in the raw data
        s_up = _as_clean_string(raw)
        miss_letter = _extract_missing_letter(s_up)

        has_any_letter = miss_letter.notna().any()

        if has_any_letter:
            # Treat NA(D) as "don't know much", NA(N) as "no answer"
            dk_mask = miss_letter.eq("D")
            na_mask = miss_letter.eq("N")

            # Any other NA(letter) codes are treated as missing-but-not-in-table.
            # Table 3 only prints D and N separately; other missing are NOT expected here.
            other_mask = miss_letter.isin(list(missing_other_codes)) & ~(dk_mask | na_mask)
            other_count = int(other_mask.sum())

            # Also handle blank/pandas NA values (true NaN) if present.
            # If they exist alongside explicit NA(d)/NA(n), they should be classified as "No answer"
            # only if the data uses them for NA(n). Otherwise, we cannot split reliably.
            # We'll only allow them if there are none (strict), to prevent silent misclassification.
            plain_nan_mask = raw.isna() | s_up.isna()
            plain_nan_count = int(plain_nan_mask.sum())

            if other_count > 0 or plain_nan_count > 0:
                # Fail loudly: cannot reproduce Table 3's two missing rows exactly if
                # missing values exist beyond NA(d)/NA(n) and aren't mapped.
                raise ValueError(
                    f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                    f"found missing values not coded as NA(d)/NA(n) (other NA(*) count={other_count}, plain NA/blank count={plain_nan_count}). "
                    f"Re-export data preserving distinct missing codes or remove/resolve other missing categories."
                )

            return counts_1_5, int(dk_mask.sum()), int(na_mask.sum()), mean_val

        # No explicit missing letters: cannot split DK vs NA (must not guess)
        # But we can detect that there are missing/non-1..5 values.
        nonvalid_mask = ~(x_num.isin([1, 2, 3, 4, 5])) | x_num.isna()
        nonvalid_count = int(nonvalid_mask.sum())
        if nonvalid_count > 0:
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                f"dataset does not preserve explicit NA(d)/NA(n) codes. Found {nonvalid_count} missing/non-1..5 values."
            )

        # No missing at all
        return counts_1_5, 0, 0, mean_val

    # -----------------------
    # Build numeric table (counts + mean)
    # -----------------------
    genre_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        counts_1_5, dk_n, na_n, mean_val = _compute_counts_for_var(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable text output (3 blocks of 6 genres like paper)
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"

    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing categories.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table