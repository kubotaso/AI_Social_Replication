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
    # Helpers: detect explicit GSS-style missing tags, if present
    # -----------------------
    def _as_string_upper(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _dk_na_masks(raw: pd.Series):
        """
        Returns boolean masks for explicit DK and NA(n) encodings, if present as strings.
        Accepts variants like '[NA(d)]', 'NA(d)', etc.
        """
        s = _as_string_upper(raw)
        dk = s.str.contains(r"\[?NA\(D\)\]?", regex=True, na=False)
        na = s.str.contains(r"\[?NA\(N\)\]?", regex=True, na=False)
        return dk, na

    # -----------------------
    # Tabulation for one variable
    # -----------------------
    def _tabulate_one(raw: pd.Series, varname: str):
        # parse numeric where possible
        x = pd.to_numeric(raw, errors="coerce")

        # valid attitude responses
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # attempt to use explicit DK/NA codes if present in raw strings
        dk_mask, na_mask = _dk_na_masks(raw)
        has_explicit = bool((dk_mask | na_mask).any())

        # missing pool: values not in 1..5, including NaN
        miss_pool = valid.isna()

        if has_explicit:
            # ensure explicit DK/NA are included in missing pool; any other missing treated as DK by default
            other_missing = miss_pool & ~(dk_mask | na_mask)
            dk_mask = dk_mask | other_missing
        else:
            # If the dataset extract does not preserve distinct DK vs NA, we cannot split them.
            # Fail fast with a clear message (do not fabricate numbers).
            missing_n = int(miss_pool.sum())
            if missing_n > 0:
                raise ValueError(
                    f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                    f"dataset does not preserve explicit NA(d)/NA(n) codes (or equivalent string labels). "
                    f"Found {missing_n} missing/non-1..5 values."
                )
            dk_mask = pd.Series(False, index=raw.index)
            na_mask = pd.Series(False, index=raw.index)

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
    # Build table (numeric)
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
    # Save human-readable text (3 blocks of 6 genres, with row labels)
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", display.index)
    display = display.reset_index(drop=True)

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing (M) categories.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table