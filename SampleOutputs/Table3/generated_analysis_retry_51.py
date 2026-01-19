def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + filter to 1993
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)

    # normalize column names
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
    # Missing code parsing
    # -----------------------
    # We must compute DK vs NA from the raw extract. This requires that the extract preserves
    # distinct missing codes (e.g., '[NA(d)]' and '[NA(n)]' or similar).
    # If the extract collapses them to blank/NA, separation is impossible from raw data.
    dk_patterns = [
        r"\[NA\(D\)\]",
        r"\bNA\(D\)\b",
        r"\bDON'?T\s*KNOW\b",
        r"\bDONT\s*KNOW\b",
        r"\bDK\b",
        r"\bDON'?T\s*KNOW\s*MUCH\b",
        r"\bNOT\s*SURE\b",
    ]
    na_patterns = [
        r"\[NA\(N\)\]",
        r"\bNA\(N\)\b",
        r"\bNO\s*ANSWER\b",
    ]

    dk_re = "(" + "|".join(dk_patterns) + ")"
    na_re = "(" + "|".join(na_patterns) + ")"

    def _series_as_string(s):
        # keep <NA> for missing; convert others to stripped upper strings
        return s.astype("string").str.strip().str.upper()

    def _tabulate_one(raw):
        # numeric values as floats/ints where possible
        x = pd.to_numeric(raw, errors="coerce")

        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask)

        s_up = _series_as_string(raw)

        dk_mask = s_up.str.contains(dk_re, regex=True, na=False)
        na_mask = s_up.str.contains(na_re, regex=True, na=False)

        # Any explicit DK/NA codes must not be counted as valid even if numeric coercion did something odd
        valid = valid.mask(dk_mask | na_mask)

        # Remaining "other missing" are those that are not valid and not explicitly DK/NA
        other_missing_mask = (~valid_mask) & ~(dk_mask | na_mask)

        # If there exist "other missing" but no explicit DK/NA in this variable, we cannot split DK vs NA.
        # (The paper table requires the split.)
        if other_missing_mask.any() and (dk_mask.sum() + na_mask.sum() == 0):
            raise ValueError(
                "Dataset does not preserve explicit NA(d)/NA(n) (or equivalent) codes, "
                "so '(M) Don't know much about it' vs '(M) No answer' cannot be separated."
            )

        # If there are some explicit DK/NA, but also other-missing, assign other-missing to NA (most conservative)
        # so that we never inflate "Don't know much about it".
        na_mask = na_mask | other_missing_mask

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
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable output formatting
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

    # Split into 3 blocks of 6 genres (matching the paper layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing categories.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table