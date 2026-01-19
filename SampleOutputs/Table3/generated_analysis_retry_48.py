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
    # Helpers: preserve distinct missing codes if present
    # -----------------------
    def _as_str_series(s: pd.Series) -> pd.Series:
        # keep <NA> as <NA>
        return s.astype("string")

    def _code_masks(raw: pd.Series):
        """
        Returns:
          valid_num: numeric series with only 1..5 else NaN
          dk_mask: boolean mask for [NA(d)] / NA(d) / don't know / dk
          na_mask: boolean mask for [NA(n)] / NA(n) / no answer
          other_missing_mask: missing but not classified as dk/na (left out of DK/NA rows)
        """
        s_str = _as_str_series(raw).str.strip()

        # numeric parse for substantive codes
        x = pd.to_numeric(raw, errors="coerce")
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit string encodings like "[NA(d)]"
        s_up = s_str.str.upper()

        # detect explicit NA(d), NA(n)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # sometimes exports contain literal labels
        # (be conservative to avoid misclassifying ordinary text)
        dk_mask = dk_mask | s_up.str.fullmatch(r"(DK|DON'T KNOW|DONT KNOW|DON.?T KNOW MUCH ABOUT IT)", na=False)
        na_mask = na_mask | s_up.str.fullmatch(r"(NO ANSWER|NA|N/A)", na=False)

        # missing pool: anything not a valid 1..5
        missing_pool = valid_num.isna()

        # other missing: in missing pool but not dk/na
        other_missing_mask = missing_pool & ~(dk_mask | na_mask)

        return valid_num, dk_mask, na_mask, other_missing_mask

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    # Track whether we can compute DK/NA from data
    any_dk = False
    any_na = False
    any_other_missing = False

    for genre_label, var in genre_map:
        raw = df[var]
        valid_num, dk_mask, na_mask, other_missing_mask = _code_masks(raw)

        any_dk = any_dk or bool(dk_mask.any())
        any_na = any_na or bool(na_mask.any())
        any_other_missing = any_other_missing or bool(other_missing_mask.any())

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]

        # Only count DK/NA when they are explicitly distinguishable in the data
        table.loc["(M) Don't know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        table.loc["Mean", genre_label] = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

    # If the dataset doesn't preserve DK/NA distinctions, do NOT error;
    # instead, keep DK/NA at 0 and document the limitation in the output file.
    # If there are missing values but none classified, they are "other missing" and not allocated.
    limitation_note = None
    if not any_dk and not any_na:
        # check if there are missing/non-1..5 values at all
        total_other = 0
        for _, var in genre_map:
            _, _, _, other_missing_mask = _code_masks(df[var])
            total_other += int(other_missing_mask.sum())
        if total_other > 0:
            limitation_note = (
                "NOTE: This CSV extract does not preserve distinct missing categories for music items "
                "(e.g., '[NA(d)]' vs '[NA(n)]'). Counts for the two '(M)' rows are therefore 0, and "
                "any missing/non-1..5 values are not separately allocated. Means are computed on valid 1–5 only."
            )

    # -----------------------
    # Save human-readable text (3 panels of 6 genres)
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
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        if limitation_note:
            f.write(limitation_note + "\n")
        f.write("\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table