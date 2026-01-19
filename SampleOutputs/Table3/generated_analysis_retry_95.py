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

    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        "(M) Don\u2019t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Helpers: detect DK/NA
    # -----------------------
    def _as_clean_string(s: pd.Series) -> pd.Series:
        # keep <NA> for missing; normalize whitespace/case
        return s.astype("string").str.strip().str.upper()

    def _detect_missing_masks(raw: pd.Series):
        """
        Returns:
          dk_mask: "don't know much about it" (prefer NA(d) if present)
          na_mask: "no answer" (prefer NA(n) if present)
          unclassified_missing_mask: missing/non-1..5 values not classified as dk/na
          valid_numeric: numeric series with only 1..5 retained, else NaN
        """
        s_str = _as_clean_string(raw)

        # Numeric parse for substantive 1..5
        x = pd.to_numeric(raw, errors="coerce")
        valid_numeric = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Start with explicit NA(d)/NA(n) patterns (common in labelled exports)
        dk_mask = s_str.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b|\bDON['’]T KNOW\b|\bDONT KNOW\b|\bDK\b", regex=True, na=False)
        na_mask = s_str.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b|\bNO ANSWER\b|\bNA\b", regex=True, na=False)

        # Some exports encode missing as distinct numeric codes; catch common ones if present.
        # Note: we only classify if a code is clearly DK/NA-like.
        # If these do not exist in the data, they have no effect.
        dk_numeric_codes = {8, 98}
        na_numeric_codes = {9, 99}

        x_num = pd.to_numeric(raw, errors="coerce")
        dk_mask = dk_mask | x_num.isin(list(dk_numeric_codes))
        na_mask = na_mask | x_num.isin(list(na_numeric_codes))

        # Missing/non-1..5 pool
        missing_or_invalid = valid_numeric.isna()

        # Anything explicitly classified is removed from unclassified pool
        unclassified_missing = missing_or_invalid & ~(dk_mask | na_mask)

        # If the export does not preserve DK vs NA explicitly, we cannot split them.
        # In that case, we report all missing/non-1..5 as DK and set NA to 0
        # (but still keep correct totals and keep mean correct).
        if missing_or_invalid.any() and (dk_mask.sum() + na_mask.sum() == 0):
            dk_mask = missing_or_invalid.copy()
            na_mask = pd.Series(False, index=raw.index)
            unclassified_missing = pd.Series(False, index=raw.index)

        return dk_mask, na_mask, unclassified_missing, valid_numeric

    # -----------------------
    # Build numeric results table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        raw = df[var]
        dk_mask, na_mask, unclassified_missing, valid = _detect_missing_masks(raw)

        # Frequencies 1..5
        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don\u2019t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # Safety: if there are unclassified missing values *and* we did find some explicit DK/NA,
        # we should not silently drop them. Fold them into "No answer" (conservative).
        # (If there were no explicit codes, we already put all missing into DK above.)
        if unclassified_missing.any() and (dk_mask.sum() + na_mask.sum() > 0):
            table.loc["(M) No answer", genre_label] = int(table.loc["(M) No answer", genre_label] + int(unclassified_missing.sum()))

    # -----------------------
    # Human-readable text output (3 blocks of 6 genres)
    # -----------------------
    # Create a display copy with integer counts and 2dp means
    display = table.copy()

    for idx in display.index:
        if idx == "Mean":
            display.loc[idx] = display.loc[idx].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            display.loc[idx] = display.loc[idx].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display.insert(0, "Attitude", display.index)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding (M) categories.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table