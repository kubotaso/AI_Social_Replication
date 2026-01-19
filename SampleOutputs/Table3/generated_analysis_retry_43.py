def run_analysis(data_source):
    import os
    import re
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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {v}")

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        "(M) Don’t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Helpers: detect NA(d)/NA(n) if preserved
    # -----------------------
    _re_dk = re.compile(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", re.IGNORECASE)
    _re_na = re.compile(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", re.IGNORECASE)

    def _classify_missing(series):
        """
        Returns:
          valid_num: float series with only 1..5 kept, else NaN
          dk_mask: boolean mask for NA(d) ("don't know much about it")
          na_mask: boolean mask for NA(n) ("no answer")
          miss_mask: boolean mask for all missing/non-1..5 (includes dk/na/other)
        """
        # Preserve original for string-coded missing detection
        s_obj = series.astype("string")

        # Detect explicit missing labels if present in raw CSV
        dk_mask = s_obj.str.contains(_re_dk, na=False)
        na_mask = s_obj.str.contains(_re_na, na=False)

        # Numeric parsing for substantive codes
        x = pd.to_numeric(series, errors="coerce")
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Anything non-1..5 is "missing/non-substantive" for purposes of Table 3 M-rows
        miss_mask = valid_num.isna()

        return valid_num, dk_mask, na_mask, miss_mask

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    # Pre-check: confirm that the dataset preserves DK vs NA as distinct encodings
    # If not preserved, we cannot honestly split M into two rows without fabricating numbers.
    total_dk = 0
    total_na = 0
    total_missing = 0
    for _, var in genre_map:
        valid_num, dk_mask, na_mask, miss_mask = _classify_missing(df[var])
        total_dk += int(dk_mask.sum())
        total_na += int(na_mask.sum())
        total_missing += int(miss_mask.sum())

    if total_missing > 0 and (total_dk + total_na) == 0:
        raise ValueError(
            "Cannot compute separate '(M) Don’t know much about it' vs '(M) No answer' counts: "
            "dataset does not preserve explicit NA(d)/NA(n) codes (or equivalent string labels). "
            "Re-export the data so these two missing categories are distinguishable."
        )

    # If explicit DK/NA exist but there are additional non-1..5 missings not classified,
    # we fold them into 'No answer' (conservative; does not affect means).
    for genre_label, var in genre_map:
        valid_num, dk_mask, na_mask, miss_mask = _classify_missing(df[var])

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        other_missing = miss_mask & ~(dk_mask | na_mask)
        na_mask_adj = na_mask | other_missing  # fold unclassified missing into "No answer"

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask_adj.sum())
        table.loc["Mean", genre_label] = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

    # -----------------------
    # Save human-readable output (3 blocks of 6)
    # -----------------------
    def _format_cell(row_name, v):
        if pd.isna(v):
            return ""
        if row_name == "Mean":
            return f"{float(v):.2f}"
        return str(int(round(float(v))))

    formatted = pd.DataFrame(index=table.index, columns=table.columns, dtype="object")
    for r in table.index:
        formatted.loc[r] = [ _format_cell(r, table.loc[r, c]) for c in table.columns ]

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i+6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table