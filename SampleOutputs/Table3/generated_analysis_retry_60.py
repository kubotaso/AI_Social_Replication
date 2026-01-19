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

    # Filter: YEAR == 1993 (exclude missing automatically by comparison)
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
    # We must compute DK vs NA from raw data; never hard-code paper numbers.
    #
    # Strategy:
    # 1) Count substantive responses strictly as codes 1..5 (numeric).
    # 2) If the file preserves explicit missing codes as strings such as "[NA(d)]" and "[NA(n)]",
    #    count them separately.
    # 3) If explicit codes are NOT preserved (e.g., blanks/NaN), we cannot split DK vs NA from this CSV;
    #    to keep the pipeline running and still produce the 8-row Table 3 structure, we:
    #       - set DK = total_missing (all non-1..5 / NA)
    #       - set NA = 0
    #    This avoids runtime errors and keeps counts/means correct for the substantive categories and mean.
    #    (If you need the exact split, the CSV must preserve DK vs NA distinctly.)
    def _as_clean_str(s):
        return s.astype("string").str.strip()

    def _explicit_missing_masks(raw_series):
        s = _as_clean_str(raw_series).str.upper()
        # Common encodings seen in GSS extracts
        dk = s.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)
        return dk, na

    def _tabulate_one(raw_series):
        x = pd.to_numeric(raw_series, errors="coerce")

        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_mask, na_mask = _explicit_missing_masks(raw_series)
        explicit_any = int(dk_mask.sum()) + int(na_mask.sum()) > 0

        # "Missing pool" includes anything not in 1..5 (including NaN after numeric coercion)
        miss_pool = valid.isna()

        if explicit_any:
            # If explicit DK/NA exist, treat any other missings as DK (conservative)
            other_missing = miss_pool & ~(dk_mask | na_mask)
            dk_n = int((dk_mask | other_missing).sum())
            na_n = int(na_mask.sum())
        else:
            # CSV does not preserve distinct missing categories: cannot split DK vs NA.
            dk_n = int(miss_pool.sum())
            na_n = 0

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

    # Round means to 2 decimals for presentation (keep numeric table as-is? We will output formatted)
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # Insert row label column
    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # -----------------------
    # Save human-readable text (3 blocks of 6 genres like the paper layout)
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: If the CSV does not preserve distinct DK vs No-answer codes, '(M) No answer' will be 0 and all missing will be counted under DK.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table