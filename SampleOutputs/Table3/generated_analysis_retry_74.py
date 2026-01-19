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

    missing_row_labels = [
        "(M) Don't know much about it",
        "(M) No answer",
    ]

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        *missing_row_labels,
        "Mean",
    ]

    # -----------------------
    # Helpers for missing-code detection
    # -----------------------
    def _norm_str_series(s):
        # Robust string normalization, preserving NaN
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw):
        """
        Try to identify distinguishable DK vs No-answer categories.

        Supports:
        - GSS-style labeled NA strings: [NA(d)] and [NA(n)] (any case/spaces)
        - Common textual variants: "DON'T KNOW", "DONT KNOW", "DK", "NO ANSWER", "NA"
        - Numeric-coded missing commonly used in some extracts: 8=DK, 9=NA (heuristic)
        """
        s_up = _norm_str_series(raw)

        # Token-based (string) detection
        # Note: keep patterns tight to reduce false positives (e.g., "N/A" in free text unlikely here)
        dk_pat = re.compile(r"(\[?\s*NA\s*\(\s*D\s*\)\s*\]?)|(\bDONT\s*KNOW\b)|(\bDON'T\s*KNOW\b)|(\bDK\b)|(\bDONOT\s*KNOW\b)")
        na_pat = re.compile(r"(\[?\s*NA\s*\(\s*N\s*\)\s*\]?)|(\bNO\s*ANSWER\b)|(\bNOANSWER\b)|(\bNA\b)")

        dk_str = s_up.str.contains(dk_pat, regex=True, na=False)
        na_str = s_up.str.contains(na_pat, regex=True, na=False)

        # Numeric-coded missing (heuristic)
        x = pd.to_numeric(raw, errors="coerce")
        dk_num = x.eq(8)
        na_num = x.eq(9)

        dk = dk_str | dk_num
        na = na_str | na_num

        # Avoid overlaps (if any ambiguous "NA" also matches DK pattern, NA wins only if explicit NA(n))
        both = dk & na
        if both.any():
            # prioritize explicit NA(n) over generic NA; otherwise keep DK
            na_explicit = s_up.str.contains(re.compile(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?"), regex=True, na=False) | x.eq(9)
            dk = dk & ~na_explicit
            na = na | (both & na_explicit)

        return dk, na

    def _tabulate_one(raw, varname):
        """
        Returns:
          counts_1_5: Series indexed 1..5
          dk_n: int
          na_n: int
          mean_val: float
        """
        x = pd.to_numeric(raw, errors="coerce")

        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        counts_1_5 = valid.value_counts(dropna=True).reindex([1, 2, 3, 4, 5], fill_value=0).astype(int)

        # Identify DK/NA explicitly if possible
        dk_mask_exp, na_mask_exp = _explicit_missing_masks(raw)

        # "Missing pool" = anything not a valid 1..5 (including NaN and out-of-range codes)
        missing_pool = ~x.isin([1, 2, 3, 4, 5])
        missing_pool = missing_pool.fillna(True)  # numeric NaN => missing

        classified = (dk_mask_exp | na_mask_exp)
        unclassified_missing = missing_pool & ~classified

        # If we have any explicit classification, treat remaining unclassified missing as "No answer"
        # (typically system missing/refused/skipped end up here). This avoids collapsing into DK.
        if classified.any():
            dk_n = int(dk_mask_exp.sum())
            na_n = int((na_mask_exp | unclassified_missing).sum())
            mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
            return counts_1_5, dk_n, na_n, mean_val

        # Otherwise, the CSV has lost distinguishable missing categories (everything is just NaN).
        # In that case we cannot recover DK vs No answer from raw data alone.
        # We fail loudly rather than inventing a split.
        total_missing = int(missing_pool.sum())
        if total_missing > 0:
            raise ValueError(
                f"Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts for {varname}: "
                f"this CSV export does not preserve distinguishable missing categories (e.g., '[NA(d)]'/'[NA(n)]' "
                f"or numeric codes like 8/9). Found {total_missing} missing/non-1..5 values."
            )

        dk_n = 0
        na_n = 0
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build Table 3
    # -----------------------
    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

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
    # Human-readable output
    # -----------------------
    formatted = table.copy()

    # counts as ints, mean as 2 decimals
    for idx in formatted.index:
        if idx == "Mean":
            formatted.loc[idx] = formatted.loc[idx].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[idx] = formatted.loc[idx].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres to match typical layout
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            f.write(display.loc[:, ["Attitude"] + cols].to_string(index=False))
            f.write("\n\n")

    return table