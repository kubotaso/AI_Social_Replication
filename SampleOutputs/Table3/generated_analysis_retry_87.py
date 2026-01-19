def run_analysis(data_source):
    import os
    import re
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
            raise ValueError(f"Required column missing: {v}")

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
    # Missing-code parsing helpers
    # -----------------------
    # We need to distinguish DK vs NA if the raw export preserved them.
    # Accepted encodings:
    #  - string tokens like "[NA(d)]", "NA(d)", "DK", "DON'T KNOW", etc.
    #  - numeric codes like 8/9, 98/99, 0/-1/-2 (best-effort; depends on export)
    #
    # If the export collapsed them to plain NaN with no distinction, we cannot
    # reconstruct DK vs NA from raw data alone; in that case we will error
    # to avoid silently producing incorrect rows.

    DK_TOKEN_RE = re.compile(r"(\[?\s*NA\s*\(\s*D\s*\)\s*\]?|\bDK\b|DON[’']?T\s+KNOW)", re.IGNORECASE)
    NA_TOKEN_RE = re.compile(r"(\[?\s*NA\s*\(\s*N\s*\)\s*\]?|\bNO\s+ANSWER\b)", re.IGNORECASE)

    # Common numeric missing conventions (best-effort). GSS extracts vary.
    DK_NUMERIC = {8, 98}      # sometimes used for don't know
    NA_NUMERIC = {9, 99}      # sometimes used for no answer
    # other missing-like codes that might appear; treated as "other missing" and not allocable
    OTHER_MISS_NUMERIC = {0, 97, -1, -2, -3, -7, -8, -9, 998, 999}

    def _string_masks(raw):
        s = raw.astype("string")
        s = s.str.strip()
        s_up = s.str.upper()

        dk = s_up.fillna("").str.contains(DK_TOKEN_RE)
        na = s_up.fillna("").str.contains(NA_TOKEN_RE)
        return dk, na

    def _numeric_series(raw):
        return pd.to_numeric(raw, errors="coerce")

    def _tabulate_one(raw, varname):
        # Build masks for explicit DK/NA from strings
        dk_s, na_s = _string_masks(raw)

        # Parse numeric
        x = _numeric_series(raw)

        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        # numeric DK/NA masks
        dk_n = x.isin(list(DK_NUMERIC))
        na_n = x.isin(list(NA_NUMERIC))

        dk_mask = dk_s | dk_n
        na_mask = na_s | na_n

        # Any row that is not valid and not classified DK/NA is "unclassified missing"
        not_valid = ~valid_mask
        unclassified_missing = not_valid & ~(dk_mask | na_mask)

        # If we have ANY unclassified missing, we cannot split DK vs NA reliably
        # (this is exactly the failure mode seen in prior runs: NaN collapsed).
        if int(unclassified_missing.sum()) > 0:
            # If absolutely no explicit DK/NA were detected, it's definitely a collapsed export.
            # If some explicit were detected but there are still unclassified, export is mixed/incomplete.
            sample_vals = raw.loc[unclassified_missing].head(10).tolist()
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                f"found {int(unclassified_missing.sum())} missing/non-1..5 values that are not classifiable as DK/NA "
                f"from this CSV export. Example unclassified raw values: {sample_vals}. "
                f"Re-export data preserving DK vs No-answer codes (e.g., '[NA(d)]'/'[NA(n)]' or distinct numeric codes)."
            )

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, int(dk_mask.sum()), int(na_mask.sum()), mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_cnt, na_cnt, mean_val = _tabulate_one(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_cnt
        table.loc["(M) No answer", genre_label] = na_cnt
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Format for file output (3 panels of 6)
    # -----------------------
    formatted = table.copy()

    # keep counts as integers; mean to 2 decimals
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
            f.write(f"Panel {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table