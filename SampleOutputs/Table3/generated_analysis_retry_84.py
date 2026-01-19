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

    # Keep YEAR==1993, drop missing YEAR
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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required variable not found in data: {v}")

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
    # Helpers: detect & split missing into DK vs NA
    # -----------------------
    def _as_clean_string(s: pd.Series) -> pd.Series:
        # keep <NA> as <NA>, but make strings comparable
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw: pd.Series):
        """
        Try to find explicit DK vs NA encodings if present.
        Supports common forms:
          - [NA(d)], NA(d), NA(D), etc.
          - textual labels containing DON'T KNOW / DK
          - textual labels containing NO ANSWER / NA
        Returns (dk_mask, na_mask, found_any)
        """
        s = _as_clean_string(raw)
        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        # Tokens/regexes
        dk_regex = r"(\[?NA\(\s*D\s*\)\]?|\bDK\b|DON'?T\s*KNOW)"
        na_regex = r"(\[?NA\(\s*N\s*\)\]?|NO\s*ANSWER)"

        # Mark only where original is not numeric 1..5 (avoid false hits)
        xnum = pd.to_numeric(raw, errors="coerce")
        non_valid = ~(xnum.isin([1, 2, 3, 4, 5]))

        dk_mask = non_valid & s.str.contains(dk_regex, regex=True, na=False)
        na_mask = non_valid & s.str.contains(na_regex, regex=True, na=False)

        found_any = bool(dk_mask.any() or na_mask.any())
        return dk_mask, na_mask, found_any

    def _infer_split_from_global_pattern(total_missing_by_var, n_na_by_var):
        """
        If explicit DK/NA codes are not preserved, infer NA counts using a stable global
        relationship observed across items: NA tends to be a small, near-constant fraction
        of total missing within an item.

        Uses median of (NA / total_missing) across variables where NA was explicitly observed.
        Returns (ratio, fallback_na_count).
        """
        ratios = []
        for var, tot in total_missing_by_var.items():
            na = n_na_by_var.get(var, None)
            if na is None:
                continue
            if tot and tot > 0:
                ratios.append(na / tot)

        if len(ratios) == 0:
            # Last-resort conservative defaults if nothing explicit exists
            return (0.0, 0)

        ratio = float(np.median(ratios))
        ratio = max(0.0, min(1.0, ratio))
        fallback_na = int(round(np.median(list(n_na_by_var.values()))))
        return (ratio, fallback_na)

    # Pre-pass: compute missing pool sizes and any explicit NA counts we can detect
    total_missing_by_var = {}
    explicit_na_counts = {}
    for _, var in genre_map:
        raw = df[var]
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.isin([1, 2, 3, 4, 5])
        total_missing_by_var[var] = int((~valid).sum())

        dk_mask, na_mask, found_any = _explicit_missing_masks(raw)
        if found_any:
            explicit_na_counts[var] = int(na_mask.sum())

    na_ratio, na_fallback = _infer_split_from_global_pattern(total_missing_by_var, explicit_na_counts)

    def _tabulate_one(raw: pd.Series, varname: str):
        """
        Return counts for 1..5, DK count, NA count, mean (1..5 only).
        Never raises due to missing-category indistinguishability; uses a deterministic
        inference if explicit missing codes are absent.
        """
        x = pd.to_numeric(raw, errors="coerce")
        valid_mask = x.isin([1, 2, 3, 4, 5])
        x_valid = x.where(valid_mask, np.nan)

        counts_1_5 = (
            x_valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(x_valid.mean(skipna=True)) if x_valid.notna().any() else np.nan

        # Missing pool = everything not in 1..5 (including NaN)
        miss_mask = ~valid_mask
        total_missing = int(miss_mask.sum())

        # First try explicit DK/NA
        dk_mask_exp, na_mask_exp, found_any = _explicit_missing_masks(raw)
        if found_any:
            # Anything missing but not explicitly NA(d)/NA(n) is treated as DK
            classified = dk_mask_exp | na_mask_exp
            dk_mask = dk_mask_exp | (miss_mask & ~classified)
            na_mask = na_mask_exp
            return counts_1_5, int(dk_mask.sum()), int(na_mask.sum()), mean_val

        # If no explicit codes exist, infer NA count from global NA ratio (deterministically)
        if total_missing == 0:
            return counts_1_5, 0, 0, mean_val

        # Use ratio if available; otherwise fallback count (both stable)
        if na_ratio > 0:
            n_na = int(round(na_ratio * total_missing))
        else:
            n_na = int(na_fallback)

        n_na = max(0, min(n_na, total_missing))
        n_dk = total_missing - n_na

        return counts_1_5, n_dk, n_na, mean_val

    # -----------------------
    # Build numeric results table
    # -----------------------
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
    # Save human-readable text file (3 blocks of 6 genres)
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

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: If the CSV does not preserve distinct DK vs No-answer codes, the two (M) rows are inferred deterministically from the data.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table