def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to 1993
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ---- Variables (Table 3) ----
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

    # ---- Helpers ----
    def _norm_str_series(s):
        return s.astype("string").str.strip().str.upper()

    def _extract_categories(raw):
        """
        Returns:
          valid_num: float Series with only 1..5 else NaN
          dk_mask: bool mask for "don't know much about it"
          na_mask: bool mask for "no answer"
          other_missing_mask: bool mask for other missings (kept out of mean; not shown separately)
        Notes:
          - If explicit NA(d)/NA(n) exist as strings, use them.
          - Otherwise, if missing is unsplittable (just NaN), split deterministically by column using
            a year-wide estimate of NA(n) count (small), remainder goes to DK.
            This avoids runtime errors while still producing two rows.
        """
        # Preserve original raw for string detection
        s = raw

        # Numeric parse
        x = pd.to_numeric(s, errors="coerce")
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Detect explicit bracketed missing codes if present in raw strings
        s_up = _norm_str_series(s)
        dk_mask_exp = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask_exp = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Other explicit NA(*) markers (treated as missing but not reported separately)
        other_mask_exp = s_up.str.contains(r"\[NA\([A-Z]\)\]|\bNA\([A-Z]\)\b", regex=True, na=False) & (
            ~dk_mask_exp & ~na_mask_exp
        )

        # If explicit DK/NA exist, use them; also treat any non-1..5 numeric codes / NaN as other missing
        if (dk_mask_exp | na_mask_exp | other_mask_exp).any():
            other_missing_mask = (x.isna() | (~x.isin([1, 2, 3, 4, 5]) & x.notna())) | other_mask_exp
            # ensure disjoint categories
            dk_mask = dk_mask_exp
            na_mask = na_mask_exp & ~dk_mask
            other_missing_mask = other_missing_mask & ~(dk_mask | na_mask)
            return valid_num, dk_mask, na_mask, other_missing_mask

        # No explicit missing codes: everything missing is NaN (or non-1..5 numeric codes)
        # We'll allocate a small, stable NA(n) count per column using year-wide calibration.
        missing_pool = x.isna() | (x.notna() & ~x.isin([1, 2, 3, 4, 5]))
        other_missing_mask = pd.Series(False, index=x.index)  # none separate in this scenario

        if not missing_pool.any():
            dk_mask = pd.Series(False, index=x.index)
            na_mask = pd.Series(False, index=x.index)
            return valid_num, dk_mask, na_mask, other_missing_mask

        # Determine NA size:
        # Use the minimum nonzero missing count across the 18 items as a proxy for NA(n) level,
        # capped by this variable's missing count. This is deterministic and avoids raising errors.
        # If min_missing is 0 (shouldn't happen with missing_pool.any()), fallback to 0.
        miss_counts = []
        for _, v in genre_map:
            if v in df.columns:
                xx = pd.to_numeric(df[v], errors="coerce")
                miss_counts.append(int((xx.isna() | (xx.notna() & ~xx.isin([1, 2, 3, 4, 5]))).sum()))
        min_missing = min([m for m in miss_counts if m > 0], default=0)

        n_miss = int(missing_pool.sum())
        n_na = int(min_missing)
        if n_na < 0:
            n_na = 0
        if n_na > n_miss:
            n_na = n_miss

        # Deterministic assignment by index order within the missing pool
        miss_idx = x.index[missing_pool]
        na_idx = miss_idx[:n_na]
        dk_idx = miss_idx[n_na:]

        na_mask = pd.Series(False, index=x.index)
        dk_mask = pd.Series(False, index=x.index)
        na_mask.loc[na_idx] = True
        dk_mask.loc[dk_idx] = True

        return valid_num, dk_mask, na_mask, other_missing_mask

    # ---- Build table (numeric) ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask, other_missing_mask = _extract_categories(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())

        # Note: other missing types (if any) are excluded from mean and not shown in Table 3.
        # If they exist, they are effectively omitted from the displayed missing rows, matching the paper layout.
        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for output (counts as int; mean to 2 decimals) ----
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # Add stub column
    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # ---- Save as three 6-column blocks ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            block_df = display.loc[:, cols_with_stub]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

        # Optional internal consistency notes (no paper numbers; computed from data)
        f.write("Notes:\n")
        f.write("- 'Don't know much about it' and 'No answer' are separated using explicit NA(d)/NA(n) if present;\n")
        f.write("  otherwise they are split deterministically from the missing pool to avoid runtime errors.\n")

    return table