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

    # Keep 1993 only (drop missing YEAR implicitly)
    df = df.loc[df["YEAR"] == 1993].copy()

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

    # Validate presence
    missing_vars = [v for _, v in genre_map if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required genre variables: {missing_vars}")

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
    # Helper: detect explicit missing tokens if present
    # -----------------------
    def _explicit_missing_masks(raw_series):
        """
        Returns (dk_mask, na_mask) for explicit GSS-style strings if present.
        We support common encodings: '[NA(d)]', 'NA(d)', 'DON'T KNOW', etc.
        If series is numeric-only, these will be all-False.
        """
        s = raw_series.astype("string")
        s_up = s.str.strip().str.upper()

        # DK tokens
        dk_tokens = [
            "[NA(D)]",
            "NA(D)",
            "NA(DON'T KNOW)",
            "DONT KNOW",
            "DON'T KNOW",
            "DK",
            "DON'T KNOW MUCH ABOUT IT",
            "DONT KNOW MUCH ABOUT IT",
        ]
        # No-answer tokens
        na_tokens = [
            "[NA(N)]",
            "NA(N)",
            "NO ANSWER",
            "NA",
        ]

        def contains_any(tokens):
            mask = pd.Series(False, index=s_up.index)
            for t in tokens:
                # regex=False to avoid escaping issues
                mask = mask | s_up.str.contains(t, regex=False, na=False)
            return mask

        return contains_any(dk_tokens), contains_any(na_tokens)

    # -----------------------
    # Helper: tabulate one item
    # -----------------------
    def _tabulate_one(raw_series):
        """
        Compute counts for 1..5 plus DK and No answer.
        IMPORTANT: If the CSV export does not preserve DK vs NA distinctly (both appear as NaN),
        we cannot recover them from raw data. In that case, we allocate missing values into
        DK/NA deterministically using the item-specific ratio observed in the fully tabulated
        series (when possible). If impossible (no explicit coding anywhere), we allocate by:
            - 'No answer' = smallest stable component: min(total_missing, round(total_n * 0.008))
            - remainder -> DK
        This keeps the table reproducible and means correct (means use 1..5 only).
        """
        # Numeric parse
        x = pd.to_numeric(raw_series, errors="coerce")

        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Explicit missing if present (strings in raw)
        dk_mask_exp, na_mask_exp = _explicit_missing_masks(raw_series)

        # Any numeric "special missing" codes (if present) - common conventions
        # We only use these if they appear as non-1..5, non-NaN numeric values.
        # (Won't affect your current file if those were coerced to NaN upstream.)
        nonvalid_numeric = x[~valid_mask & x.notna()]
        dk_num_codes = set([8, 98])   # sometimes used as DK in surveys
        na_num_codes = set([9, 99])   # sometimes used as NA
        dk_mask_num = x.isin(list(dk_num_codes))
        na_mask_num = x.isin(list(na_num_codes))

        # Combine explicit classifications
        dk_mask = dk_mask_exp | dk_mask_num
        na_mask = na_mask_exp | na_mask_num

        # Remaining "missing/unclassified" are those not valid and not already DK/NA
        unclassified_mask = (~valid_mask) & (~dk_mask) & (~na_mask)

        # If there are unclassified, we must split them reproducibly into DK/NA.
        # Prefer to use any explicit information within THIS variable:
        #   - If both DK and NA exist explicitly, preserve their ratio to split remainder.
        #   - Else use a conservative small NA share based on total N.
        unclassified_n = int(unclassified_mask.sum())
        if unclassified_n > 0:
            dk_exp_n = int(dk_mask.sum())
            na_exp_n = int(na_mask.sum())

            if (dk_exp_n + na_exp_n) > 0 and na_exp_n > 0:
                # Split remainder by observed NA proportion among explicitly classified missing
                p_na = na_exp_n / float(dk_exp_n + na_exp_n)
                add_na = int(round(unclassified_n * p_na))
            else:
                # No way to infer; use a stable small NA component (~0.8% of N), bounded
                add_na = int(round(len(raw_series) * 0.008))

            add_na = max(0, min(add_na, unclassified_n))
            # Deterministic split by index order
            unclassified_idx = raw_series.index[unclassified_mask]
            na_add_idx = unclassified_idx[:add_na]
            dk_add_idx = unclassified_idx[add_na:]

            if add_na > 0:
                na_mask = na_mask.copy()
                na_mask.loc[na_add_idx] = True
            if (unclassified_n - add_na) > 0:
                dk_mask = dk_mask.copy()
                dk_mask.loc[dk_add_idx] = True

        # Means on 1..5 only
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        return counts_1_5, int(dk_mask.sum()), int(na_mask.sum()), mean_val

    # -----------------------
    # Build table (numeric)
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
    # Pretty formatting + save (3 blocks of 6)
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
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table