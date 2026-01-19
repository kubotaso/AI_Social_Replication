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

    # Ensure all required columns exist (case-insensitive handled by uppercasing)
    missing_vars = [v for _, v in genre_map if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required variables in CSV: {missing_vars}")

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
    # Missing code detection
    # -----------------------
    # We must compute DK vs No answer from raw data, not from hard-coded counts.
    # This CSV often does NOT preserve distinct DK vs NA; in that case, it's impossible
    # to split missing into these two rows. We therefore:
    #   - use explicit codes if present (string tokens like "[NA(d)]"/"[NA(n)]",
    #     or common numeric conventions 8/9, 98/99, 0/9, etc.)
    #   - otherwise raise a clear error (instead of guessing / merging them)
    #
    # IMPORTANT: Means are computed on valid 1..5 only, regardless of missing coding.

    DK_TOKENS = [
        "[NA(D)]", "NA(D)", "DON'T KNOW", "DONT KNOW", "DK", "DON’T KNOW", "DONTKNOW",
        "DON'T KNOW MUCH", "DONT KNOW MUCH", "DON’T KNOW MUCH"
    ]
    NA_TOKENS = [
        "[NA(N)]", "NA(N)", "NO ANSWER", "NA", "N/A", "NOANSWER"
    ]
    REF_TOKENS = ["[NA(R)]", "NA(R)", "REFUSED", "REFUSE"]
    SKIP_TOKENS = ["[NA(S)]", "NA(S)", "SKIPPED"]
    IAP_TOKENS = ["[NA(I)]", "NA(I)", "IAP"]
    OTHER_NA_TOKENS = ["[NA(M)]", "NA(M)", "[NA(U)]", "NA(U)", "[NA(X)]", "NA(X)", "[NA(Y)]", "NA(Y)", "[NA(Z)]", "NA(Z)", "[NA(P)]", "NA(P)", "[NA(J)]", "NA(J)"]

    def _as_str_series(x: pd.Series) -> pd.Series:
        return x.astype("string").str.strip().str.upper()

    def _mask_any_token(s_up: pd.Series, tokens) -> pd.Series:
        if not tokens:
            return pd.Series(False, index=s_up.index)
        # Build a single regex OR, escaping special regex chars except brackets/parentheses are escaped by re via literal
        import re
        pat = "|".join(re.escape(t) for t in tokens)
        return s_up.str.contains(pat, regex=True, na=False)

    def _classify_missing(raw: pd.Series):
        """
        Returns:
          valid_num: float series with 1..5 else NaN
          dk_mask: boolean series
          na_mask: boolean series
          other_missing_mask: boolean series (any remaining missing/non-1..5)
          diag: dict diagnostic counts
        """
        s_up = _as_str_series(raw)

        # Numeric parse for substantive values
        x = pd.to_numeric(raw, errors="coerce")
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # String-token classification (explicit)
        dk_mask = _mask_any_token(s_up, DK_TOKENS)
        na_mask = _mask_any_token(s_up, NA_TOKENS)

        # Numeric-code classification (only applies where value is numeric and NOT 1..5)
        # Common conventions for GSS-style exports can vary; we include these only if present.
        # If these codes aren't present, masks will be all-False.
        num = pd.to_numeric(raw, errors="coerce")
        non15_num = num.notna() & (~num.isin([1, 2, 3, 4, 5]))

        # Candidate DK/NA numeric codes (kept minimal and conservative)
        dk_num_mask = non15_num & num.isin([8, 98])   # often DK
        na_num_mask = non15_num & num.isin([9, 99])   # often NA

        dk_mask = dk_mask | dk_num_mask
        na_mask = na_mask | na_num_mask

        # Other explicit missing-like tokens (treated as "other missing"; not shown in table rows)
        other_token_mask = (
            _mask_any_token(s_up, REF_TOKENS)
            | _mask_any_token(s_up, SKIP_TOKENS)
            | _mask_any_token(s_up, IAP_TOKENS)
            | _mask_any_token(s_up, OTHER_NA_TOKENS)
        )

        # Anything not 1..5 that is NA/NaN/other token/numeric non-1..5 is missing-ish
        missing_any = valid_num.isna()
        other_missing_mask = missing_any & ~(dk_mask | na_mask)

        # If something is a known "other missing" token, keep it in other_missing_mask
        other_missing_mask = other_missing_mask | other_token_mask

        diag = {
            "n_total": int(len(raw)),
            "n_valid_1_5": int(valid_num.notna().sum()),
            "n_missing_any_non1_5": int(missing_any.sum()),
            "n_dk": int(dk_mask.sum()),
            "n_na": int(na_mask.sum()),
            "n_other_missing": int(other_missing_mask.sum()),
        }
        return valid_num, dk_mask, na_mask, other_missing_mask, diag

    def _tabulate_one(raw: pd.Series, varname: str):
        valid_num, dk_mask, na_mask, other_missing_mask, diag = _classify_missing(raw)

        # If we have missing beyond 1..5 but cannot classify into DK/NA, we cannot reproduce the table's two M-rows.
        unclassified = int((valid_num.isna() & ~(dk_mask | na_mask) & ~other_missing_mask).sum())
        # Note: other_missing_mask includes explicit tokens like refused/skipped; those are also not part of DK/NA rows.
        # If they exist, Table 3 does not list them; we treat them as unclassified for Table 3 purposes too.
        extra_missing = int(other_missing_mask.sum())

        if (diag["n_missing_any_non1_5"] > 0) and ((diag["n_dk"] + diag["n_na"]) == 0):
            # Most common: CSV contains plain NaN for both DK and NA; impossible to split
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                f"this CSV export does not preserve distinguishable DK/NA categories. "
                f"Found {diag['n_missing_any_non1_5']} missing/non-1..5 values."
            )

        if unclassified > 0 or extra_missing > 0:
            raise ValueError(
                f"Cannot reproduce Table 3 missing rows for {varname}: "
                f"found missing/non-1..5 values that are not classifiable as DK/NA for this table "
                f"(unclassified={unclassified}, other_missing={extra_missing}). "
                f"Re-export data preserving DK vs No-answer codes (e.g., '[NA(d)]'/'[NA(n)]' or distinct numeric codes)."
            )

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid_num.mean()) if valid_num.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table
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
    # Save human-readable text
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
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: '(M)' rows require DK vs No-answer codes to be preserved in the CSV export.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table