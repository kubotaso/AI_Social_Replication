def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + filter to 1993
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
        "(M) Don't know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Missing-category detection (must be computed, not hard-coded)
    # -----------------------
    # Goal:
    # - Count valid codes 1..5
    # - Separately count:
    #     (M) Don't know much about it  == NA(d) or equivalent DK strings/codes
    #     (M) No answer                 == NA(n) or equivalent NA strings/codes
    #
    # This CSV extract sometimes collapses both to blank/NaN. In that case, the split is
    # not identifiable from the raw file; we then raise a clear error (rather than
    # silently inventing numbers or outputting zeros).
    #
    # If the file contains explicit encodings (e.g., "[NA(d)]", "NA(d)", "DK", etc.),
    # we will compute them.

    def _as_string_series(s):
        # keep <NA> as <NA>, preserve original tokens as best as possible
        return s.astype("string")

    def _detect_missing_masks(raw):
        """
        Returns:
          valid_num: numeric Series with only 1..5 kept, others NaN
          dk_mask: boolean mask for DK-about-it
          na_mask: boolean mask for No answer
          other_missing_mask: missings not classifiable as DK vs NA
        """
        s_str = _as_string_series(raw).str.strip()

        # numeric parse for substantive categories
        s_num = pd.to_numeric(s_str, errors="coerce")
        valid_num = s_num.where(s_num.isin([1, 2, 3, 4, 5]), np.nan)

        # masks for explicit NA tokens
        up = s_str.str.upper()

        # Common GSS-style explicit missing tokens that might survive export
        dk_tokens = [
            r"\[NA\(D\)\]",
            r"\bNA\(D\)\b",
            r"\bNAD\b",
            r"\bDON'?T\s+KNOW\b",
            r"\bDONT\s+KNOW\b",
            r"\bDK\b",
        ]
        na_tokens = [
            r"\[NA\(N\)\]",
            r"\bNA\(N\)\b",
            r"\bNAN\b",  # sometimes appears as literal "NaN" string; treated as NA (no answer) only if it's a string token
            r"\bNO\s+ANSWER\b",
        ]

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        # Identify explicit DK / NA strings
        # Note: literal pandas missing (<NA>) should not match regex; handle separately below.
        for pat in dk_tokens:
            dk_mask = dk_mask | up.str.contains(pat, regex=True, na=False)
        for pat in na_tokens:
            # Treat string "NAN" as "no answer" only if it is a string token, not true numeric NaN
            na_mask = na_mask | up.str.contains(pat, regex=True, na=False)

        # Some exports use numeric special codes; if present, classify cautiously.
        # We only classify if we see those codes in the data (otherwise ignore).
        # (These are not from the paper; they're generic survey-missing conventions.)
        special_num = pd.to_numeric(s_str, errors="coerce")
        if special_num.notna().any():
            # Typical patterns: 8=DK, 9=NA; 98/99; 0; -1/-2, etc.
            dk_mask = dk_mask | special_num.isin([8, 98, -8, -98])
            na_mask = na_mask | special_num.isin([9, 99, -9, -99])

        # Missing pool (anything not in 1..5)
        missing_pool = valid_num.isna()

        # If something is explicitly DK/NA, it's missing, but should not be counted as "other"
        other_missing_mask = missing_pool & ~(dk_mask | na_mask)

        return valid_num, dk_mask, na_mask, other_missing_mask

    # -----------------------
    # Build table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    any_identifiable_split = False
    unclassifiable_details = []

    for genre_label, var in genre_map:
        raw = df[var]
        valid_num, dk_mask, na_mask, other_missing_mask = _detect_missing_masks(raw)

        # Frequencies for 1..5
        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        other_n = int(other_missing_mask.sum())

        if dk_n > 0 or na_n > 0:
            any_identifiable_split = True

        if other_n > 0:
            # If we cannot classify other missing into DK vs NA, we cannot reproduce Table 3 correctly.
            # We record details and later raise a single informative error.
            unclassifiable_details.append((var, other_n))

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

    # If the export collapsed DK and NA into generic NA, we cannot split them.
    # We must not fabricate the split.
    if len(unclassifiable_details) > 0:
        # If there were *no* explicit DK/NA codes anywhere, be explicit about the root cause.
        if not any_identifiable_split:
            total_unclass = sum(n for _, n in unclassifiable_details)
            example_vars = ", ".join([f"{v}({n})" for v, n in unclassifiable_details[:6]])
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                "this CSV export does not preserve distinguishable missing categories (e.g., '[NA(d)]' and '[NA(n)]' "
                "or distinct numeric codes like 8/9). "
                f"Found {total_unclass} missing/non-1..5 values across items that are not classifiable. "
                f"Examples: {example_vars}. Re-export data preserving DK vs No-answer codes."
            )
        else:
            # Mixed situation: some items have explicit splits, others have ambiguous missings.
            example_vars = ", ".join([f"{v}({n})" for v, n in unclassifiable_details[:6]])
            raise ValueError(
                "Cannot fully reproduce Table 3 because some items contain missing/non-1..5 values that are not "
                "classifiable into '(M) Don't know much about it' vs '(M) No answer' from this CSV export. "
                f"Examples: {example_vars}. Re-export with explicit missing codes for all items."
            )

    # -----------------------
    # Formatting + save
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
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table