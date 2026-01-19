def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import re

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

    missing_other_tokens = [
        "[NA(I)]", "[NA(M)]", "[NA(R)]", "[NA(S)]", "[NA(U)]", "[NA(X)]", "[NA(Y)]",
        "[NA(Z)]", "[NA(P)]", "[NA(J)]"
    ]

    def _series_as_clean_str(s):
        # Keep NA as <NA> so string ops behave; uppercase and strip
        return s.astype("string").str.strip().str.upper()

    def _mask_token(s_str, token):
        # literal token match inside cell (handles cases like "[NA(d)]" or "NA(d)")
        t = token.upper()
        # escape for regex literal match
        pat = re.escape(t)
        return s_str.str.contains(pat, regex=True, na=False)

    def _extract_dk_na_masks(raw):
        """
        Return (dk_mask, na_mask, explicit_found)
        explicit_found True iff we can see distinguishable DK/NA markers/codes.
        """
        s_str = _series_as_clean_str(raw)

        # Common explicit encodings we might see
        dk_tokens = ["[NA(D)]", "NA(D)", "DK", "DON'T KNOW", "DONT KNOW"]
        na_tokens = ["[NA(N)]", "NA(N)", "NO ANSWER", "NOANSWER"]

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        found_any = False
        for t in dk_tokens:
            m = _mask_token(s_str, t)
            if m.any():
                found_any = True
                dk_mask |= m

        for t in na_tokens:
            m = _mask_token(s_str, t)
            if m.any():
                found_any = True
                na_mask |= m

        # Also handle numeric-coded DK/NA if present (common in some extracts: 8/9 or 98/99)
        # We only treat these as explicit if they appear in the data.
        x = pd.to_numeric(raw, errors="coerce")
        numeric_dk_codes = [8, 98]
        numeric_na_codes = [9, 99]
        for code in numeric_dk_codes:
            m = x.eq(code)
            if m.any():
                found_any = True
                dk_mask |= m
        for code in numeric_na_codes:
            m = x.eq(code)
            if m.any():
                found_any = True
                na_mask |= m

        return dk_mask, na_mask, found_any

    def _tabulate_one(raw, varname):
        """
        Produce:
        - counts for 1..5
        - DK count
        - NA count
        - mean on 1..5
        """
        x = pd.to_numeric(raw, errors="coerce")

        # Valid 1..5
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Identify explicit DK/NA if possible
        dk_mask, na_mask, explicit_found = _extract_dk_na_masks(raw)

        # Missing/non-1..5 pool (includes NaN after numeric conversion, and any non-1..5 numeric codes)
        nonvalid_mask = ~x.isin([1, 2, 3, 4, 5]) | x.isna()

        # Exclude explicit DK/NA from "other missing"
        other_missing_mask = nonvalid_mask & ~(dk_mask | na_mask)

        if other_missing_mask.any():
            # If there are other missing categories beyond DK/NA, Table 3 doesn't show them separately.
            # We fold them into "No answer" (a conventional bucket), BUT ONLY if DK/NA are
            # distinguishable (explicit). If DK/NA are not distinguishable, we cannot split.
            if not explicit_found:
                # Provide actionable error: cannot split DK vs NA from this export
                example_nonvalid = int(nonvalid_mask.sum())
                raise ValueError(
                    f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts "
                    f"for {varname}: this CSV export does not preserve distinguishable DK/NA categories "
                    f"(e.g., '[NA(d)]'/'[NA(n)]' or distinct numeric codes like 8/9). "
                    f"Found {example_nonvalid} missing/non-1..5 values."
                )
            # Fold any other missing into NA
            na_mask = na_mask | other_missing_mask

        # If explicit codes are present but DK/NA are both absent and there are no other missings,
        # that's fine (they're just zero).
        # If explicit codes are absent and there are no nonvalids, also fine (zeros).
        if (not explicit_found) and nonvalid_mask.any():
            # We have nonvalids but cannot classify them into DK vs NA
            example_nonvalid = int(nonvalid_mask.sum())
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts "
                f"for {varname}: this CSV export does not preserve distinguishable DK/NA categories "
                f"(e.g., '[NA(d)]'/'[NA(n)]' or numeric 8/9). Found {example_nonvalid} missing/non-1..5 values."
            )

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table
    # -----------------------
    col_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=col_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

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
    # Save human-readable text (3 blocks of 6)
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
    blocks = [col_labels[i:i + 6] for i in range(0, len(col_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            f.write(display.loc[:, ["Attitude"] + cols].to_string(index=False))
            f.write("\n\n")

    return table