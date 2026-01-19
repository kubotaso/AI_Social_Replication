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

    # Ensure all required columns exist
    missing_vars = [var for _, var in genre_map if var not in df.columns]
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
    # Missing handling
    # -----------------------
    # We must compute DK vs NA from raw data, not hard-code paper numbers.
    #
    # This CSV appears to store substantive answers as 1..5 and collapses all missing
    # (DK/NA/etc.) to NaN (or other non-1..5). In that case DK vs NA is not identifiable.
    #
    # Strategy:
    #   1) If explicit tokens exist (e.g., "[NA(d)]", "[NA(n)]", "dk", "no answer"),
    #      use them.
    #   2) Else, if numeric missing codes exist (e.g., 8/9, 98/99), use them.
    #   3) Else, fall back to: DK = all missing/non-1..5, NA = 0
    #      and write a note to the output file warning that NA cannot be separated.
    #
    # This avoids runtime errors and never uses paper numbers.

    def _as_str_series(s):
        # keep <NA> as <NA> for string dtype; normalize whitespace/case
        return s.astype("string").str.strip().str.lower()

    def _detect_missing_masks(raw):
        """
        Returns:
            valid (float series with only 1..5 or NaN),
            dk_mask (bool series),
            na_mask (bool series),
            note (str or None)
        """
        # numeric parse first
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # identify non-valid positions (candidate missing)
        nonvalid = valid.isna()

        # If there are no nonvalid, trivial
        if int(nonvalid.sum()) == 0:
            return valid, pd.Series(False, index=raw.index), pd.Series(False, index=raw.index), None

        # Try numeric code separation (common survey conventions)
        # (Not guaranteed for this instrument; harmless if absent)
        # We only use codes where x is not NaN.
        dk_num_codes = {8, 98}
        na_num_codes = {9, 99}
        dk_mask_num = x.isin(list(dk_num_codes))
        na_mask_num = x.isin(list(na_num_codes))

        if int((dk_mask_num | na_mask_num).sum()) > 0:
            # Any remaining nonvalid that is NaN/other goes to DK (cannot further separate)
            other_nonvalid = nonvalid & ~(dk_mask_num | na_mask_num)
            dk_mask = dk_mask_num | other_nonvalid
            na_mask = na_mask_num
            note = None
            # If we had to dump additional unclassifiable into DK, note it
            if int(other_nonvalid.sum()) > 0:
                note = "Some missing/non-1..5 values were unclassifiable; counted under DK."
            return valid, dk_mask, na_mask, note

        # Try explicit string tokens
        s = _as_str_series(raw)

        # Only attempt token parsing where raw is not numeric valid (avoid misclassifying "1" etc.)
        s_nonvalid = s.where(nonvalid, pd.NA)

        # DK tokens
        dk_tokens = [
            "[na(d)]",
            "na(d)",
            "don't know",
            "dont know",
            "dk",
            "don't know much",
            "dont know much",
        ]
        # NA tokens
        na_tokens = [
            "[na(n)]",
            "na(n)",
            "no answer",
            "na",
            "n/a",
        ]
        # refused/other: if present, treat as NA (closest to "no answer")
        refused_tokens = [
            "[na(r)]",
            "na(r)",
            "refused",
        ]

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        # contains checks (safe with NA)
        for t in dk_tokens:
            dk_mask = dk_mask | s_nonvalid.str.contains(t, regex=False, na=False)
        for t in na_tokens:
            na_mask = na_mask | s_nonvalid.str.contains(t, regex=False, na=False)
        for t in refused_tokens:
            na_mask = na_mask | s_nonvalid.str.contains(t, regex=False, na=False)

        if int((dk_mask | na_mask).sum()) > 0:
            other_nonvalid = nonvalid & ~(dk_mask | na_mask)
            # anything still unclassified -> DK (unknown type of missing)
            dk_mask = dk_mask | other_nonvalid
            note = None
            if int(other_nonvalid.sum()) > 0:
                note = "Some missing/non-1..5 values were unclassifiable; counted under DK."
            return valid, dk_mask, na_mask, note

        # Fallback: cannot separate; all missing counted as DK
        dk_mask = nonvalid.copy()
        na_mask = pd.Series(False, index=raw.index)
        note = "DK vs No answer not distinguishable in this CSV export; all missing counted as DK, No answer set to 0."
        return valid, dk_mask, na_mask, note

    # -----------------------
    # Build table (numeric)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")
    notes = {}

    for genre_label, var in genre_map:
        raw = df[var]
        valid, dk_mask, na_mask, note = _detect_missing_masks(raw)
        if note:
            notes[var] = note

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable text output (3 blocks of 6)
    # -----------------------
    formatted = table.copy()

    # Format: counts as integers; mean to 2 decimals
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
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: DK/No answer counts are computed from raw codes if distinguishable; otherwise all missing are counted as DK.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

        if notes:
            f.write("Missing-category parsing notes (by variable):\n")
            for _, var in genre_map:
                if var in notes:
                    f.write(f"- {var}: {notes[var]}\n")

    return table