def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Normalize columns to uppercase to match mapping
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")
    df = df.loc[df["YEAR"].eq(1993)].copy()

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

    # ---- Helpers to preserve and count DK vs No answer distinctly ----
    def _as_clean_string(s: pd.Series) -> pd.Series:
        # Keep actual NaN as <NA> so we can detect it
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw: pd.Series):
        """
        Detect explicit GSS-style missing tokens if present as strings.
        We purposefully do NOT rely on pandas NA parsing alone.
        """
        s = _as_clean_string(raw)

        # Common encodings seen across GSS exports
        dk_tokens = [
            "[NA(D)]", "NA(D)", "NA(D))", "DON'T KNOW", "DONT KNOW", "DK",
            "DON’T KNOW", "DONTKNOW", "DON'TKNOW", "DONT_KNOW", "DON'T_KNOW",
            "DON’T KNOW MUCH ABOUT IT", "DONT KNOW MUCH ABOUT IT", "DON'T KNOW MUCH ABOUT IT",
        ]
        na_tokens = [
            "[NA(N)]", "NA(N)", "NA(N))", "NO ANSWER", "NA", "N/A", "NOANSWER",
        ]

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        for t in dk_tokens:
            dk_mask |= s.eq(t)
        for t in na_tokens:
            na_mask |= s.eq(t)

        # Also accept bracketed single-letter NA(...) variants
        dk_mask |= s.str.contains(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", regex=True, na=False)
        na_mask |= s.str.contains(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", regex=True, na=False)

        return dk_mask, na_mask

    def _tabulate_one(raw: pd.Series, varname: str):
        """
        Returns:
          counts_1_5 (Series indexed 1..5),
          dk_n (int),
          na_n (int),
          mean_val (float)
        Rules:
          - Count 1..5 as substantive.
          - Count DK and NA ONLY if they are explicitly distinguishable in the raw file,
            OR if they are coded as distinct numeric codes (commonly 8/9).
          - If raw contains only generic NaN for both DK and NA, we cannot split; raise a clear error.
        """
        # numeric parsing for substantive + potential numeric missing codes
        x = pd.to_numeric(raw, errors="coerce")

        valid_mask = x.isin([1, 2, 3, 4, 5])
        counts_1_5 = x.loc[valid_mask].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).astype(int)

        # explicit string-coded missing
        dk_mask_str, na_mask_str = _explicit_missing_masks(raw)
        dk_n_str = int(dk_mask_str.sum())
        na_n_str = int(na_mask_str.sum())

        # numeric-coded missing (support common patterns if present)
        # Many GSS extracts use 8/9 for DK/NA on attitude items; we only use them if observed.
        observed_vals = set(pd.unique(x.dropna()).tolist())
        dk_code_candidates = [8, 98]
        na_code_candidates = [9, 99]

        dk_mask_num = pd.Series(False, index=raw.index)
        na_mask_num = pd.Series(False, index=raw.index)

        for c in dk_code_candidates:
            if c in observed_vals:
                dk_mask_num |= x.eq(c)
        for c in na_code_candidates:
            if c in observed_vals:
                na_mask_num |= x.eq(c)

        dk_n = dk_n_str + int(dk_mask_num.sum())
        na_n = na_n_str + int(na_mask_num.sum())

        # Remaining missing/unclassifiable pool
        # Everything that is not 1..5 and not classified as DK/NA is "other missing".
        classified_mask = valid_mask | dk_mask_str | na_mask_str | dk_mask_num | na_mask_num

        other_missing_n = int((~classified_mask).sum())

        # If there are unclassified missings, we cannot truthfully split into DK vs NA
        # because Table 3 requires those two distinct rows.
        if other_missing_n > 0:
            # Provide concrete diagnostics
            total_non_1_5 = int((~valid_mask).sum())
            examples = int(other_missing_n)
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                f"the CSV does not preserve distinguishable DK/NA categories for {varname}. "
                f"Non-1..5 values: {total_non_1_5}; Unclassified among them: {examples}. "
                "Re-export data preserving DK vs No-answer codes (e.g., '[NA(d)]'/'[NA(n)]' or numeric 8/9)."
            )

        mean_val = float(x.loc[valid_mask].mean()) if int(valid_mask.sum()) > 0 else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # ---- Build Table 3 structure ----
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

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

    # ---- Save human-readable output (3 blocks of 6 genres) ----
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    display = formatted.copy()
    display.insert(0, "Attitude", display.index)
    display = display.reset_index(drop=True)

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table