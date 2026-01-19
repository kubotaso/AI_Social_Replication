def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Load and filter (1993 only)
    # ----------------------------
    df = pd.read_csv(data_source, low_memory=False)

    # standardize column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found.")

    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ----------------------------
    # Table 3 variables / labels
    # ----------------------------
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

    # ----------------------------
    # Helpers: robust parsing + missing split
    # ----------------------------
    def _norm_string(s: pd.Series) -> pd.Series:
        # Keep as pandas StringDtype; normalize whitespace/case.
        return s.astype("string").str.strip().str.upper()

    def _split_missing(raw: pd.Series):
        """
        Returns:
          valid_num: numeric Series with only 1..5; else NaN
          dk_mask: boolean mask for NA(d) / DK
          na_mask: boolean mask for NA(n) / no answer
        Requirements:
          - If explicit NA(d)/NA(n) exist (as strings), use them.
          - Else if explicit numeric codes exist for DK/NA (not 1..5), infer via mode heuristic:
              most common -> DK, second -> NA
            (works for typical GSS encodings such as 8/9; includes other NA-codes if present)
          - Else if all missing are plain NaN, we cannot split DK vs NA from microdata -> raise.
        """
        # Detect explicit bracketed codes in text, if present
        s_up = _norm_string(raw)

        dk_mask_txt = s_up.str.contains(r"\[?\s*NA\(D\)\s*\]?", regex=True, na=False)
        na_mask_txt = s_up.str.contains(r"\[?\s*NA\(N\)\s*\]?", regex=True, na=False)

        # Numeric parse
        x = pd.to_numeric(raw, errors="coerce")

        # Valid substantive
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If text-coded DK/NA exist, use them
        if int(dk_mask_txt.sum()) + int(na_mask_txt.sum()) > 0:
            # Ensure disjoint
            dk_mask = dk_mask_txt & ~na_mask_txt
            na_mask = na_mask_txt & ~dk_mask_txt
            return valid_num, dk_mask, na_mask

        # Identify explicit "non-substantive but numeric" codes (e.g., 8/9, 0, 98/99)
        # Anything numeric but not 1..5 is considered missing-type for the purpose of splitting.
        other_code_mask = x.notna() & (~x.isin([1, 2, 3, 4, 5]))

        if other_code_mask.any():
            codes = x.loc[other_code_mask].astype(int)
            vc = codes.value_counts()

            # Heuristic: DK tends to be more frequent than NA.
            # Choose most frequent as DK, second most frequent as NA if exists; rest fold into DK.
            dk_code = int(vc.index[0])
            na_code = int(vc.index[1]) if len(vc.index) > 1 else None

            dk_mask = x.astype("float64").eq(dk_code)
            na_mask = pd.Series(False, index=x.index)
            if na_code is not None:
                na_mask = x.astype("float64").eq(na_code)

            # Any remaining "other codes" (beyond dk/na) -> fold into DK to keep table 2-row structure
            remainder = other_code_mask & (~dk_mask) & (~na_mask)
            dk_mask = dk_mask | remainder

            # Plain NaN: if there are also explicit codes, fold NaN into DK (dominant bucket)
            dk_mask = dk_mask | x.isna()

            # Ensure disjoint
            na_mask = na_mask & (~dk_mask)
            return valid_num, dk_mask, na_mask

        # At this point, missingness is only NaN and there are no explicit codes to split.
        # We cannot compute DK vs NA separately from microdata.
        if x.isna().any():
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                "dataset does not contain explicit missing codes (e.g., NA(d)/NA(n) strings or distinct numeric codes)."
            )

        # No missing at all
        return valid_num, pd.Series(False, index=x.index), pd.Series(False, index=x.index)

    # ----------------------------
    # Build Table 3
    # ----------------------------
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required variable not found: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = _split_missing(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())
        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ----------------------------
    # Save human-readable text (3 blocks of 6 genres)
    # ----------------------------
    formatted = table.copy()

    for idx in formatted.index:
        if idx == "Mean":
            formatted.loc[idx] = formatted.loc[idx].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[idx] = formatted.loc[idx].map(
                lambda v: "" if pd.isna(v) else str(int(round(float(v))))
            )

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing categories.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            f.write(display.loc[:, cols_with_stub].to_string(index=False))
            f.write("\n\n")

    return table