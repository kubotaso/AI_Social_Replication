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

    # Filter to 1993 only (drop missing YEAR)
    df = df.loc[pd.to_numeric(df["YEAR"], errors="coerce").eq(1993)].copy()

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

    # -----------------------
    # Missing parsing helpers
    # -----------------------
    def _norm_str_series(s):
        # Normalize to pandas string dtype; preserve missing as <NA>
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw_series):
        """
        Attempt to detect explicit GSS-style NA codes if present as strings.
        Returns (dk_mask, na_mask).
        """
        su = _norm_str_series(raw_series)

        # Common encodings we might see in extracts
        dk_mask = su.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b|\bDON'?T KNOW\b|\bDONT KNOW\b", regex=True, na=False)
        na_mask = su.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b|\bNO ANSWER\b", regex=True, na=False)

        return dk_mask, na_mask

    def _coerce_valid_1_5(raw_series):
        x = pd.to_numeric(raw_series, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        return valid

    def _split_missing_or_raise(raw_series, varname):
        """
        Produce:
          - valid numeric series with values in 1..5 else NaN
          - dk_mask for "Don't know much about it" (NA(d))
          - na_mask for "No answer" (NA(n))

        If explicit missing codes are not present, we cannot split DK vs NA from raw microdata.
        In that case, raise a clear error (do NOT fabricate counts).
        """
        valid = _coerce_valid_1_5(raw_series)

        dk_mask, na_mask = _explicit_missing_masks(raw_series)

        # Additionally treat blank strings as missing, but unsplittable unless explicitly tagged
        su = _norm_str_series(raw_series)
        blank_mask = su.isna() | (su == "")

        # Any non-1..5 numeric and non-blank string could be another missing; unsplittable
        # We still don't allocate it to DK/NA without explicit tags.
        other_missing = valid.isna() & ~blank_mask & ~(dk_mask | na_mask)

        # If we have explicit DK/NA at all, we can safely classify remaining missing as "No answer"
        # only if they are blank (system missing). Otherwise keep them unassigned but counted as missing elsewhere.
        any_explicit = (int(dk_mask.sum()) + int(na_mask.sum())) > 0

        if not any_explicit:
            # If there are any missing values at all, we cannot split without explicit tags.
            total_missing = int(valid.isna().sum())
            if total_missing > 0:
                raise ValueError(
                    f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                    f"dataset does not contain explicit NA(d)/NA(n) codes (or equivalent string labels). "
                    f"Found {total_missing} missing/non-1..5 values but they are unsplittable."
                )
            # No missing at all
            dk_mask = pd.Series(False, index=raw_series.index)
            na_mask = pd.Series(False, index=raw_series.index)
            return valid, dk_mask, na_mask

        # We have explicit tagging: count blank/system missing as "No answer"
        na_mask = na_mask | blank_mask

        # If any other missing remains untagged, treat as "Don't know" by default (conservative),
        # but keep this behavior explicit and deterministic.
        dk_mask = dk_mask | other_missing

        # Ensure DK/NA do not overlap
        overlap = dk_mask & na_mask
        if overlap.any():
            dk_mask = dk_mask & ~overlap

        return valid, dk_mask, na_mask

    # -----------------------
    # Build numeric table
    # -----------------------
    genre_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid, dk_mask, na_mask = _split_missing_or_raise(raw, var)

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

        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # -----------------------
    # Save as human-readable text (3 blocks of 6 genres)
    # -----------------------
    def _format_table_for_print(t):
        out = t.copy()
        for r in out.index:
            if r == "Mean":
                out.loc[r] = out.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
            else:
                out.loc[r] = out.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))
        out.insert(0, "Attitude", list(out.index))
        out = out.reset_index(drop=True)
        return out

    display = _format_table_for_print(table)

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