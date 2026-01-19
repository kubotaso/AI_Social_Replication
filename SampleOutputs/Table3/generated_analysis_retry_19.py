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

    # Filter to 1993 only
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

    # -----------------------
    # Helpers for missing codes
    # -----------------------
    def _stringify(series: pd.Series) -> pd.Series:
        # Keep NA as <NA> for string dtype; do not fill with "NAN"
        return series.astype("string")

    def _missing_kind_masks(raw_series: pd.Series):
        """
        Identify GSS missing categories when they are present as strings like "[NA(d)]", "NA(d)", etc.
        Returns:
          dk_mask: Don't know much about it  (NA(d))
          na_mask: No answer                (NA(n))
          other_missing_mask: other missing codes (NA(r), NA(i), ...), plus blank/NaN if any
          numeric: numeric conversion of raw (NaN if non-numeric)
        """
        s_str = _stringify(raw_series).str.strip()
        s_up = s_str.str.upper()

        # explicit NA(d) and NA(n)
        dk_mask = s_up.str.contains(r"\[?NA\(D\)\]?", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[?NA\(N\)\]?", regex=True, na=False)

        # any other explicit NA(<letter>) missing code
        any_na_code_mask = s_up.str.contains(r"\[?NA\([A-Z]\)\]?", regex=True, na=False)
        other_missing_mask = any_na_code_mask & ~(dk_mask | na_mask)

        numeric = pd.to_numeric(raw_series, errors="coerce")
        # blank/real NaN in the file also counts as missing (but unsplit unless explicit codes exist)
        other_missing_mask = other_missing_mask | numeric.isna() | s_str.isna()

        # Ensure disjointness for dk/na vs other
        other_missing_mask = other_missing_mask & ~(dk_mask | na_mask)

        return dk_mask, na_mask, other_missing_mask, numeric

    def _compute_counts_and_mean(raw_series: pd.Series):
        dk_mask, na_mask, other_missing_mask, numeric = _missing_kind_masks(raw_series)

        # valid substantive codes
        valid = numeric.where(numeric.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # If explicit NA(d)/NA(n) are present, use them.
        # If not, we cannot reliably split NaN into DK vs NA from this extract.
        has_explicit_split = (dk_mask | na_mask).any()

        if has_explicit_split:
            dk_count = int(dk_mask.sum())
            na_count = int(na_mask.sum())
            # Any other missing stays unreported (Table 3 only has two M rows)
        else:
            # No explicit codes: dataset likely uses numeric codes for DK/NA.
            # In this case, look for typical GSS missing numeric codes outside 1..5
            # (e.g., 8/9, 0, 98/99). If found, split by frequency:
            # - NA tends to be the smallest missing category; DK larger.
            other_num_missing = numeric.notna() & (~numeric.isin([1, 2, 3, 4, 5]))
            if other_num_missing.any():
                vc = numeric.loc[other_num_missing].value_counts()
                if len(vc) == 1:
                    # Only one missing code present; treat it as DK (can't separate NA)
                    dk_code = vc.index[0]
                    dk_count = int((numeric == dk_code).sum())
                    na_count = 0
                else:
                    dk_code = vc.idxmax()
                    na_code = vc.idxmin()
                    dk_count = int((numeric == dk_code).sum())
                    na_count = int((numeric == na_code).sum())
            else:
                # Only blank NaNs: cannot split; keep both at 0 rather than fabricating.
                dk_count = 0
                na_count = 0

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_count, na_count, mean_val

    # -----------------------
    # Build Table 3
    # -----------------------
    col_labels = [g[0] for g in genre_map]
    table = pd.DataFrame(index=row_labels, columns=col_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        counts_1_5, dk_count, na_count, mean_val = _compute_counts_and_mean(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Format for human-readable output
    # -----------------------
    formatted = table.copy()

    # counts as integers; mean to 2 decimals (with trailing zeros)
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else str(int(round(float(v))))
            )

    # Add stub column
    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # -----------------------
    # Save as 3 blocks (6 genres each)
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [col_labels[i : i + 6] for i in range(0, len(col_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            f.write(display.loc[:, cols_with_stub].to_string(index=False))
            f.write("\n\n")

        # Diagnostics: warn if DK/NA rows are all zero (helps catch missing-code parsing issues)
        dk_row = table.loc["(M) Don't know much about it"]
        na_row = table.loc["(M) No answer"]
        if (dk_row.fillna(0).sum() == 0) and (na_row.fillna(0).sum() == 0):
            f.write(
                "NOTE: DK and No answer counts are all zero. This usually indicates the extract does not preserve\n"
                "separate missing codes (e.g., [NA(d)]/[NA(n)]), or they are encoded differently (numeric codes).\n"
                "If your dataset uses numeric missing codes (e.g., 8/9, 98/99), adjust the missing-code mapping.\n"
            )

    return table