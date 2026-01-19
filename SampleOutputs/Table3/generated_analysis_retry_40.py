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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {v}")

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        "(M) Don’t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Helpers: parse explicit GSS-style missing strings
    # -----------------------
    def _as_string_upper(s: pd.Series) -> pd.Series:
        # Keep pandas NA; convert others to string and normalize
        return s.astype("string").str.strip().str.upper()

    def _missing_masks_from_strings(raw: pd.Series):
        """
        Return boolean masks for DK=[NA(d)] and NA=[NA(n)] if present as strings.
        Accepts variants like '[NA(D)]', 'NA(D)', etc.
        """
        su = _as_string_upper(raw)
        dk = su.str.contains(r"\[NA\(\s*D\s*\)\]|\bNA\(\s*D\s*\)\b", regex=True, na=False)
        na = su.str.contains(r"\[NA\(\s*N\s*\)\]|\bNA\(\s*N\s*\)\b", regex=True, na=False)
        any_explicit = bool((dk | na).any())
        return dk, na, any_explicit

    def _compute_for_var(raw: pd.Series, varname: str):
        """
        Compute counts for 1..5, DK, NA, and mean (1..5 only).
        Requirement: DK and NA must be computable from raw data (explicit codes).
        If the extract collapsed them into generic NaN, they are not identifiable; we error.
        """
        # Numeric values (for 1..5 and mean)
        x = pd.to_numeric(raw, errors="coerce")

        # Valid substantive responses
        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        # Explicit missing codes if present as strings
        dk_mask_str, na_mask_str, any_explicit = _missing_masks_from_strings(raw)

        # If explicit codes are present, use them; treat any other non-1..5 as additional missing (unclassified),
        # but do NOT silently assign it to DK/NA (Table 3 requires DK vs NA separately).
        if any_explicit:
            dk_n = int(dk_mask_str.sum())
            na_n = int(na_mask_str.sum())

            # Detect other non-1..5 (including NaN) not accounted for by explicit DK/NA
            other_missing = (~valid_mask) & (~dk_mask_str) & (~na_mask_str)
            other_missing_n = int(other_missing.sum())

            if other_missing_n != 0:
                # Not safe to allocate; fail loudly so the user can export with all missing codes preserved.
                examples = (
                    _as_string_upper(raw.loc[other_missing])
                    .value_counts(dropna=False)
                    .head(10)
                    .to_dict()
                )
                raise ValueError(
                    f"{varname}: Found {other_missing_n} additional missing/non-1..5 values not labeled as "
                    f"[NA(d)] or [NA(n)] in the extract. Cannot allocate them to DK vs No answer. "
                    f"Examples (top 10): {examples}"
                )

        else:
            # No explicit codes in extract. Then DK vs NA cannot be identified.
            # If there are any missing/non-1..5 values at all, we must error (per feedback).
            missing_pool = ~valid_mask
            missing_n = int(missing_pool.sum())
            if missing_n != 0:
                # show some examples of non-1..5 entries if any non-null raw values exist there
                nonnull_other = raw.loc[missing_pool].dropna()
                examples = _as_string_upper(nonnull_other).value_counts().head(10).to_dict()
                raise ValueError(
                    f"Cannot compute separate '(M) Don’t know much about it' vs '(M) No answer' counts for {varname}: "
                    f"dataset extract does not preserve explicit NA(d)/NA(n) codes (or equivalent). "
                    f"Found {missing_n} missing/non-1..5 values. Non-null examples (top 10): {examples}"
                )
            dk_n = 0
            na_n = 0

        # Counts 1..5
        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table (numeric)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _compute_for_var(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Format for display (counts as ints; mean rounded to 2 dp)
    # -----------------------
    formatted = table.copy()
    for idx in formatted.index:
        if idx == "Mean":
            formatted.loc[idx] = formatted.loc[idx].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[idx] = formatted.loc[idx].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # -----------------------
    # Write text file in 3 blocks (6 cols each) like the paper layout
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts for response categories; Mean computed on 1–5 excluding all missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table