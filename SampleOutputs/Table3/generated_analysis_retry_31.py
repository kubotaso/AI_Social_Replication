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
        "(M) Don’t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Missing-code handling
    # -----------------------
    # We must compute DK vs NA from raw data.
    # If the file includes explicit GSS-style NA labels (e.g., "[NA(d)]", "[NA(n)]"),
    # we use them directly. If it does not, we *cannot* truthfully split missing into DK vs NA,
    # so we default to: DK = all missing/non-1..5; NA = 0 (and document in the output file).
    # This avoids the prior runtime errors and avoids fabricating a split.
    def _as_upper_string(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_counts(raw: pd.Series):
        s_up = _as_upper_string(raw)

        # Accept several plausible encodings in extracts
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b|\bDON'?T\s+KNOW\b|\bDONT\s+KNOW\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b|\bNO\s+ANSWER\b", regex=True, na=False)
        return dk_mask, na_mask

    def _compute_components(raw: pd.Series):
        # numeric valid responses (1..5)
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit missing codes (if present as strings)
        dk_mask, na_mask = _explicit_missing_counts(raw)
        has_explicit = (dk_mask.any() or na_mask.any())

        # "missing pool" = everything not 1..5 (including true NaN)
        missing_pool = valid.isna()

        if has_explicit:
            # If explicit codes exist, use them.
            # Any other missing/non-1..5 not explicitly classified is treated as DK (conservative).
            other_missing = missing_pool & ~(dk_mask | na_mask)
            dk_mask = dk_mask | other_missing
            return valid, dk_mask, na_mask, True
        else:
            # No explicit split possible from this extract
            dk_mask = missing_pool.copy()
            na_mask = pd.Series(False, index=raw.index)
            return valid, dk_mask, na_mask, False

    # -----------------------
    # Build table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    any_explicit_any_var = False
    any_unsplittable_any_var = False
    unsplittable_vars = []

    for genre_label, var in genre_map:
        raw = df[var]
        valid, dk_mask, na_mask, split_ok = _compute_components(raw)
        any_explicit_any_var = any_explicit_any_var or split_ok
        if not split_ok:
            any_unsplittable_any_var = True
            unsplittable_vars.append(var)

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
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # -----------------------
    # Save human-readable text output
    # -----------------------
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (paper layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i: i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        if any_unsplittable_any_var and not any_explicit_any_var:
            f.write(
                "NOTE: This extract does not contain explicit missing-code labels that distinguish\n"
                "'(M) Don’t know much about it' vs '(M) No answer'. In this case, the table reports:\n"
                "  (M) Don’t know much about it = all non-1..5 responses (including NaN)\n"
                "  (M) No answer = 0\n"
                "To reproduce the published DK/NA split, use a GSS extract that preserves NA(d) and NA(n).\n\n"
            )
        elif any_unsplittable_any_var:
            f.write(
                "NOTE: Some variables lacked explicit DK/NA labels; for those, all missing/non-1..5 were\n"
                "counted under '(M) Don’t know much about it' and '(M) No answer' set to 0.\n"
                f"Affected variables: {', '.join(unsplittable_vars)}\n\n"
            )

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table