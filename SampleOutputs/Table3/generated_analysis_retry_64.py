def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + standardize
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
    # Missing-category handling
    # -----------------------
    # We must count DK vs No answer separately. This is only possible if the CSV
    # preserves distinct codes/labels for these two categories. We therefore:
    #  1) Parse explicit string encodings like "[NA(d)]" and "[NA(n)]" (any case/spacing).
    #  2) Parse common textual labels containing "don't know" vs "no answer".
    #  3) Parse common numeric codes (some GSS extracts use 8=DK, 9=NA; also 98/99).
    # If the file collapses all missings into blank/NaN, we raise a clear error rather than
    # fabricating a split (not allowed per instructions).

    _VALID = {1, 2, 3, 4, 5}

    def _as_str_series(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip()

    def _classify_missing(raw: pd.Series):
        # Return: numeric_valid (1..5 else NaN), dk_mask, na_mask, other_missing_mask
        s_str = _as_str_series(raw)
        s_up = s_str.str.upper()

        # numeric parsing for substantive + possible numeric DK/NA codes
        s_num = pd.to_numeric(s_str, errors="coerce")

        valid = s_num.where(s_num.isin(list(_VALID)), np.nan)

        # --- Explicit NA(.) encodings and text patterns ---
        # Codes in prompt: [NA(d)] = DK, [NA(n)] = No answer
        dk_mask = s_up.str.contains(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", regex=True, na=False)

        # Textual labels (robust to variants)
        # DK: "DON'T KNOW", "DONT KNOW", "DON’T KNOW", "DON'T KNOW MUCH"
        dk_mask = dk_mask | s_up.str.contains(r"\bDON[’']?T\b|\bDONT\b", regex=True, na=False) & s_up.str.contains(
            r"\bKNOW\b", regex=True, na=False
        )
        # No answer: "NO ANSWER"
        na_mask = na_mask | s_up.str.contains(r"\bNO\s+ANSWER\b", regex=True, na=False)

        # --- Numeric code fallbacks (only if present) ---
        # Common survey conventions; applied only when value is non-missing numeric.
        # We treat 8/98 as DK and 9/99 as No answer if they exist.
        dk_mask = dk_mask | s_num.isin([8, 98])
        na_mask = na_mask | s_num.isin([9, 99])

        # Anything that's neither valid nor classified but is missing/nonvalid
        nonvalid = ~s_num.isin(list(_VALID)) | s_num.isna()
        other_missing = nonvalid & ~(dk_mask | na_mask)

        return valid, dk_mask, na_mask, other_missing

    # -----------------------
    # Build table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    # Track whether we ever saw separable DK/NA codes
    saw_separable_any = False
    problems = []

    for genre_label, var in genre_map:
        raw = df[var]
        valid, dk_mask, na_mask, other_missing = _classify_missing(raw)

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        other_n = int(other_missing.sum())

        if (dk_n + na_n) > 0:
            saw_separable_any = True

        # If we have unclassified missing values, we cannot split them without inventing.
        # We allow other_n == 0. If other_n > 0 and dk/na are both zero, file has collapsed missings.
        if other_n > 0:
            problems.append((var, other_n, dk_n, na_n))

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
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # If any unclassified missing exist, we cannot produce a correct DK vs NA split.
    # Raise a helpful error; do not fabricate.
    if problems:
        # If we never saw separable codes, it's almost certainly collapsed NA in export.
        if not saw_separable_any:
            examples = ", ".join([f"{v}({n})" for v, n, _, _ in problems[:6]])
            total_other = sum(n for _, n, _, _ in problems)
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                "this CSV export does not preserve distinguishable missing categories. "
                f"Found {total_other} missing/non-1..5 values across items that are not classifiable. "
                f"Examples: {examples}. Re-export data preserving DK vs No-answer codes "
                "(e.g., '[NA(d)]' and '[NA(n)]' or explicit string labels/codes)."
            )
        else:
            # Mixed situation: some items have separable codes, but there are extra unclassified missings.
            examples = ", ".join([f"{v}(other_missing={n}, dk={dk}, na={na})" for v, n, dk, na in problems[:6]])
            raise ValueError(
                "Found missing values that are neither valid (1–5) nor classifiable as DK/No-answer. "
                "Cannot report separate missing rows without inventing a split. "
                f"Examples: {examples}. Ensure export preserves distinct DK and No-answer codes for all items."
            )

    # -----------------------
    # Save human-readable text (3 blocks of 6 columns)
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