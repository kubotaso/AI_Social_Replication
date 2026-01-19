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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {v}")

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
    # Missing-category detection
    # -----------------------
    def _as_str_series(x):
        # keep pandas NA as <NA> in string dtype
        return x.astype("string")

    def _missing_type_from_string_tokens(s_up):
        # Return two boolean masks for DK and NA(n) based on common encodings
        # Accepts: "[NA(d)]", "NA(d)", "NA(D)", "DON'T KNOW", "DONT KNOW", "DK"
        #          "[NA(n)]", "NA(n)", "NO ANSWER", "NA" (careful), "NO ANS"
        dk = (
            s_up.str.contains(r"\[NA\(\s*D\s*\)\]", regex=True, na=False)
            | s_up.str.contains(r"\bNA\(\s*D\s*\)\b", regex=True, na=False)
            | s_up.str.contains(r"\bDON['’]?\s*T\s+KNOW\b", regex=True, na=False)
            | s_up.str.contains(r"\bDONT\s+KNOW\b", regex=True, na=False)
            | s_up.str.fullmatch(r"\s*DK\s*", na=False)
        )
        na_ans = (
            s_up.str.contains(r"\[NA\(\s*N\s*\)\]", regex=True, na=False)
            | s_up.str.contains(r"\bNA\(\s*N\s*\)\b", regex=True, na=False)
            | s_up.str.contains(r"\bNO\s+ANSWER\b", regex=True, na=False)
            | s_up.str.contains(r"\bNO\s+ANS\b", regex=True, na=False)
        )
        return dk, na_ans

    def _tabulate_one(raw):
        # Parse numeric values for substantive categories
        x_num = pd.to_numeric(raw, errors="coerce")
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        # Try to classify missing types using explicit string tokens (if present)
        s = _as_str_series(raw)
        s_up = s.str.strip().str.upper()

        dk_mask, na_mask = _missing_type_from_string_tokens(s_up)

        # Only consider explicit DK/NA labels when the cell is not a valid 1..5
        nonvalid = ~x_num.isin([1, 2, 3, 4, 5])
        dk_mask = dk_mask & nonvalid
        na_mask = na_mask & nonvalid

        # If any other nonvalid values exist, they are "other missing/invalid" and cannot be split
        other_missing = nonvalid & ~(dk_mask | na_mask)

        # Determine whether this CSV preserves separable missing types:
        # - If there exist any nonvalid values and none are classifiable as DK or NA, it's collapsed.
        # - If there are some classifiable, but also a remainder of "other_missing", splitting is ambiguous.
        if int(nonvalid.sum()) > 0:
            if int((dk_mask | na_mask).sum()) == 0:
                raise ValueError(
                    "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                    "this CSV export does not preserve distinguishable missing categories "
                    "(e.g., '[NA(d)]' and '[NA(n)]' or labeled DK/No answer strings)."
                )
            if int(other_missing.sum()) > 0:
                # ambiguous remainder: cannot be allocated to DK vs NA without assumptions
                raise ValueError(
                    "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                    "found missing/invalid values that are not labeled as DK or No answer in the CSV export."
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
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var])

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