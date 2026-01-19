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

    # Ensure required columns exist (case-insensitive already handled)
    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

    # -----------------------
    # Missing category handling
    # -----------------------
    # This CSV may not preserve distinct DK vs No-answer codes; often both appear as NaN.
    # We will:
    #   1) Count DK and No-answer only if the raw values carry explicit codes/labels.
    #   2) If they are not distinguishable, we still produce the table but set DK/NA to NaN
    #      (not zero) and report combined missing as an integrity check in the output file.
    #
    # This avoids the prior error (forcing an arbitrary split or collapsing into DK).
    def _as_str_upper(s):
        # string dtype keeps <NA>; fillna("") ensures vectorized string ops are safe
        return s.astype("string").fillna("").str.strip().str.upper()

    def _detect_missing_masks(raw):
        """
        Return (dk_mask, na_mask, has_distinct) based on explicit encodings in the raw series.
        Recognizes common patterns:
          - '[NA(d)]' / 'NA(d)' / 'DK' / "DON'T KNOW"
          - '[NA(n)]' / 'NA(n)' / 'NO ANSWER'
        """
        su = _as_str_upper(raw)

        # Explicit NA(d) / NA(n)
        dk_mask = su.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = su.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Some exports may use verbal labels
        dk_mask = dk_mask | su.str.contains(r"\bDON'?T\s+KNOW\b|\bDONT\s+KNOW\b|\bDK\b", regex=True, na=False)
        na_mask = na_mask | su.str.contains(r"\bNO\s+ANSWER\b|\bNA\b(?!\()", regex=True, na=False)

        has_distinct = bool(dk_mask.any() or na_mask.any())
        return dk_mask, na_mask, has_distinct

    def _tabulate_one(raw):
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_mask, na_mask, has_distinct = _detect_missing_masks(raw)

        # If distinct codes exist, count them and treat any other non-1..5 as additional NA (unclassified)
        # but DO NOT merge it into DK (this was the earlier systematic error).
        if has_distinct:
            # "Other missing" means: not valid and not explicitly classified
            other_missing = valid.isna() & ~(dk_mask | na_mask)
            # Treat other_missing as No answer (generic missing bucket in this export)
            na_mask = na_mask | other_missing
            dk_n = int(dk_mask.sum())
            na_n = int(na_mask.sum())
        else:
            dk_n = np.nan
            na_n = np.nan

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        combined_missing = int(valid.isna().sum())

        return counts_1_5, dk_n, na_n, mean_val, combined_missing

    # -----------------------
    # Build table (counts + mean)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")
    combined_missing_report = {}

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val, combined_missing = _tabulate_one(df[var])
        combined_missing_report[genre_label] = combined_missing

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable output
    # -----------------------
    formatted = table.copy()

    def _fmt_cell(row_name, v):
        if pd.isna(v):
            return ""
        if row_name == "Mean":
            return f"{float(v):.2f}"
        return str(int(round(float(v))))

    for r in formatted.index:
        formatted.loc[r] = formatted.loc[r].map(lambda v, rr=r: _fmt_cell(rr, v))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (paper layout)
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories (1–5). Mean computed on 1–5 excluding missing.\n")
        f.write("Note: This CSV export may not preserve distinct '(M) Don't know much about it' vs '(M) No answer'.\n")
        f.write("If those categories are not explicitly coded in the file, their cells are left blank.\n\n")

        # Diagnostic: combined missing per item (non-1..5, including NaN)
        f.write("Diagnostic: Combined missing/non-1..5 count per genre (cannot be split unless coded distinctly):\n")
        diag = pd.Series(combined_missing_report).to_frame(name="Combined missing/non-1..5 (count)")
        f.write(diag.to_string())
        f.write("\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table