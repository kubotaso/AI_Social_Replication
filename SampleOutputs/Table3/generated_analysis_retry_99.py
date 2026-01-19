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
    # Helpers: classify DK vs NA when available; otherwise fall back safely
    # -----------------------
    def _as_clean_str(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _detect_missing_codes(raw: pd.Series):
        """
        Returns (dk_mask, na_mask, classified_any)
        Detects common encodings:
          - GSS-style: [NA(d)], NA(d), [NA(n)], NA(n)
          - textual: "DON'T KNOW", "DONT KNOW", "DK", "NO ANSWER", "NA", "N/A", "REFUSED"
          - numeric: 8/9, 98/99, 998/999 (handled separately)
        """
        s = _as_clean_str(raw)

        dk_mask = s.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Textual fallbacks (rare in numeric extracts, but harmless)
        dk_text = s.str.contains(r"\bDONT\s+KNOW\b|\bDON'T\s+KNOW\b|\bDK\b", regex=True, na=False)
        na_text = s.str.contains(r"\bNO\s+ANSWER\b|\bNO\s+ANS\b|\bN/?A\b|\bREFUSED\b", regex=True, na=False)

        dk_mask = dk_mask | dk_text
        na_mask = na_mask | na_text

        classified_any = bool(dk_mask.any() or na_mask.any())
        return dk_mask, na_mask, classified_any

    def _tabulate_one(raw: pd.Series, varname: str):
        """
        Computes counts for 1..5, DK count, NA count, and mean on 1..5.
        If DK/NA are not distinguishable in the CSV (all missing collapsed to NaN),
        we *still produce the two rows*:
          - assign all missing to DK
          - assign NA as 0
        This avoids runtime errors while keeping structure correct.
        """
        # Numeric coercion
        x = pd.to_numeric(raw, errors="coerce")

        # Common numeric DK/NA conventions (kept separate if present)
        dk_num = x.isin([8, 98, 998])
        na_num = x.isin([9, 99, 999])

        # String-based explicit missing codes
        dk_str, na_str, any_str_classified = _detect_missing_codes(raw)

        dk_mask = dk_num | dk_str
        na_mask = na_num | na_str

        # Valid substantive
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # "Other" non-1..5 values (including NaN). If DK/NA not distinguishable, keep structure:
        # - If we detected any DK/NA coding, assign remaining missing/non-1..5 to DK (conservative).
        # - If we detected no DK/NA coding at all, assign all remaining missing/non-1..5 to DK and NA=0.
        classified_any = bool(any_str_classified or dk_num.any() or na_num.any())
        other_missing = valid.isna() & ~(dk_mask | na_mask)

        if classified_any:
            dk_mask = dk_mask | other_missing
        else:
            # CSV collapsed missing categories (common): keep separate rows, but cannot split
            dk_mask = dk_mask | other_missing
            # na_mask stays as-is (likely all False)

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
    # Build table (numeric)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Format for human-readable output
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

    # -----------------------
    # Save (three 6-genre blocks like the paper)
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write(
            "Note: If DK vs No answer codes are not preserved in the CSV export, all missing are reported as DK and No answer as 0.\n\n"
        )
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table