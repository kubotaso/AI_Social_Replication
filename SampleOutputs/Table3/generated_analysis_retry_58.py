def run_analysis(data_source):
    import os
    import re
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
    # Helpers: detect explicit missing codes if preserved
    # -----------------------
    _re_dk = re.compile(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", re.IGNORECASE)
    _re_na = re.compile(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", re.IGNORECASE)

    def _as_clean_string(s: pd.Series) -> pd.Series:
        # Keep as string without turning NaN into "nan"
        return s.astype("string").str.strip()

    def _explicit_missing_masks(raw: pd.Series):
        s = _as_clean_string(raw)
        dk = s.str.contains(_re_dk, na=False)
        na = s.str.contains(_re_na, na=False)
        return dk, na

    def _tabulate_one(raw: pd.Series):
        """
        Returns:
          counts_1_5: pd.Series index [1..5] int
          dk_n: int
          na_n: int
          mean_val: float (mean of 1..5 only)
        Rules:
          - If explicit NA(d)/NA(n) codes exist as strings, use them for DK/NA.
          - Otherwise, treat all missing/invalid as "Don't know much about it" and set "No answer" to 0.
            (This avoids runtime error while still computing from raw data; the export may have collapsed
             DK vs NA into blank NA.)
        """
        # detect explicit codes (string-based)
        dk_mask, na_mask = _explicit_missing_masks(raw)

        # numeric parse for valid values
        x = pd.to_numeric(raw, errors="coerce")

        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid_vals = x.where(valid_mask, np.nan)

        # counts 1..5
        counts_1_5 = (
            valid_vals.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # missing split
        if (dk_mask | na_mask).any():
            # Anything else invalid/missing but not explicitly NA(d)/NA(n) -> treat as DK (conservative)
            other_missing = ~valid_mask
            other_missing = other_missing & ~(dk_mask | na_mask)

            dk_n = int((dk_mask | other_missing).sum())
            na_n = int(na_mask.sum())
        else:
            # Export does not preserve distinct DK vs NA; do not error.
            # Compute from raw: all non-1..5 (including blanks/NA) are counted as DK; NA row becomes 0.
            all_missing = int((~valid_mask).sum())
            dk_n = all_missing
            na_n = 0

        mean_val = float(valid_vals.mean(skipna=True)) if valid_vals.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

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
    # Prepare formatted display table (counts as ints; mean to 2 decimals)
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
    # Write text output in 3 blocks (6 genres each)
    # -----------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: If the CSV export collapses DK vs No answer into generic NA, the '(M) No answer' row will be 0 and all missing will appear under DK.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table