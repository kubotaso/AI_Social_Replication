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

    # Filter to 1993 only (exclude missing YEAR automatically by comparison)
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
    # Missing category parsing
    # -----------------------
    def _as_clean_str(s: pd.Series) -> pd.Series:
        # Keep NA as <NA> (pandas string dtype), normalize whitespace/case
        return s.astype("string").str.strip().str.upper()

    def _extract_dk_na_masks(raw: pd.Series):
        """
        Return boolean masks for DK and NA if distinguishable in raw export, else None.
        Supports common encodings:
          - '[NA(d)]', 'NA(d)', 'DON'T KNOW', 'DONT KNOW', 'DK'
          - '[NA(n)]', 'NA(n)', 'NO ANSWER', 'NA'
        """
        s_str = _as_clean_str(raw)

        # Explicit GSS-style tags
        dk_tag = s_str.str.contains(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", regex=True, na=False)
        na_tag = s_str.str.contains(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", regex=True, na=False)

        # Text labels (if any)
        dk_txt = s_str.str.contains(r"\bDONT\s+KNOW\b|\bDON'T\s+KNOW\b|\bDK\b", regex=True, na=False)
        na_txt = s_str.str.contains(r"\bNO\s+ANSWER\b|\bNOANS\b", regex=True, na=False)

        dk_mask = dk_tag | dk_txt
        na_mask = na_tag | na_txt

        if int(dk_mask.sum()) == 0 and int(na_mask.sum()) == 0:
            return None, None

        # If overlap due to messy labels, prioritize explicit tags first, then NA text, then DK text
        overlap = dk_mask & na_mask
        if overlap.any():
            # Keep NA where NA tag/text present
            na_only = na_mask.copy()
            dk_only = dk_mask & ~na_only
            return dk_only, na_only

        return dk_mask, na_mask

    def _tabulate_one(raw: pd.Series, varname: str):
        # Parse numeric values; valid attitudes are 1..5
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Determine DK/NA counts
        dk_mask, na_mask = _extract_dk_na_masks(raw)

        # All non-valid (including NaN) are missing pool for Table 3 M-categories
        missing_pool = valid.isna()

        if dk_mask is None and na_mask is None:
            # This export does not preserve distinct DK vs No answer.
            # We can still compute (1)-(5) and mean, but cannot split M.
            raise ValueError(
                f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
                "this CSV export does not preserve distinguishable DK/NA categories (e.g., '[NA(d)]'/'[NA(n)]', "
                "DK/No-answer labels, or distinct numeric codes). Re-export data preserving these codes."
            )

        # Classify remaining missing beyond explicitly tagged DK/NA:
        # Any missing not explicitly NA is treated as DK (consistent with instrument where DK dominates),
        # but this only applies when tags exist for at least some cases.
        dk_mask = dk_mask.fillna(False)
        na_mask = na_mask.fillna(False)

        other_missing = missing_pool & ~(dk_mask | na_mask)
        dk_mask = dk_mask | other_missing

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # Sanity: DK+NA should equal missing pool
        # If not, something strange in parsing; still proceed but keep consistent totals.
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build Table 3
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
    # Save human-readable text (3 blocks of 6)
    # -----------------------
    formatted = table.copy()

    # Counts as integers; Mean rounded to 2 decimals
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