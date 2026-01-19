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
    # Helpers: robust missing parsing
    # -----------------------
    def _as_clean_string(s: pd.Series) -> pd.Series:
        # Keep as string but preserve NA as <NA> for easier masking
        return s.astype("string").str.strip()

    def _detect_special_missing_masks(raw: pd.Series):
        """
        Detect distinguishable DK vs No-answer categories if present as:
        - explicit tokens like '[NA(d)]', 'NA(d)', 'DK', 'DON'T KNOW', etc.
        - numeric codes commonly used for missingness (8/9), if present
        """
        s_str = _as_clean_string(raw)
        s_up = s_str.str.upper()

        # Explicit NA() style markers
        dk_na_pattern = r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?"
        na_na_pattern = r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?"

        dk_mask = s_up.str.contains(dk_na_pattern, regex=True, na=False)
        na_mask = s_up.str.contains(na_na_pattern, regex=True, na=False)

        # Common text labels (defensive; won't trigger on pure numeric columns)
        dk_tokens = [
            "DON'T KNOW",
            "DONT KNOW",
            "DON’T KNOW",
            "DK",
            "DON'T KNOW MUCH",
            "DONT KNOW MUCH",
            "DON’T KNOW MUCH",
        ]
        na_tokens = ["NO ANSWER", "NA", "N/A", "REFUSED", "SKIPPED"]

        if not dk_mask.any():
            for t in dk_tokens:
                dk_mask = dk_mask | s_up.str.fullmatch(re.escape(t), na=False) | s_up.str.contains(re.escape(t), na=False)
        if not na_mask.any():
            for t in na_tokens:
                na_mask = na_mask | s_up.str.fullmatch(re.escape(t), na=False)

        # Numeric codes (only apply where parsing yields those exact integers)
        x = pd.to_numeric(raw, errors="coerce")
        dk_mask = dk_mask | x.eq(8)
        na_mask = na_mask | x.eq(9)

        # Ensure no overlap
        overlap = dk_mask & na_mask
        if overlap.any():
            # Prefer explicit NA(n) for overlaps; remove from DK
            dk_mask = dk_mask & ~overlap

        return dk_mask, na_mask

    def _tabulate_one(raw: pd.Series):
        # Parse numeric, keep 1..5 as valid; everything else is missing of some form
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        dk_mask, na_mask = _detect_special_missing_masks(raw)

        # Anything non-valid and not explicitly DK/NA is "unclassified missing"
        unclassified = valid.isna() & ~(dk_mask | na_mask)

        # If we cannot distinguish DK vs NA from the file, we still produce the table
        # but we cannot truthfully split unclassified missing into the two rows.
        # Therefore: put unclassified into "Don't know much..." ONLY IF that category
        # is explicitly identifiable; otherwise set both to NaN and report combined missing in file note.
        can_split = bool(dk_mask.any() or na_mask.any() or (pd.to_numeric(raw, errors="coerce").isin([8, 9]).any()))

        if can_split:
            # Conservative: unclassified missing goes to DK (closest interpretation) so both rows exist
            dk_mask = dk_mask | unclassified
            dk_n = int(dk_mask.sum())
            na_n = int(na_mask.sum())
            split_note = None
        else:
            dk_n = np.nan
            na_n = np.nan
            split_note = int(unclassified.sum())

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val, split_note

    # -----------------------
    # Build table (numeric)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    split_notes = {}
    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val, split_note = _tabulate_one(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

        if split_note is not None and split_note > 0:
            split_notes[genre_label] = split_note

    # -----------------------
    # Format for human-readable output
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            def _fmt_count(v):
                if pd.isna(v):
                    return ""
                return str(int(round(float(v))))
            formatted.loc[r] = formatted.loc[r].map(_fmt_count)

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (paper layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        if split_notes:
            f.write(
                "\nNOTE: This CSV does not preserve distinct DK vs No-answer codes for some items; "
                "for those items the '(M)' rows are left blank. Unclassified missing counts by item:\n"
            )
            for k in genre_labels:
                if k in split_notes:
                    f.write(f"  - {k}: {split_notes[k]}\n")
            f.write("\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table