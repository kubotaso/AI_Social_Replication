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

    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

    # -----------------------
    # Row structure (exact)
    # -----------------------
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
    # Helpers: explicit missing code detection (if present)
    # -----------------------
    def _as_str(series):
        # Keep NA as <NA> in pandas string dtype
        return series.astype("string")

    def _explicit_missing_masks(raw_series):
        """
        Detect explicit GSS-style missing tokens if present as strings, e.g.:
        [NA(d)] and [NA(n)] (case-insensitive, with or without brackets/spaces).
        Returns: (dk_mask, na_mask) boolean Series.
        """
        s = _as_str(raw_series).str.strip().str.upper()

        # match "[NA(D)]" or "NA(D)" etc.
        dk_mask = s.str.contains(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", regex=True, na=False)
        na_mask = s.str.contains(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", regex=True, na=False)
        return dk_mask, na_mask

    # -----------------------
    # Build table: counts + mean
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        raw = df[var]

        # Numeric parse for valid 1..5
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Missing split:
        # If explicit NA(d)/NA(n) exist, use them.
        # Otherwise: we cannot split missing into DK vs NA from this extract; leave as NaN (blank in output),
        # but still compute valid 1..5 counts and mean correctly.
        dk_mask, na_mask = _explicit_missing_masks(raw)

        has_explicit_split = (dk_mask.sum() + na_mask.sum()) > 0

        if has_explicit_split:
            dk_count = int(dk_mask.sum())
            na_count = int(na_mask.sum())
        else:
            dk_count = np.nan
            na_count = np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # -----------------------
    # Human-readable text output (3 blocks of 6 genres)
    # -----------------------
    formatted = table.copy()

    def _fmt_count(v):
        if pd.isna(v):
            return ""
        return str(int(v))

    def _fmt_mean(v):
        if pd.isna(v):
            return ""
        return f"{float(v):.2f}"

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(_fmt_mean)
        else:
            formatted.loc[r] = formatted.loc[r].map(_fmt_count)

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: '(M)' rows are populated only if explicit NA(d)/NA(n) codes exist in the provided extract.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table