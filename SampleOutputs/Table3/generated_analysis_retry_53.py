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

    # Filter: YEAR == 1993 (exclude missing YEAR automatically)
    df = df.loc[pd.to_numeric(df["YEAR"], errors="coerce").eq(1993)].copy()

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
            raise ValueError(f"Required variable not found: {var}")

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
    # Helpers: parse values + missing categories
    # -----------------------
    def _norm_text_series(s):
        # robust string normalization (keeps <NA> as <NA>)
        return s.astype("string").str.strip().str.upper()

    def _detect_explicit_missing(s):
        """
        Returns (dk_mask, na_mask) using explicit GSS-style string labels if present.
        We only count as DK/NA when the raw value clearly indicates NA(d) or NA(n).
        """
        su = _norm_text_series(s)

        dk = (
            su.str.contains(r"\[NA\(D\)\]", regex=True, na=False)
            | su.str.fullmatch(r"NA\(D\)", na=False)
            | su.str.contains(r"\bDON[’']?T\s+KNOW\b", regex=True, na=False)
            | su.str.contains(r"\bDONT\s+KNOW\b", regex=True, na=False)
            | su.str.contains(r"\bDK\b", regex=True, na=False)
            | su.str.contains(r"\bDON[’']?T\s+KNOW\s+MUCH\b", regex=True, na=False)
        )

        na = (
            su.str.contains(r"\[NA\(N\)\]", regex=True, na=False)
            | su.str.fullmatch(r"NA\(N\)", na=False)
            | su.str.contains(r"\bNO\s+ANSWER\b", regex=True, na=False)
            | su.str.contains(r"\bNA\b", regex=True, na=False)
        )

        # If a token triggers both (e.g., "DK/NA"), don't assign here; it will remain "other missing"
        both = dk & na
        if bool(both.any()):
            dk = dk & ~both
            na = na & ~both

        return dk, na

    def _tabulate_one(raw):
        """
        Computes:
          - counts for 1..5
          - DK count: explicit [NA(d)] / dont know labels only
          - NA count: explicit [NA(n)] / no answer labels only
          - Mean on valid 1..5 only
        Any other non-1..5 values (including blanks/NaN) are treated as missing-but-unspecified:
          - excluded from mean
          - not forced into DK or NA (so we never fabricate split)
        """
        # numeric values (valid attitudes are 1..5)
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        dk_mask, na_mask = _detect_explicit_missing(raw)

        # counts 1..5
        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())

        mean_val = float(valid.mean(skipna=True)) if bool(valid.notna().any()) else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table (numeric)
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
    # Format for text output
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[r] = formatted.loc[r].map(
                lambda v: "" if pd.isna(v) else str(int(round(float(v))))
            )

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (paper-like layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: '(M)' counts require explicit DK/NA codes in the extract; unspecified missings are not split.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table