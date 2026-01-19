def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names (GSS extracts often come lowercase)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in dataset.")

    # Filter to 1993 only
    df = df.loc[df["YEAR"] == 1993].copy()

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
            raise ValueError(f"Required variable not found: {v}")

    # Exact row order/labels as in the target table
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
    # Helpers
    # -----------------------
    def _as_clean_string(s: pd.Series) -> pd.Series:
        # preserve missing as <NA>
        return s.astype("string").str.strip()

    def _count_missing_buckets(raw: pd.Series):
        """
        Returns:
            counts_1_5: pd.Series indexed [1..5] counts
            dk_n: int count for NA(d) (don't know much about it)
            na_n: int count for NA(n) (no answer)
            mean_val: float mean on valid 1..5
        This function NEVER fabricates DK/NA split. If the dataset does not preserve
        distinct NA(d)/NA(n), it will:
          - set dk_n and na_n to NaN (not computable), but still compute 1..5 and mean.
        """
        # Numeric parse for substantive codes
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # Try to detect explicit GSS-style missing strings like "[NA(d)]" and "[NA(n)]"
        s = _as_clean_string(raw).str.upper()
        # Some exports may store as "NA(d)" without brackets; handle both
        dk_mask = s.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        if int(dk_mask.sum()) == 0 and int(na_mask.sum()) == 0:
            # No explicit codes preserved; cannot split missing pool into DK vs NA reliably.
            dk_n = np.nan
            na_n = np.nan
        else:
            dk_n = int(dk_mask.sum())
            na_n = int(na_mask.sum())

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    col_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=col_labels, dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _count_missing_buckets(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable output (with Attitude stub column + 3 blocks of 6 genres)
    # -----------------------
    display = table.copy()

    # Format counts as integers where possible; missing DK/NA as blank if not computable
    for r in display.index:
        if r == "Mean":
            display.loc[r] = display.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            def fmt_count(v):
                if pd.isna(v):
                    return ""
                return str(int(round(float(v))))
            display.loc[r] = display.loc[r].map(fmt_count)

    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genres = [g for g, _ in genre_map]
    blocks = [genres[i:i + 6] for i in range(0, len(genres), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: If the CSV export does not preserve distinct missing codes '[NA(d)]' and '[NA(n)]',\n")
        f.write("the two (M) rows cannot be separated and will be left blank.\n\n")

        for i, cols in enumerate(blocks, start=1):
            f.write(f"Block {i}:\n")
            f.write(display.loc[:, ["Attitude"] + cols].to_string(index=False))
            f.write("\n\n")

    return table