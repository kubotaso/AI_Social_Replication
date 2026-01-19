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
    # Helpers to preserve DK vs No answer
    # -----------------------
    def _as_clean_str(s):
        # robust string view (keeps <NA> as <NA>)
        return s.astype("string").str.strip()

    def _explicit_na_masks(raw):
        """
        Detect explicit missing categories if the export preserved them as strings.
        Supports patterns like:
          [NA(d)], NA(d), NA(D), dk, don't know, don't know much about it
          [NA(n)], NA(n), NA(N), no answer
        """
        s = _as_clean_str(raw).str.upper()

        # DK patterns
        dk = (
            s.str.contains(r"\[?NA\(D\)\]?", regex=True, na=False)
            | s.str.contains(r"\bDK\b", regex=True, na=False)
            | s.str.contains("DON'T KNOW", na=False)
            | s.str.contains("DONT KNOW", na=False)
        )

        # No-answer patterns
        na = (
            s.str.contains(r"\[?NA\(N\)\]?", regex=True, na=False)
            | s.str.contains("NO ANSWER", na=False)
        )

        return dk, na

    def _tabulate_one(raw, varname):
        """
        Returns:
          counts_1_5: Series indexed [1..5]
          dk_n: int
          na_n: int
          mean_val: float (mean of 1..5 only)
        """
        # Parse numeric values for substantive categories
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Identify explicit DK/NA if present
        dk_mask, na_mask = _explicit_na_masks(raw)

        # "Missing pool" = everything not in 1..5 (including true NaN, blanks, strings, etc.)
        # Use numeric parse to find which rows are not substantive.
        non_substantive = ~x.isin([1, 2, 3, 4, 5])
        non_substantive = non_substantive.fillna(True)  # NaN numeric parse -> non-substantive

        # If any explicit DK/NA exist, we can classify:
        if int(dk_mask.sum()) + int(na_mask.sum()) > 0:
            dk_n = int((non_substantive & dk_mask).sum())
            na_n = int((non_substantive & na_mask).sum())

            # Any remaining non-substantive that isn't explicitly DK/NA:
            # Treat as "No answer" to avoid inflating DK; this matches typical GSS handling
            # where residual missings (refused, skipped, etc.) are closer to "no answer".
            other_n = int((non_substantive & ~(dk_mask | na_mask)).sum())
            na_n += other_n

        else:
            # Export collapsed missing types (blanks/NaN only): cannot split DK vs NA from raw data.
            # We must still produce the Table 3 structure. For this instrument, the only
            # defensible inference from the CSV alone is: all non-substantive are "Don't know much",
            # and "No answer" is 0, OR vice-versa. Either choice is arbitrary.
            #
            # To avoid arbitrary guessing that would *look* precise, we:
            # - set DK = total non-substantive
            # - set NA = 0
            # and clearly note this in the saved output file header.
            dk_n = int(non_substantive.sum())
            na_n = 0

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    any_explicit_missing = False
    for genre_label, var in genre_map:
        raw = df[var]
        dk_mask, na_mask = _explicit_na_masks(raw)
        if int(dk_mask.sum()) + int(na_mask.sum()) > 0:
            any_explicit_missing = True
            break

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = np.round(mean_val, 2) if np.isfinite(mean_val) else np.nan

    # -----------------------
    # Human-readable text output (3 blocks of 6 columns)
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        if not any_explicit_missing:
            f.write(
                "\nNOTE: This CSV export does not preserve distinguishable missing categories for DK vs No answer.\n"
                "All non-1..5 responses are reported under \"(M) Don't know much about it\" and \"(M) No answer\" is set to 0.\n"
                "To reproduce the published DK/No answer split, re-export data preserving missing codes (e.g., [NA(d)] and [NA(n)]).\n"
            )
        f.write("\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table