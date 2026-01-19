def run_analysis(data_source):
    import os
    import re
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
    # Helpers for DK vs No answer
    # -----------------------
    # We must NOT guess/split missing. We only report DK/NA separately if the raw file
    # actually preserves distinguishable codes/strings. Otherwise, raise a clear error.
    def _series_tokens(s):
        # returns uppercase string series with NA filled as empty string
        return s.astype("string").fillna("").str.strip().str.upper()

    def _mask_from_patterns(s_up, patterns):
        mask = pd.Series(False, index=s_up.index)
        for pat in patterns:
            mask = mask | s_up.str.contains(pat, regex=True, na=False)
        return mask

    # Common encodings for missing in GSS extracts
    DK_PATS = [
        r"\[NA\(D\)\]",
        r"\bNA\(D\)\b",
        r"\bDON'?T\s+KNOW\b",
        r"\bDONT\s+KNOW\b",
        r"\bDK\b",
        r"\bDON'?T\s+KNOW\s+MUCH\b",
    ]
    NA_PATS = [
        r"\[NA\(N\)\]",
        r"\bNA\(N\)\b",
        r"\bNO\s+ANSWER\b",
        r"\bN/?A\b",
    ]

    def _tabulate_one(raw):
        # raw may be numeric or strings. We detect explicit DK/NA tokens in the original.
        s_up = _series_tokens(raw)
        dk_mask = _mask_from_patterns(s_up, DK_PATS)
        na_mask = _mask_from_patterns(s_up, NA_PATS)

        # numeric values
        x = pd.to_numeric(raw, errors="coerce")
        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        # what is "missing/non-1..5" in the file (including true NaNs and other codes)
        nonvalid_mask = ~valid_mask

        # If we have any nonvalid values, we must be able to classify each as DK or NA,
        # otherwise we cannot produce Table 3's two separate missing rows without guessing.
        nonvalid_unclassified = nonvalid_mask & ~(dk_mask | na_mask) & raw.notna()
        # also treat actual NaN as unclassified unless it's already tagged via tokens (it won't be)
        nan_unclassified = raw.isna() & nonvalid_mask & ~(dk_mask | na_mask)

        # If there are ANY unclassified nonvalid values, the export doesn't preserve DK vs NA
        # (or uses numeric codes we don't know). Fail fast with diagnostics.
        unclassified_n = int((nonvalid_unclassified | nan_unclassified).sum())
        if unclassified_n > 0:
            examples = []
            # show a few example raw values (including NaN)
            vals = raw.loc[(nonvalid_unclassified | nan_unclassified)].head(10)
            for v in vals.tolist():
                examples.append(repr(v))
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                "this CSV export does not preserve distinguishable missing categories for at least one item. "
                f"Unclassified non-1..5/missing count={unclassified_n}. "
                f"Example unclassified raw values: {', '.join(examples)}. "
                "Re-export data preserving '[NA(d)]' and '[NA(n)]' (or equivalent distinct codes)."
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
    # Write human-readable text file (3 blocks of 6 like the paper layout)
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

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing (M) categories.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table