def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names for robustness (file uses lower-case in the sample)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to YEAR == 1993 (comparison excludes NA automatically)
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ---- Variables (Table 3) ----
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

    # ---- Helpers ----
    def _norm_token(x):
        """Normalize one cell value to a comparable token (e.g., '[NA(d)]' -> 'NA(D)')."""
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s == "":
            return None
        su = s.upper()
        su = su.replace(" ", "")
        # Common encodings to normalize
        su = su.replace("[", "").replace("]", "")
        return su

    def _split_missing(raw_series):
        """
        Identify DK vs NA using explicit NA(d)/NA(n) tokens if present.
        Otherwise, raise (cannot be inferred from plain NaN without risking wrong results).
        """
        tokens = raw_series.map(_norm_token)

        # Detect explicit missing tokens
        dk_mask = tokens.isin({"NA(D)", "NA(D).", "NA(DK)", "NA(DONTKNOW)", "NA(DON'TKNOW)"})
        na_mask = tokens.isin({"NA(N)", "NA(N).", "NA(NA)", "NA(NOANSWER)"})

        # Also allow patterns like 'NA(D)' embedded (rare)
        dk_mask = dk_mask | tokens.fillna("").str.contains(r"NA\(D\)", regex=True)
        na_mask = na_mask | tokens.fillna("").str.contains(r"NA\(N\)", regex=True)

        # If dataset uses explicit DK/NA coding, we can proceed. If not, we refuse to guess.
        if (dk_mask.sum() + na_mask.sum()) == 0:
            # If everything is numeric 1..5 with no missing, that's fine (DK/NA both 0).
            x = pd.to_numeric(raw_series, errors="coerce")
            any_missing = x.isna().any() or (x.notna() & ~x.isin([1, 2, 3, 4, 5])).any()
            if any_missing:
                raise ValueError(
                    "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                    "dataset does not contain explicit NA(d)/NA(n) codes for this variable."
                )
            return dk_mask, na_mask

        return dk_mask, na_mask

    # ---- Build numeric table (counts + mean) ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        # Parse numeric valid responses
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Missing split for table rows (requires explicit NA(d)/NA(n) if missing exists)
        dk_mask, na_mask = _split_missing(raw)

        # Any other non-1..5 numeric codes or NaNs are "missing", but the table only displays DK and NA.
        # If present beyond DK/NA, they won't be shown; but we ensure we don't silently misclassify them.
        other_missing = (x.isna() | (x.notna() & ~x.isin([1, 2, 3, 4, 5]))) & ~(dk_mask | na_mask)
        if other_missing.any():
            raise ValueError(
                f"Found missing/invalid codes in {var} that are not NA(d) or NA(n); cannot map to Table 3 rows."
            )

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # ---- Create display table: counts as integers; Mean rounded to 2 decimals ----
    display = table.copy()

    for idx in display.index:
        if idx == "Mean":
            display.loc[idx] = display.loc[idx].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            display.loc[idx] = display.loc[idx].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display.insert(0, "Attitude", display.index)
    display = display.reset_index(drop=True)

    # ---- Save as three 6-column blocks (layout similar to printed table) ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing (NA codes).\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table