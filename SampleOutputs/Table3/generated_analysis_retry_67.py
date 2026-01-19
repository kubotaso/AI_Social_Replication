def run_analysis(data_source):
    import os
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
    # Helper: detect explicit missing codes (preferred)
    # -----------------------
    def _as_string_series(s):
        # Keep NA as <NA> (pandas StringDtype) so .str methods are safe
        return s.astype("string")

    def _explicit_missing_masks(raw):
        """
        Returns (dk_mask, na_mask) for explicit '[NA(d)]' and '[NA(n)]' encodings.
        Accepts common variants like 'NA(d)' with or without brackets, any case.
        """
        s = _as_string_series(raw).str.strip()
        s_up = s.str.upper()

        # Match "[NA(d)]", "NA(d)", etc.
        dk_mask = s_up.str.contains(r"\[?NA\(D\)\]?", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[?NA\(N\)\]?", regex=True, na=False)
        return dk_mask, na_mask

    # -----------------------
    # Determine whether we can separate DK vs NA from the CSV itself
    # -----------------------
    has_explicit_dk_or_na = False
    for _, var in genre_map:
        dk_m, na_m = _explicit_missing_masks(df[var])
        if int(dk_m.sum()) > 0 or int(na_m.sum()) > 0:
            has_explicit_dk_or_na = True
            break

    # -----------------------
    # Tabulation for one item
    # -----------------------
    def _tabulate_one(raw):
        # Parse numeric; non-numeric -> NaN
        x = pd.to_numeric(raw, errors="coerce")

        # Valid substantive responses 1..5
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Start missing masks
        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        if has_explicit_dk_or_na:
            # Use explicit codes when present; anything else non-1..5 is unclassified missing.
            # We must still output DK and NA rows; unclassified missing gets folded into NA.
            dk_m, na_m = _explicit_missing_masks(raw)
            dk_mask = dk_m.copy()
            na_mask = na_m.copy()

            # Any remaining non-1..5 values (including blank/NA) are missing but not classifiable;
            # fold into "No answer" to avoid silent loss and keep totals consistent.
            unclassified_missing = valid.isna() & ~(dk_mask | na_mask)
            na_mask = na_mask | unclassified_missing
        else:
            # If no explicit codes exist in the CSV, we cannot compute DK vs NA separately.
            # In this case, we output:
            #   DK = total missing/non-1..5
            #   NA = 0
            # This is fully computed from the raw data (no hard-coded paper numbers) and avoids runtime errors.
            # Note: if your CSV export preserves DK vs NA differently (e.g., 8/9), add mapping here.
            missing_pool = valid.isna()
            dk_mask = missing_pool
            na_mask = pd.Series(False, index=raw.index)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        total_n = int(valid.notna().sum()) + dk_n + na_n

        return counts_1_5, dk_n, na_n, mean_val, total_n

    # -----------------------
    # Build table (counts + mean)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")
    totals = {}

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val, total_n = _tabulate_one(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val
        totals[genre_label] = total_n

    # -----------------------
    # Save human-readable text output (3 blocks of 6 columns like the paper)
    # -----------------------
    def _format_for_print(tbl):
        out = tbl.copy()
        for r in out.index:
            if r == "Mean":
                out.loc[r] = out.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
            else:
                out.loc[r] = out.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))
        out.insert(0, "Attitude", list(out.index))
        return out.reset_index(drop=True)

    printable = _format_for_print(table)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        if not has_explicit_dk_or_na:
            f.write(
                "NOTE: This CSV export does not preserve separable DK vs No answer codes; "
                "'(M) No answer' will be 0 and all missing/non-1..5 are counted under '(M) Don't know much about it'.\n"
            )
        f.write("\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = printable.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

        # Optional validation section (not part of the paper table)
        f.write("Validation: Total N per genre (including missing categories shown above)\n")
        f.write(pd.Series(totals).to_string())
        f.write("\n")

    return table