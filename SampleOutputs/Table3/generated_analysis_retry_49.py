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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required variable missing from dataset: {v}")

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
    # Helpers: preserve NA(d)/NA(n) if present
    # -----------------------
    def _as_str_series(s):
        # String dtype keeps <NA> values; normalize whitespace/case for pattern matches
        return s.astype("string").str.strip()

    def _explicit_missing_masks(raw):
        s = _as_str_series(raw).str.upper()

        # Accept both "[NA(d)]" and "NA(d)" variants; same for NA(n).
        dk_mask = s.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)
        return dk_mask, na_mask

    def _tabulate_one(raw, varname):
        # Parse numeric values for substantive categories
        xnum = pd.to_numeric(raw, errors="coerce")
        valid = xnum.where(xnum.isin([1, 2, 3, 4, 5]), np.nan)

        # If the file preserved explicit NA(d)/NA(n), use them
        dk_mask, na_mask = _explicit_missing_masks(raw)
        if int(dk_mask.sum()) + int(na_mask.sum()) > 0:
            # Any remaining missing that isn't explicitly NA(d)/NA(n) is left unclassified;
            # Table 3 only wants those two M-rows, so we don't invent splits.
            # (If such codes exist, the input extract isn't consistent with the spec.)
            other_missing = valid.isna() & ~(dk_mask | na_mask)
            if int(other_missing.sum()) > 0:
                examples = (
                    _as_str_series(raw.loc[other_missing])
                    .dropna()
                    .astype(str)
                    .head(10)
                    .tolist()
                )
                raise ValueError(
                    f"Found missing/non-1..5 values for {varname} that are not NA(d) or NA(n). "
                    f"Cannot map to Table 3's two M categories. Examples: {examples}"
                )
            dk_n = int(dk_mask.sum())
            na_n = int(na_mask.sum())
        else:
            # If explicit codes are not preserved, we cannot separate DK vs NA from the raw data
            # without copying published numbers (not allowed). Fail clearly.
            non_1_5 = xnum.notna() & ~xnum.isin([1, 2, 3, 4, 5])
            if int(non_1_5.sum()) > 0:
                ex = xnum.loc[non_1_5].astype(float).head(10).tolist()
                raise ValueError(
                    f"Dataset does not preserve explicit NA(d)/NA(n) codes for {varname}. "
                    f"Found non-1..5 numeric codes {sorted(set(ex))[:10]}. "
                    f"Re-export with distinct '[NA(d)]' and '[NA(n)]' values."
                )
            if int(valid.isna().sum()) > 0:
                raise ValueError(
                    f"Dataset does not preserve explicit NA(d)/NA(n) codes for {varname}. "
                    f"Found {int(valid.isna().sum())} missing values that cannot be split into "
                    f"'Don't know much about it' vs 'No answer'. Re-export with '[NA(d)]' and '[NA(n)]'."
                )
            dk_n, na_n = 0, 0

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean()) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table (counts + mean)
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

        # Internal consistency check: total N per genre (including M categories) should equal number of records
        total = (
            int(counts_1_5.sum())
            + int(dk_n)
            + int(na_n)
        )
        if total != len(df):
            # This can happen if the dataset contains other missing codes beyond NA(d)/NA(n)
            # or if some values are truly blank (NA) without labels.
            raise ValueError(
                f"Totals do not reconcile for {var} ({genre_label}). "
                f"Sum(1-5)+DK+NA = {total}, but N rows = {len(df)}. "
                f"Ensure all missing are encoded explicitly as [NA(d)] or [NA(n)] in the extract."
            )

    # -----------------------
    # Save as 3-panel text (6 columns each)
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
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Missing categories are reported as two distinct rows: [NA(d)] and [NA(n)] from the extract.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Panel {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table