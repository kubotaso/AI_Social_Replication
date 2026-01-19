def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + standardize columns
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to 1993 only (exclude missing YEAR implicitly)
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

    missing_code_map = {
        "D": "(M) Don't know much about it",
        "N": "(M) No answer",
    }

    row_order = [
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
    # Helpers to detect explicit missing tags like "[NA(d)]"
    # -----------------------
    def _as_str_series(s):
        # Keep <NA> for missing; safe operations
        return s.astype("string").str.strip()

    def _extract_na_letter(token_series):
        """
        Extract NA letter from patterns like:
          [NA(d)], NA(d), [na(D)], etc.
        Returns uppercase letter or <NA>.
        """
        s = _as_str_series(token_series).str.upper()
        # Capture the letter inside NA(...)
        # Examples matched: [NA(D)], NA(D)
        letter = s.str.extract(r"(?:\[\s*)?NA\(\s*([A-Z])\s*\)(?:\s*\])?", expand=False)
        return letter

    def _tabulate_one(series, varname):
        """
        Returns:
          counts_1_5: pd.Series indexed [1..5]
          dk_count: int
          na_count: int
          mean_val: float (mean of valid 1..5)
          used_explicit_missing: bool
        """
        raw = series

        # Identify explicit NA(...) codes if present in string form
        na_letter = _extract_na_letter(raw)

        has_explicit = na_letter.notna().any()

        # Parse numeric values (works for numeric columns; strings -> numeric)
        x = pd.to_numeric(raw, errors="coerce")

        # Valid substantive responses: 1..5 only
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        if has_explicit:
            # Explicit missing codes exist; count DK/NA from those codes.
            dk_count = int((na_letter == "D").sum())
            na_count = int((na_letter == "N").sum())

            # Any other non-1..5 and non-explicit-missing are treated as missing but NOT
            # assignable to DK vs NA; fold into "No answer" to keep the table complete
            # without inventing a new row (paper has only these two M rows).
            other_missing = valid.isna() & na_letter.isna()
            if other_missing.any():
                na_count += int(other_missing.sum())
        else:
            # No explicit missing codes preserved in this CSV.
            # We can still compute valid 1..5 counts and the mean, but cannot split the
            # missing pool into DK vs No answer without external information.
            # Do NOT guess: keep integrity by setting DK/NA as NaN and report combined missing.
            dk_count = np.nan
            na_count = np.nan

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_count, na_count, mean_val, bool(has_explicit)

    # -----------------------
    # Build table
    # -----------------------
    genre_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_order, columns=genre_labels, dtype="float64")

    missing_split_available_all = True

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        counts_1_5, dk_n, na_n, mean_val, has_explicit = _tabulate_one(df[var], var)
        missing_split_available_all = missing_split_available_all and has_explicit

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Save human-readable output (three 6-column blocks like the paper)
    # -----------------------
    def _format_for_print(tdf):
        out = tdf.copy()
        for r in out.index:
            if r == "Mean":
                out.loc[r] = out.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
            else:
                out.loc[r] = out.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))
        out.insert(0, "Attitude", out.index)
        out = out.reset_index(drop=True)
        return out

    formatted = _format_for_print(table)

    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]
    out_path = "./output/table3_frequency_distributions_gss1993.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        if not missing_split_available_all:
            f.write(
                "\nNOTE: This CSV export does not preserve distinguishable missing categories (e.g., '[NA(d)]' vs '[NA(n)]')\n"
                "for at least one item. The two '(M)' rows are left blank where the split cannot be computed.\n"
                "Re-export data preserving those codes to reproduce the paper's two missing rows.\n"
            )
        f.write("\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"\nBlock {bi}:\n")
            block_df = formatted.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n")

    return table