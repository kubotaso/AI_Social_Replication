def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to YEAR == 1993
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

    # ---- Missing-code handling ----
    # We must NEVER invent DK vs NA splits; we only split if the raw data encodes them distinctly.
    # If the dataset has only NaN (blank) missing values for these items, then DK vs NA are unsplittable.
    # In that case we report:
    #   DK = NaN (not available)
    #   NA = NaN (not available)
    # and still compute the mean correctly from valid 1..5 codes.
    def _detect_explicit_missing(series):
        """
        Returns dict with masks:
          dk_mask: NA(d) (don't know much about it)
          na_mask: NA(n) (no answer)
          other_missing_mask: other explicit NA(...) string codes (refused, skipped, etc.)
          numeric_missing_mask: numeric codes outside 1..5 (treated as missing, unsplittable)
          nan_mask: actual NaN
        """
        s = series

        # String-based explicit NA codes
        s_str = s.astype("string")
        s_up = s_str.str.strip().str.upper()

        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Any other [NA(x)] markers
        any_na_code = s_up.str.contains(r"\[NA\([A-Z]\)\]|\bNA\([A-Z]\)\b", regex=True, na=False)
        other_missing_mask = any_na_code & (~dk_mask) & (~na_mask)

        # Numeric parsing
        x = pd.to_numeric(s, errors="coerce")
        nan_mask = x.isna() & (~any_na_code)  # true NaN/blanks (not explicit NA strings)

        # Numeric codes outside 1..5 (e.g., 8, 9) are missing but do not allow DK vs NA split
        numeric_missing_mask = x.notna() & (~x.isin([1, 2, 3, 4, 5]))

        return {
            "dk_mask": dk_mask,
            "na_mask": na_mask,
            "other_missing_mask": other_missing_mask,
            "numeric_missing_mask": numeric_missing_mask,
            "nan_mask": nan_mask,
            "x": x,
        }

    # ---- Build table ----
    cols = [g[0] for g in genre_map]
    table = pd.DataFrame(index=row_labels, columns=cols, dtype="float64")

    missing_note_lines = []
    missing_note_lines.append("Missing-data handling for (M) rows:")
    missing_note_lines.append("- (M) rows are computed ONLY when explicit NA(d)/NA(n) codes exist in raw data.")
    missing_note_lines.append("- If the dataset encodes missing as plain NaN (blanks) or as unspecific numeric codes,")
    missing_note_lines.append("  then DK vs No answer cannot be separated and those cells are reported as 'NA'.")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        info = _detect_explicit_missing(df[var])
        x = info["x"]

        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Determine whether explicit DK/NA exist
        has_explicit_dk_na = (info["dk_mask"].sum() + info["na_mask"].sum()) > 0

        if has_explicit_dk_na:
            dk_count = int(info["dk_mask"].sum())
            na_count = int(info["na_mask"].sum())
        else:
            dk_count = np.nan
            na_count = np.nan

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for output ----
    def _fmt_cell(row_name, v):
        if pd.isna(v):
            return "NA"
        if row_name == "Mean":
            return f"{float(v):.2f}"
        return str(int(round(float(v))))

    formatted = pd.DataFrame(index=table.index, columns=table.columns, dtype="object")
    for r in table.index:
        formatted.loc[r] = [ _fmt_cell(r, table.loc[r, c]) for c in table.columns ]

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # ---- Save as three 6-column blocks ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [cols[i:i+6] for i in range(0, len(cols), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding all missing.\n\n")
        f.write("\n".join(missing_note_lines))
        f.write("\n\n")

        for bi, block_cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + block_cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table