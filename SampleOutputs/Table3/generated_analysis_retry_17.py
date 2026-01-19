def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---------- Load ----------
    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names for robustness
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to 1993 only
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ---------- Variables (Table 3) ----------
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

    missing_label_map = {
        "D": "(M) Don't know much about it",
        "N": "(M) No answer",
    }

    # ---------- Helpers ----------
    def _to_str_upper(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _missing_letter(s_up: pd.Series) -> pd.Series:
        """
        Extracts the missing-code letter from tokens like:
          "[NA(d)]", "NA(d)", "na(d)", "[na(n)]"
        Returns uppercase letter (e.g., "D", "N") or <NA>.
        """
        # Capture the letter inside NA(...)
        letter = s_up.str.extract(r"NA\s*\(\s*([A-Z])\s*\)", expand=False)
        return letter

    def _counts_and_mean(raw: pd.Series):
        """
        Compute counts for codes 1..5 plus DK=[NA(d)] and NA=[NA(n)].
        Other missing codes are ignored for DK/NA rows (still excluded from mean).
        """
        s_up = _to_str_upper(raw)

        # Identify explicit NA(d) / NA(n) codes if present as strings
        miss_letter = _missing_letter(s_up)
        dk_mask = miss_letter.eq("D")
        na_mask = miss_letter.eq("N")

        # Numeric parsing (handles floats and numeric strings)
        x = pd.to_numeric(raw, errors="coerce")

        # Valid substantive responses
        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid_num = x.where(valid_mask, np.nan)

        # Counts for 1..5
        counts_1_5 = (
            x.loc[valid_mask]
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # DK/NA counts:
        #   If explicit NA(d)/NA(n) are present, count them directly.
        #   Else, treat DK/NA as not distinguishable in this extract -> set to 0 (not inferred).
        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        return counts_1_5, dk_count, na_count, mean_val

    # ---------- Build numeric table ----------
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        counts_1_5, dk_count, na_count, mean_val = _counts_and_mean(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---------- Format for output (counts as ints; mean to 2 decimals) ----------
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # Add explicit stub column for row labels
    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # ---------- Save as three 6-column blocks ----------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: DK/NA rows are computed only if explicit GSS missing codes [NA(d)]/[NA(n)] are present in the data extract.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            block_df = display.loc[:, cols_with_stub]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table