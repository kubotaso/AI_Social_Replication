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
    def _series_to_string(s):
        # Use pandas string dtype for robust NA handling
        return s.astype("string").str.strip()

    def _dk_na_masks(raw_series):
        """
        Determine masks for:
          - DK: [NA(d)] / NA(d) / 'DONT KNOW'
          - NA: [NA(n)] / NA(n) / 'NO ANSWER'

        Priority:
          1) If bracketed NA codes exist in the raw data, use them.
          2) Else, if 0/8/9 style codes exist, map 8->DK, 9->NA (common for GSS items).
          3) Else, treat all remaining missings as NA (cannot split reliably).
        """
        s_str = _series_to_string(raw_series).str.upper()

        # 1) Explicit bracketed missing codes
        dk = s_str.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s_str.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        if int(dk.sum()) + int(na.sum()) > 0:
            return dk, na

        # 2) Numeric-coded missing values (e.g., 8=DK, 9=NA)
        x = pd.to_numeric(raw_series, errors="coerce")

        dk2 = x.eq(8)
        na2 = x.eq(9)
        if int(dk2.sum()) + int(na2.sum()) > 0:
            return dk2, na2

        # 3) Word-coded missing values
        dk3 = s_str.str.contains(r"DON'?T\s+KNOW|DONT\s+KNOW|\bDK\b", regex=True, na=False)
        na3 = s_str.str.contains(r"NO\s+ANSWER|\bNA\b", regex=True, na=False)
        if int(dk3.sum()) + int(na3.sum()) > 0:
            return dk3, na3

        # 4) Fallback: cannot split; count all remaining missing as "No answer"
        missing = x.isna() | s_str.isna()
        return pd.Series(False, index=raw_series.index), missing

    # ---- Build numeric table (counts + mean) ----
    genre_labels = [g[0] for g in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        # Valid 1-5
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_mask, na_mask = _dk_na_masks(raw)

        # Ensure DK/NA do not count any valid 1-5 (paranoia)
        valid_mask = x.isin([1, 2, 3, 4, 5])
        dk_mask = dk_mask & (~valid_mask)
        na_mask = na_mask & (~valid_mask) & (~dk_mask)

        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for display ----
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

    # ---- Save as three 6-column blocks ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            f.write(display.loc[:, cols_with_stub].to_string(index=False))
            f.write("\n\n")

    return table