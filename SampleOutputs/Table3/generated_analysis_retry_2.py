def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Load + filter
    # ----------------------------
    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names (robust to provided lower-case CSV)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")
    df = df.loc[df["YEAR"] == 1993].copy()

    # ----------------------------
    # Variable mapping (Table 3)
    # ----------------------------
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
        "(M) Don’t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # ----------------------------
    # Helpers: missing classification
    # ----------------------------
    def _as_string(s):
        # Keep true missing as <NA>, not "nan"
        return s.astype("string")

    def _classify_missing(raw_series):
        """
        Return masks for:
          - DK: "Don't know much about it"  -> NA(d)
          - NA: "No answer"                -> NA(n)

        We first look for explicit NA(d)/NA(n) string encodings.
        If not present (as in many released CSVs), we fall back to
        GSS convention for these music items:
          8 = Don't know much about it
          9 = No answer
        """
        s = _as_string(raw_series)
        s_upper = s.str.upper()

        # Explicit string encodings (rare in plain CSVs but supported)
        dk_mask = (
            s_upper.str.contains(r"\[?\s*NA\s*\(\s*D\s*\)\s*\]?", regex=True, na=False)
            | s_upper.str.contains(r"\bNA\s*\(\s*D\s*\)\b", regex=True, na=False)
        )
        na_mask = (
            s_upper.str.contains(r"\[?\s*NA\s*\(\s*N\s*\)\s*\]?", regex=True, na=False)
            | s_upper.str.contains(r"\bNA\s*\(\s*N\s*\)\b", regex=True, na=False)
        )

        # Numeric fallback for typical GSS coding on these items
        x = pd.to_numeric(raw_series, errors="coerce")
        # Only apply numeric fallback where we did not already classify via strings
        dk_mask = dk_mask | ((x == 8) & ~dk_mask & ~na_mask)
        na_mask = na_mask | ((x == 9) & ~dk_mask & ~na_mask)

        return dk_mask, na_mask

    # ----------------------------
    # Build numeric table
    # ----------------------------
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        x = pd.to_numeric(raw, errors="coerce")

        # Valid substantive responses for counts + mean
        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        counts_1_5 = (
            x.loc[valid_mask]
            .value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_mask, na_mask = _classify_missing(raw)

        # Ensure DK/NA counts are only from non-valid responses (avoid double counting)
        dk_count = int((dk_mask & ~valid_mask).sum())
        na_count = int((na_mask & ~valid_mask).sum())

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ----------------------------
    # Formatting (counts as integers; mean to 2 decimals)
    # ----------------------------
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # ----------------------------
    # Write output with 3-panel layout (6 + 6 + 6)
    # ----------------------------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    cols = list(formatted.columns)

    panels = [cols[0:6], cols[6:12], cols[12:18]]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for i, pcols in enumerate(panels, start=1):
            block = formatted.loc[:, pcols]
            f.write(f"Panel {i}\n")
            f.write(block.to_string(index=True))
            f.write("\n\n")

    return table