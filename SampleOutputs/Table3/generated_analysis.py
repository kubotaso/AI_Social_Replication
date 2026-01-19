def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names to upper for robust access
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Filter: YEAR == 1993
    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")
    df = df.loc[df["YEAR"] == 1993].copy()

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

    # Rows (as in the paper table)
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

    # Helper: coerce to numeric where possible
    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    # Try to detect NA(d) and NA(n) if present as strings; otherwise treat as NaN.
    # Common encodings: "[NA(d)]", "NA(d)", "DONT KNOW", "DON'T KNOW", "DK", etc.
    def _classify_missing(raw_series):
        s = raw_series.astype("string")
        s_upper = s.str.upper()

        # Initialize as unknown missing bucket
        dk_mask = pd.Series(False, index=s.index)
        na_mask = pd.Series(False, index=s.index)

        # Patterns for "don't know much about it"
        dk_patterns = [
            r"\[?NA\(D\)\]?",
            r"\bNA\(D\)\b",
            r"\bDONT\s*KNOW\b",
            r"\bDON'T\s*KNOW\b",
            r"\bDK\b",
            r"\bDONOTKNOW\b",
        ]
        # Patterns for "no answer"
        na_patterns = [
            r"\[?NA\(N\)\]?",
            r"\bNA\(N\)\b",
            r"\bNO\s*ANSWER\b",
            r"\bNA\b",
            r"\bN/A\b",
        ]

        for pat in dk_patterns:
            dk_mask = dk_mask | s_upper.str.contains(pat, regex=True, na=False)
        for pat in na_patterns:
            na_mask = na_mask | s_upper.str.contains(pat, regex=True, na=False)

        return dk_mask, na_mask

    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        # Numeric responses (1-5)
        x = _to_num(raw)
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Counts for 1-5
        counts_1_5 = valid.value_counts(dropna=True).reindex([1, 2, 3, 4, 5], fill_value=0).astype(int)

        # Missing detail buckets (best-effort)
        dk_mask, na_mask = _classify_missing(raw)

        # Only consider those not valid 1-5 as missing candidates
        not_valid = ~x.isin([1, 2, 3, 4, 5])
        dk_count = int((dk_mask & not_valid).sum())
        na_count = int((na_mask & not_valid & ~dk_mask).sum())

        # If codes are numeric but represent DK/NA (unknown coding), they will land in not_valid but not in dk/na masks.
        # Keep them excluded from mean; do not force them into "No answer" unless they are explicit.
        # The table spec only shows two M rows; here we report explicit matches, otherwise 0.

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # Format for human-readable text file
    def _format_cell(row_name, val):
        if row_name == "Mean":
            if pd.isna(val):
                return ""
            return f"{val:.2f}"
        if pd.isna(val):
            return ""
        return str(int(val))

    formatted = table.copy()
    for r in formatted.index:
        formatted.loc[r] = formatted.loc[r].map(lambda v, rr=r: _format_cell(rr, v))

    # Save as fixed-width text
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        f.write(formatted.to_string(index=True))

    return table