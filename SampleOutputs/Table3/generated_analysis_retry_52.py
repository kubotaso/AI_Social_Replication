def run_analysis(data_source):
    import os
    import re
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
    # Helpers
    # -----------------------
    valid_codes = {1, 2, 3, 4, 5}

    # Accept common encodings for missing categories, if preserved in CSV
    # (strings like "[NA(d)]", "NA(d)", "DON'T KNOW", etc.). Keep it robust.
    dk_patterns = [
        r"\[NA\(D\)\]", r"\bNA\(D\)\b", r"\bDK\b", r"DON'?T\s+KNOW", r"\bDONT\s+KNOW\b",
        r"DON'?T\s+KNOW\s+MUCH", r"NOT\s+FAMILIAR", r"UNKNOWN",
    ]
    na_patterns = [
        r"\[NA\(N\)\]", r"\bNA\(N\)\b", r"NO\s+ANSWER", r"\bNA\b(?!\()", r"\bN/A\b",
    ]

    dk_re = re.compile("|".join(dk_patterns), flags=re.IGNORECASE)
    na_re = re.compile("|".join(na_patterns), flags=re.IGNORECASE)

    def classify_series(s: pd.Series):
        """
        Returns:
          valid_numeric: float series with values in 1..5 else NaN
          dk_mask: boolean mask for "Don't know much about it"
          na_mask: boolean mask for "No answer"
          other_missing_mask: boolean mask for remaining missing/nonvalid (excluded from mean; not shown in table)
        """
        raw = s

        # numeric parse for valid codes
        x = pd.to_numeric(raw, errors="coerce")
        valid_numeric = x.where(x.isin(list(valid_codes)), np.nan)

        # string parse for explicit DK/NA
        txt = raw.astype("string")

        # handle pandas <NA> nicely
        txt_norm = txt.str.strip()

        dk_mask = txt_norm.fillna("").str.contains(dk_re)
        na_mask = txt_norm.fillna("").str.contains(na_re)

        # also allow numeric-coded DK/NA if any exist (rare in this extract)
        # (kept conservative: only classify if non-1..5 numeric and not NaN)
        nonvalid_numeric = x.notna() & (~x.isin(list(valid_codes)))
        # If the dataset uses standard GSS missing codes (e.g., 8/9), map heuristically:
        # 8 = don't know, 9 = no answer (common in some GSS items)
        dk_mask = dk_mask | (nonvalid_numeric & x.eq(8))
        na_mask = na_mask | (nonvalid_numeric & x.eq(9))

        # Any other missing/nonvalid values not classified
        other_missing_mask = valid_numeric.isna() & ~(dk_mask | na_mask)

        return valid_numeric, dk_mask, na_mask, other_missing_mask

    # -----------------------
    # Validate required vars
    # -----------------------
    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

    # -----------------------
    # Build table (counts + mean)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        valid_numeric, dk_mask, na_mask, other_missing_mask = classify_series(df[var])

        counts_1_5 = (
            valid_numeric.value_counts(dropna=True)
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
        table.loc["Mean", genre_label] = float(valid_numeric.mean(skipna=True)) if valid_numeric.notna().any() else np.nan

        # If there are "other missing" not represented by DK/NA, keep computation valid but warn in file
        # (table spec only shows DK + No answer)
        # We'll record it in a sidecar summary for transparency.
        # (Do not fail; the table still computes from raw data.)
        # Store in attributes-like dict
        if "_other_missing" not in table.attrs:
            table.attrs["_other_missing"] = {}
        table.attrs["_other_missing"][genre_label] = int(other_missing_mask.sum())

    # -----------------------
    # Format + write text output (3 blocks of 6 genres)
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
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

        # Transparency about any non-DK/non-NA missing values found
        other_missing = table.attrs.get("_other_missing", {})
        any_other = any(v > 0 for v in other_missing.values())
        if any_other:
            f.write("Note: Some variables contain missing/nonvalid values not classified as '(M) Don't know much about it' or '(M) No answer'.\n")
            f.write("These are excluded from means and are not displayed in Table 3 (paper shows only the two M rows).\n")
            f.write("Other-missing counts by genre:\n")
            for g in genre_labels:
                f.write(f"  {g}: {other_missing.get(g, 0)}\n")

    return table