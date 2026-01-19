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

    # Filter to 1993 only (exclude missing YEAR automatically via comparison)
    df = df.loc[df["YEAR"] == 1993].copy()

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

    for label, var in genre_map:
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
    # Missing-category parsing
    # -----------------------
    # We must NOT invent DK/NA counts. We compute them only if the raw column
    # preserves distinguishable categories (strings like "[NA(d)]"/"[NA(n)]",
    # or explicit labels/codes). If not preserved, we set DK/NA to NA (blank in
    # the printed table) and still compute valid 1–5 frequencies + mean.
    dk_patterns = [
        r"\[NA\(D\)\]", r"\bNA\(D\)\b",
        r"DON'?T\s+KNOW", r"\bDK\b",
        r"DON'?T\s+KNOW\s+MUCH"
    ]
    na_patterns = [
        r"\[NA\(N\)\]", r"\bNA\(N\)\b",
        r"\bNO\s+ANSWER\b", r"\bNA\b"
    ]

    def _as_clean_str(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _detect_dk_na(raw: pd.Series):
        s = _as_clean_str(raw)

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        # Detect DK and NA via patterns on string representation
        for pat in dk_patterns:
            dk_mask = dk_mask | s.str.contains(pat, regex=True, na=False)
        for pat in na_patterns:
            na_mask = na_mask | s.str.contains(pat, regex=True, na=False)

        # If strings preserved, great. If not, these will remain all-False.
        return dk_mask, na_mask

    def _tabulate_one(raw: pd.Series):
        # Valid numeric 1..5 (mean computed from these only)
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Missing categories (DK vs NA) ONLY if distinguishable
        dk_mask, na_mask = _detect_dk_na(raw)

        # If numeric codes exist for DK/NA (common: 8/9, 98/99), detect them too
        # but only among non-valid (not 1..5) numeric values.
        nonvalid_num = x.where(~x.isin([1, 2, 3, 4, 5]) & x.notna(), np.nan)

        # Candidate DK/NA numeric codes (best-effort; only used if present)
        dk_num_codes = {8, 98}
        na_num_codes = {9, 99}

        dk_mask = dk_mask | nonvalid_num.isin(list(dk_num_codes))
        na_mask = na_mask | nonvalid_num.isin(list(na_num_codes))

        # Determine if we can credibly separate DK vs NA:
        # - If we found at least one DK or NA marker, we report counts.
        # - Otherwise, DK/NA are not distinguishable in this CSV; report as NaN.
        can_separate = bool(int(dk_mask.sum()) + int(na_mask.sum()) > 0)

        dk_n = int(dk_mask.sum()) if can_separate else np.nan
        na_n = int(na_mask.sum()) if can_separate else np.nan

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build table (numeric)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Formatting + write text output
    # -----------------------
    formatted = table.copy()

    def _fmt_cell(row_name, v):
        if pd.isna(v):
            return ""
        if row_name == "Mean":
            return f"{float(v):.2f}"
        return str(int(round(float(v))))

    for r in formatted.index:
        formatted.loc[r] = [ _fmt_cell(r, v) for v in formatted.loc[r].tolist() ]

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: '(M)' rows are reported only if DK vs No-answer are distinguishable in the raw export.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table