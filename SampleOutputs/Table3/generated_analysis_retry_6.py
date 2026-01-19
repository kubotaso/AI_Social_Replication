def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---------- Read data ----------
    df = pd.read_csv(data_source, low_memory=False)

    # Standardize column names to uppercase for robust access
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to YEAR == 1993 (exclude missing/NA automatically by equality check)
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ---------- Variable mapping ----------
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

    # ---------- Helpers ----------
    def _as_string_upper(s):
        # Keep pandas NA as <NA>, then uppercase for stable matching
        return s.astype("string").str.strip().str.upper()

    def _extract_bracket_code(u):
        """
        Extracts missing-code letter from formats like:
          "[NA(d)]", "NA(d)", "[NA(D)]", etc.
        Returns a Series of single letters (e.g., 'D', 'N') or <NA>.
        """
        # Match either "[NA(d)]" or "NA(d)" (case-insensitive after upper)
        m = u.str.extract(r"(?:\[\s*NA\(\s*([A-Z])\s*\)\s*\]|NA\(\s*([A-Z])\s*\))", expand=True)
        # m has two columns; pick the first non-null
        code = m[0].fillna(m[1])
        return code.astype("string")

    def _parse_numeric(raw):
        # Convert numerics stored as strings like "3.0" safely; non-numeric becomes NaN
        return pd.to_numeric(raw, errors="coerce")

    def _dk_na_masks(raw_series):
        """
        Determine DK and NA masks using bracketed NA codes if present.
        If not present, attempt numeric inference using common GSS patterns.
        """
        u = _as_string_upper(raw_series)
        code_letter = _extract_bracket_code(u)

        # Primary: explicit labeled missing codes
        dk_mask = code_letter.eq("D")  # don't know (instrument label used for DK much about it)
        na_mask = code_letter.eq("N")  # no answer

        # If explicit labels exist, use them and stop
        if int(dk_mask.sum()) + int(na_mask.sum()) > 0:
            return dk_mask.fillna(False), na_mask.fillna(False)

        # Fallback: numeric-coded DK/NA (common in some extracts)
        x = _parse_numeric(raw_series)

        # Candidate non-substantive numeric codes (exclude 1..5)
        non_substantive = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
        if not non_substantive.any():
            return pd.Series(False, index=raw_series.index), pd.Series(False, index=raw_series.index)

        vc = x.loc[non_substantive].value_counts(dropna=True)

        # Common GSS conventions: 8=DK, 9=NA (if present)
        dk_code = None
        na_code = None
        if 8 in vc.index:
            dk_code = 8
        if 9 in vc.index:
            na_code = 9

        # If 8/9 not present, infer from top frequencies:
        # DK tends to be more frequent than NA.
        if dk_code is None and len(vc) >= 1:
            dk_code = vc.index[0]
        if na_code is None and len(vc) >= 2:
            # choose the most frequent code that isn't dk_code
            for c in vc.index[1:]:
                if c != dk_code:
                    na_code = c
                    break

        dk_mask = x.eq(dk_code) if dk_code is not None else pd.Series(False, index=raw_series.index)
        na_mask = x.eq(na_code) if na_code is not None else pd.Series(False, index=raw_series.index)

        # Avoid overlap (shouldn't happen, but keep safe)
        na_mask = na_mask & (~dk_mask)
        return dk_mask.fillna(False), na_mask.fillna(False)

    # ---------- Build table ----------
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        # Identify DK / NA (counts shown as rows in the table)
        dk_mask, na_mask = _dk_na_masks(raw)

        # Numeric substantive responses (1..5)
        x = _parse_numeric(raw)
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Counts for 1..5
        counts_1_5 = valid.value_counts(dropna=True).reindex([1, 2, 3, 4, 5], fill_value=0).astype(int)

        # Counts for missing rows
        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())

        # Mean on 1..5 only
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---------- Format for human-readable output ----------
    formatted = table.copy()

    # Counts as ints, Mean rounded to 2 decimals
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # ---------- Save as 3 blocks of 6 columns ----------
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = formatted.loc[:, cols].copy()
            block_df.index.name = "Attitude"
            f.write(block_df.to_string())
            f.write("\n\n")

    return table