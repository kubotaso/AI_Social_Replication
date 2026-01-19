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

    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

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

    # -----------------------
    # Helpers
    # -----------------------
    def _to_clean_string(s: pd.Series) -> pd.Series:
        # Preserve explicit missing labels if present, without losing numeric info.
        return s.astype("string").str.strip()

    def _extract_explicit_missing_masks(raw: pd.Series):
        """
        Return masks for DK and NA when explicitly encoded as strings like:
        '[NA(d)]', 'NA(d)', '[na(d)]', etc.
        """
        s = _to_clean_string(raw).str.upper()

        dk = s.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        return dk, na

    def _compute_counts_and_mean(raw: pd.Series, varname: str):
        """
        Compute:
          - counts for 1..5
          - counts for DK and NA (must be distinguishable)
          - mean on valid 1..5
        The function supports:
          A) Explicit NA(d)/NA(n) string encodings, OR
          B) Distinct numeric codes for DK and NA (auto-detected), OR
          C) If neither exists, raises an error (cannot split).
        """
        # numeric parse
        x = pd.to_numeric(raw, errors="coerce")

        # valid substantive
        valid_mask = x.isin([1, 2, 3, 4, 5])
        valid = x.where(valid_mask, np.nan)

        counts_1_5 = (
            x.where(valid_mask)
            .value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # 1) Try explicit string missing labels
        dk_mask_s, na_mask_s = _extract_explicit_missing_masks(raw)
        if int(dk_mask_s.sum()) + int(na_mask_s.sum()) > 0:
            # Anything else that is non-1..5 and non-explicit DK/NA is just "other missing"
            # but Table 3 only wants DK and NA; if other missing exists, we fold it into NA.
            other_missing = (~valid_mask) & (~dk_mask_s) & (~na_mask_s)
            na_mask_s = na_mask_s | other_missing
            dk_n = int(dk_mask_s.sum())
            na_n = int(na_mask_s.sum())
            mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
            return counts_1_5, dk_n, na_n, mean_val

        # 2) Try distinct numeric missing codes for DK/NA (auto-detect)
        # Candidate: values that are numeric, non-1..5, and not NaN.
        nonvalid = x[~valid_mask & x.notna()]

        # If dataset uses 0, 8, 9, 98, 99, etc. we might see them here.
        # If we see exactly two distinct codes, treat them as DK and NA.
        uniq = sorted(nonvalid.unique().tolist())

        if len(uniq) == 2:
            # Heuristic mapping: NA (no answer) is typically the "hard" missing like 9/99,
            # DK (don't know) typically the "softer" missing like 8/98. Use larger code as NA.
            dk_code, na_code = (uniq[0], uniq[1])
            dk_n = int((x == dk_code).sum())
            na_n = int((x == na_code).sum())
            mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
            return counts_1_5, dk_n, na_n, mean_val

        # 3) If there are no explicit labels and only NaNs (or one/zero codes), we cannot split.
        total_missing = int((~valid_mask).sum())
        if total_missing > 0:
            raise ValueError(
                f"Cannot compute separate '(M) Don’t know much about it' vs '(M) No answer' counts for {varname}: "
                f"dataset does not preserve distinct DK/NA codes (explicit '[NA(d)]'/'[NA(n)]' strings "
                f"or two distinct numeric missing codes). Found {total_missing} missing/non-1..5 values."
            )

        # No missing at all
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, 0, 0, mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _compute_counts_and_mean(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable text output (three 6-column blocks)
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
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table