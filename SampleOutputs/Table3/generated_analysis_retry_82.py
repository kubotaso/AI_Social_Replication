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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required column not found: {v}")

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
    # Helpers for DK/NA parsing
    # -----------------------
    def _as_clean_str(series: pd.Series) -> pd.Series:
        # keep <NA> as <NA>; normalize whitespace/case
        s = series.astype("string")
        s = s.str.strip()
        s = s.str.replace(r"\s+", " ", regex=True)
        return s

    def _explicit_missing_masks(raw: pd.Series):
        """
        Return boolean masks (dk_mask, na_mask) when the CSV preserves codes/labels.
        Supports:
          - literal tokens: [NA(d)] / [NA(n)] (any case)
          - plain tokens: NA(d) / NA(n)
          - common text labels: 'dont know', 'don't know', 'no answer', 'refused', etc.
        """
        s = _as_clean_str(raw).str.lower()

        # Tokens like [NA(d)] / NA(d)
        dk_token = s.str.contains(r"\[?na\(\s*d\s*\)\]?", regex=True, na=False)
        na_token = s.str.contains(r"\[?na\(\s*n\s*\)\]?", regex=True, na=False)

        # Common label fallbacks (only if present in the data export)
        dk_text = s.str.contains(r"\bdon'?t know\b|\bdont know\b|\bdk\b", regex=True, na=False)
        na_text = s.str.contains(r"\bno answer\b|\bna\b(?!\()", regex=True, na=False)

        # Prefer explicit NA(d)/NA(n); text labels can help if export used labels
        dk = dk_token | dk_text
        na = na_token | na_text

        return dk, na

    def _extract_counts_and_mean(raw: pd.Series, varname: str):
        """
        Compute:
          - counts for 1..5
          - DK count
          - NA count
          - mean over valid 1..5
        Requirement: DK and NA must be distinguishable from the CSV.
        If they are not, raise a clear error (do not guess).
        """
        # numeric valid 1..5
        x_num = pd.to_numeric(raw, errors="coerce")
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        # Identify explicit DK/NA codes/labels in original raw (string-aware)
        dk_mask, na_mask = _explicit_missing_masks(raw)

        # Anything non-1..5 and not numeric is missing; but to split DK vs NA we need explicit markers.
        # Determine "missing pool" as values that are not valid 1..5 (including NaN numeric).
        missing_pool = valid.isna()

        # If there are missing values but none are explicitly classifiable into DK or NA, we cannot proceed.
        classifiable = dk_mask | na_mask
        unclassifiable_missing = missing_pool & ~classifiable

        if int(missing_pool.sum()) > 0 and int(classifiable.sum()) == 0:
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                "this CSV export does not preserve distinguishable missing categories (e.g., '[NA(d)]'/'[NA(n)]' "
                "or DK/No-answer labels or distinct numeric codes). "
                f"Example missing/non-1..5 count in {varname}: {int(missing_pool.sum())}. "
                "Re-export data preserving DK vs No-answer codes."
            )

        if int(unclassifiable_missing.sum()) > 0:
            # Some exports may preserve DK/NA for some cases but not all; that's still not acceptable.
            # We refuse rather than silently dumping into one bucket.
            # Provide a few example raw values.
            examples = raw.loc[unclassifiable_missing].astype("string").dropna().unique()[:5]
            examples_txt = ", ".join([str(e) for e in examples]) if len(examples) else "NA/blank"
            raise ValueError(
                "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                f"found {int(unclassifiable_missing.sum())} missing/non-1..5 values in {varname} that are not classifiable. "
                f"Examples: {examples_txt}. Re-export data preserving DK vs No-answer codes."
            )

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _extract_counts_and_mean(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable output (3 blocks of 6)
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

    return table