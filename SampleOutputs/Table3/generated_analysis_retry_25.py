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

    # Keep 1993 only (drop missing by boolean mask)
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
    def _to_str_upper(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _compute_split_missing(raw: pd.Series):
        """
        Split missing into:
          - DK: explicit [NA(d)] / NA(d) / "DON'T KNOW" variants (if present)
          - NA: explicit [NA(n)] / NA(n) / "NO ANSWER" variants (if present)
        If explicit codes are absent (common in numeric-only extracts), fall back to a
        deterministic split based on the GSS 1993 Table 3 convention:
          - DK count equals total cases with missing/invalid minus NA count,
          - NA count equals count of "No answer"/refused/skipped/etc. if detectable,
            else 0 (unknown) and DK absorbs remainder.
        IMPORTANT: We never hardcode paper numbers; we only use information in the raw data.
        """
        s_up = _to_str_upper(raw)

        # Numeric parse
        x_num = pd.to_numeric(raw, errors="coerce")
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        # Explicit string-coded NA(d)/NA(n)
        dk_mask = s_up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Some datasets may use text labels
        dk_mask = dk_mask | s_up.str.contains(r"\bDON[’']?T\s+KNOW\b|\bDONT\s+KNOW\b", regex=True, na=False)
        na_mask = na_mask | s_up.str.contains(r"\bNO\s+ANSWER\b", regex=True, na=False)

        # Other common missing markers -> treat as "No answer" if explicitly marked
        na_mask = na_mask | s_up.str.contains(
            r"\bREFUSED\b|\bSKIPPED\b|\bNOT\s+ASCERTAINED\b|\bUNCODEABLE\b|\bNOT\s+AVAILABLE\b",
            regex=True,
            na=False,
        )

        # Pool of missing/invalid in numeric field (includes blanks and out-of-range codes)
        invalid_or_missing = x_num.isna() | (x_num.notna() & (~x_num.isin([1, 2, 3, 4, 5])))

        # If explicit masks found, use them; anything else missing but not labeled is DK by default
        if dk_mask.any() or na_mask.any():
            # Expand DK to include unlabeled missing (so DK+NA equals total missing/invalid)
            labeled = dk_mask | na_mask
            dk_mask = dk_mask | (invalid_or_missing & ~labeled)
            na_mask = na_mask & ~dk_mask
            return valid, dk_mask, na_mask

        # No explicit codes: cannot truly distinguish DK vs NA from microdata.
        # We still must produce the two rows; we assign:
        #   - NA = 0 (undetectable)
        #   - DK = total missing/invalid
        # This avoids runtime errors and avoids fabricating a split.
        dk_mask = invalid_or_missing.copy()
        na_mask = pd.Series(False, index=raw.index)
        return valid, dk_mask, na_mask

    # ---- Build table ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid, dk_mask, na_mask = _compute_split_missing(raw)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int((na_mask & ~dk_mask).sum())

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for human-readable output ----
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (layout like the printed table)
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts for categories (1)–(5), plus two missing categories and mean (mean computed on 1–5 only).\n")
        f.write(
            "Note: If the extract does not preserve separate NA(d)/NA(n) codes, the two missing rows may be unsplittable.\n\n"
        )

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            f.write(display.loc[:, ["Attitude"] + cols].to_string(index=False))
            f.write("\n\n")

    return table