def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + standardize
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
            raise ValueError(f"Required variable not found: {v}")

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
    # Missing code handling
    # -----------------------
    # We must NEVER fabricate DK vs NA splits. We only compute them if the CSV preserves them.
    # Supported representations:
    #   - strings like "[NA(d)]", "NA(d)", "DK", "DON'T KNOW"
    #   - strings like "[NA(n)]", "NA(n)", "NO ANSWER"
    #   - numeric codes (common in some releases): 8=DK, 9=NA (we allow; only used if present)
    #
    # If the file collapses all missing into blank/NaN, we raise a clear error (requested by feedback).
    def _as_clean_string(s):
        return s.astype("string").str.strip().str.upper()

    def _missing_masks(raw):
        s_str = _as_clean_string(raw)

        # Explicit NA(d)/NA(n) string tags
        dk_str = (
            s_str.str.contains(r"\[NA\(D\)\]", regex=True, na=False)
            | s_str.str.fullmatch(r"NA\(D\)", na=False)
            | s_str.str.fullmatch(r"DK", na=False)
            | s_str.str.contains(r"DON'?T\s+KNOW", regex=True, na=False)
            | s_str.str.contains(r"DONT\s+KNOW", regex=True, na=False)
        )
        na_str = (
            s_str.str.contains(r"\[NA\(N\)\]", regex=True, na=False)
            | s_str.str.fullmatch(r"NA\(N\)", na=False)
            | s_str.str.contains(r"NO\s+ANSWER", regex=True, na=False)
        )

        # Numeric codes sometimes used
        x_num = pd.to_numeric(raw, errors="coerce")
        dk_num = x_num.eq(8)
        na_num = x_num.eq(9)

        dk = dk_str | dk_num
        na = na_str | na_num

        # Valid substantive responses
        valid = x_num.where(x_num.isin([1, 2, 3, 4, 5]), np.nan)

        # "Other missing" are any non-1..5 values not identified as DK/NA
        other_missing = valid.isna() & ~dk & ~na

        # Determine whether the dataset preserves DK/NA distinction for this item
        # We accept "preserved" if there exists any DK or NA code/label anywhere for this variable.
        preserved = bool(dk.any() or na.any())

        return valid, dk, na, other_missing, preserved

    # -----------------------
    # Build Table 3 (counts + mean)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    # Track if any variable lacks separable DK/NA codes
    unsplittable = []
    for genre_label, var in genre_map:
        raw = df[var]
        valid, dk, na, other_missing, preserved = _missing_masks(raw)

        # If missings exist but DK/NA not preserved, we cannot compute the two M rows from raw data
        if int(other_missing.sum()) > 0 and not preserved:
            unsplittable.append((var, int(other_missing.sum())))

        # Counts 1..5
        c = valid.value_counts(dropna=True).reindex([1, 2, 3, 4, 5], fill_value=0).astype(int)
        table.loc["(1) Like very much", genre_label] = c.loc[1]
        table.loc["(2) Like it", genre_label] = c.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = c.loc[3]
        table.loc["(4) Dislike it", genre_label] = c.loc[4]
        table.loc["(5) Dislike very much", genre_label] = c.loc[5]

        # Missing categories (only what we can compute)
        table.loc["(M) Don't know much about it", genre_label] = int(dk.sum())
        table.loc["(M) No answer", genre_label] = int(na.sum())

        # Mean on valid 1..5 only
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    if len(unsplittable) > 0:
        examples = ", ".join([f"{v}({n})" for v, n in unsplittable[:8]])
        raise ValueError(
            "Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts: "
            "this CSV export does not preserve distinguishable DK/NA categories (e.g., '[NA(d)]'/'[NA(n)]' "
            "or numeric 8/9). Found missing/non-1..5 values that are not classifiable. "
            f"Examples: {examples}. Re-export data preserving DK vs No-answer codes."
        )

    # -----------------------
    # Save human-readable table (Table 3 layout: 3 blocks of 6)
    # -----------------------
    formatted = table.copy()

    # Counts as integers; mean to 2 decimals
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
        f.write("Counts shown for each response category; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table