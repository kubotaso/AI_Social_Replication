def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 ---
    year_col = None
    for c in ["year", "YEAR"]:
        if c in df.columns:
            year_col = c
            break
    if year_col is None:
        raise KeyError("YEAR/year column not found in dataset.")
    df = df.loc[df[year_col] == 1993].copy()

    # --- Music genre variables (Table 3 columns) ---
    genres = [
        ("Latin/Salsa", "latin"),
        ("Jazz", "jazz"),
        ("Blues/R&B", "blues"),
        ("Show Tunes", "musicals"),
        ("Oldies", "oldies"),
        ("Classical/Chamber", "classicl"),
        ("Reggae", "reggae"),
        ("Swing/Big Band", "bigband"),
        ("New Age/Space", "newage"),
        ("Opera", "opera"),
        ("Bluegrass", "blugrass"),
        ("Folk", "folk"),
        ("Pop/Easy Listening", "moodeasy"),
        ("Contemporary Rock", "conrock"),
        ("Rap", "rap"),
        ("Heavy Metal", "hvymetal"),
        ("Country/Western", "country"),
        ("Gospel", "gospel"),
    ]

    # Resolve column names case-insensitively (defensive)
    colmap = {c.lower(): c for c in df.columns}

    # --- Row labels (Table 3 rows) ---
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

    # --- Missing code detection (compute, do not hardcode) ---
    # Many GSS CSV extracts store special missing values as negative integers
    # (e.g., -1/-2), or as large sentinels (8/9, 98/99), or as labeled strings.
    DK_TOKENS = {
        "d", "dk", "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_TOKENS = {"n", "na", "no answer", "noanswer"}

    # Candidate numeric codes; we will decide which are DK vs NA by inspecting labels (if any),
    # otherwise fall back to a conservative numeric heuristic.
    NUMERIC_MISSING_CANDIDATES = [-99, -98, -97, -2, -1, 0, 8, 9, 98, 99]

    def _infer_missing_codes(series: pd.Series):
        """
        Infer DK and NA codes for this series using (in order):
        1) pandas Categorical categories (rare in CSV), if present
        2) string tokens in raw data
        3) numeric sentinel candidates; try to infer DK vs NA by prevalence and common conventions
        Returns: (dk_mask, na_mask, valid_mask_1to5)
        """
        s = series

        # 1) If categorical with named categories, use category names
        if isinstance(s.dtype, pd.CategoricalDtype):
            cat = s.cat
            cats = list(cat.categories)
            # Build mapping from category label -> code
            # Note: categories might already be numeric; if so, skip this.
            if all(isinstance(x, str) for x in cats):
                lowcats = [str(x).strip().lower() for x in cats]
                dk_cats = {cats[i] for i, lc in enumerate(lowcats) if lc in DK_TOKENS}
                na_cats = {cats[i] for i, lc in enumerate(lowcats) if lc in NA_TOKENS}
                dk_mask = s.isin(list(dk_cats))
                na_mask = s.isin(list(na_cats))
                sn = pd.to_numeric(s.astype("string"), errors="coerce")
                valid_mask = sn.isin([1, 2, 3, 4, 5])
                return dk_mask.fillna(False), na_mask.fillna(False), valid_mask.fillna(False)

        # 2) Strings in raw data
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            ss = s.astype("string")
            low = ss.str.strip().str.lower()

            dk_mask = low.isin(DK_TOKENS)
            na_mask = low.isin(NA_TOKENS)

            # Numeric responses might also be stored as strings
            sn = pd.to_numeric(low, errors="coerce")
            valid_mask = sn.isin([1, 2, 3, 4, 5])

            return dk_mask.fillna(False), na_mask.fillna(False), valid_mask.fillna(False)

        # 3) Numeric: infer from common sentinel sets
        sn = pd.to_numeric(s, errors="coerce")

        valid_mask = sn.isin([1, 2, 3, 4, 5])

        # Check which candidate codes are present
        present = [code for code in NUMERIC_MISSING_CANDIDATES if (sn == code).any()]
        if not present:
            # If there are NaNs, we cannot distinguish DK vs NA; treat both as 0.
            dk_mask = pd.Series(False, index=sn.index)
            na_mask = pd.Series(False, index=sn.index)
            return dk_mask, na_mask, valid_mask

        # Prefer negative GSS-style: -1 (DK), -2 (NA) if present
        dk_codes = set()
        na_codes = set()
        if -1 in present:
            dk_codes.add(-1)
        if -2 in present:
            na_codes.add(-2)

        # If 8/9 or 98/99 present and not already assigned
        if not dk_codes and 8 in present:
            dk_codes.add(8)
        if not na_codes and 9 in present:
            na_codes.add(9)
        if not dk_codes and 98 in present:
            dk_codes.add(98)
        if not na_codes and 99 in present:
            na_codes.add(99)

        # If still unassigned, use a conservative fallback:
        # assign the more frequent code among remaining candidates to DK and next to NA
        remaining = [c for c in present if c not in dk_codes and c not in na_codes]
        if remaining:
            freqs = sorted([(c, int((sn == c).sum())) for c in remaining], key=lambda x: (-x[1], x[0]))
            if not dk_codes and freqs:
                dk_codes.add(freqs[0][0])
            if not na_codes and len(freqs) > 1:
                na_codes.add(freqs[1][0])

        dk_mask = sn.isin(list(dk_codes))
        na_mask = sn.isin(list(na_codes))
        return dk_mask.fillna(False), na_mask.fillna(False), valid_mask.fillna(False)

    # --- Build output table (counts + mean) ---
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype="object")

    # Pre-fill with empty strings to avoid "nan" in text output
    table.loc[:, :] = ""

    for genre_label, var_lower in genres:
        if var_lower not in colmap:
            raise KeyError(f"Expected variable '{var_lower}' not found in dataset.")
        var = colmap[var_lower]

        s = df[var]

        # numeric version for substantive frequencies and mean
        sn = pd.to_numeric(s, errors="coerce")

        # counts for 1..5
        for k, rlab in zip(
            [1, 2, 3, 4, 5],
            row_labels[:5]
        ):
            table.loc[rlab, genre_label] = str(int((sn == k).sum()))

        # missing type counts (DK and NA)
        dk_mask, na_mask, valid_mask = _infer_missing_codes(s)
        table.loc["(M) Don’t know much about it", genre_label] = str(int(dk_mask.sum()))
        table.loc["(M) No answer", genre_label] = str(int(na_mask.sum()))

        # mean on valid 1..5 only (exclude DK/NA and any other codes)
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = "" if pd.isna(mean_val) else f"{float(mean_val):.2f}"

    # --- Save human-readable text file ---
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(table.to_string())
        f.write("\n")

    return table