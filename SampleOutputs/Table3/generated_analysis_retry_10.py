def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 (robust to case/whitespace) ---
    year_col = next((c for c in df.columns if str(c).strip().lower() == "year"), None)
    if year_col is None:
        raise KeyError("YEAR/year column not found in dataset.")
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # --- Genre variables (Table 3 columns): (display label, dataset field lower) ---
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

    # Resolve actual column names case-insensitively
    colmap = {str(c).strip().lower(): c for c in df.columns}
    missing = [var for _, var in genres if var not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # --- Table 3 row labels (exact order) ---
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

    # --- Missing code handling ---
    # For this dataset, the "M" categories appear to be stored as blank/NaN in the CSV.
    # Still support common GSS-style numeric/text codes if present.
    DK_STR_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_STR_TOKENS = {"n", "na", "no answer", "noanswer", "none"}

    def classify_masks(series: pd.Series):
        """
        Returns masks: valid(1..5), dk, na.
        Priority:
          1) If explicit DK/NA codes exist (numeric or string), count them.
          2) Else, treat blank/NaN as DK (matches this CSV extract's pattern for these items).
        """
        s = series

        # Numeric parsing for substantive/explicit codes
        sn = pd.to_numeric(s, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        # Detect explicit numeric DK/NA codes if present
        present = set(sn.dropna().unique().tolist())
        dk_codes = {code for code in (8, 98, -1) if code in present}
        na_codes = {code for code in (9, 99, -2) if code in present}
        dk_num = sn.isin(list(dk_codes)).fillna(False) if dk_codes else pd.Series(False, index=s.index)
        na_num = sn.isin(list(na_codes)).fillna(False) if na_codes else pd.Series(False, index=s.index)

        # Detect explicit string DK/NA tokens if present
        is_str = (s.dtype == "object") or str(s.dtype).startswith("string")
        if is_str:
            low = s.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR_TOKENS).fillna(False)
            na_str = low.isin(NA_STR_TOKENS).fillna(False)
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        dk_explicit = dk_num | dk_str
        na_explicit = na_num | na_str

        # Determine whether we have any explicit missing codes at all
        has_explicit_missing = bool(dk_explicit.any() or na_explicit.any())

        # Base missing mask (not valid and not explicitly coded as DK/NA)
        nan_mask = sn.isna() | (is_str and s.astype("string").str.strip().eq("").fillna(False))

        if has_explicit_missing:
            # If explicit codes exist, do not reassign NaN to DK/NA (keep them uncounted in M rows)
            dk_mask = dk_explicit
            na_mask = na_explicit
        else:
            # If no explicit missing codes, treat NaN/blank as DK for these music-attitude items (per CSV behavior)
            dk_mask = dk_explicit | (nan_mask & ~valid_mask)
            na_mask = na_explicit  # likely zero in this extract

        # Ensure disjointness and exclude valid
        dk_mask = dk_mask & ~valid_mask
        na_mask = na_mask & ~valid_mask & ~dk_mask

        return valid_mask, dk_mask, na_mask, sn

    # --- Build table (counts + mean) ---
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=object)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]

        valid_mask, dk_mask, na_mask, sn = classify_masks(s)

        # Counts: 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # Missing categories shown in Table 3
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        # Mean on valid 1..5 only
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --- Format (counts as integers; means to 2 decimals) ---
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )
        else:
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else str(int(x))
            )

    # --- Save a human-readable text file with clearly separated columns ---
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    colnames = list(formatted.columns)

    # Column widths to avoid merged headers (ensure clear separation for "New Age/Space" and "Opera")
    widths = {}
    for c in colnames:
        max_cell_len = int(formatted[c].astype(str).map(len).max())
        widths[c] = max(len(str(c)), max_cell_len) + 4

    row_w = max(len(str(idx)) for idx in formatted.index) + 2

    def _pad(text, width, align="left"):
        text = "" if text is None else str(text)
        if len(text) >= width:
            return text
        if align == "right":
            return " " * (width - len(text)) + text
        if align == "center":
            left = (width - len(text)) // 2
            right = width - len(text) - left
            return " " * left + text + " " * right
        return text + " " * (width - len(text))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        header = _pad("", row_w, "left") + "".join(_pad(c, widths[c], "center") for c in colnames)
        f.write(header + "\n")
        for idx in formatted.index:
            line = _pad(idx, row_w, "left")
            for c in colnames:
                val = formatted.loc[idx, c]
                line += _pad(val, widths[c], "center" if idx == "Mean" else "right")
            f.write(line + "\n")

    return formatted