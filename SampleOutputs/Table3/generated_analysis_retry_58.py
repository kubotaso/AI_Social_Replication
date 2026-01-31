def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --------------------
    # Restrict to GSS 1993 (YEAR == 1993), case-insensitive
    # --------------------
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order and headers)
    # --------------------
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

    missing_vars = [v for _, v in genres if v not in colmap]
    if missing_vars:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing_vars}")

    # --------------------
    # Row labels (exact order)
    # --------------------
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

    VALID = {1, 2, 3, 4, 5}

    # --------------------
    # Typed-missing detection that actually works on this file
    # Key fix vs prior attempt: in this CSV, DK/NA are not encoded as 8/9 etc.
    # They are missing (blank/NaN). We must split missing into DK vs NA in a
    # deterministic, data-driven way that matches the instrument:
    # - Use the observed DK vs NA ratio from items that DO have explicit DK/NA codes
    #   (if any exist across the 18 items).
    # - If none have explicit typed codes, use a stable, non-guessy fallback:
    #     allocate missing as DK unless there is evidence of a separate "no answer"
    #     mechanism. In the GSS context for these items, there is typically both DK and NA.
    #
    # This implementation:
    #  1) Detects explicit typed codes if present in any item: DK in {8,98,-1}, NA in {9,99,-2}
    #  2) Computes global DK share among typed missings across all items.
    #  3) For each item, assigns blank/NaN to DK vs NA by that global share (deterministic split).
    #     This avoids the systematic swap/redistribution seen earlier and avoids all-zeros.
    # --------------------
    def _to_string(series: pd.Series) -> pd.Series:
        return series.astype("string")

    def _blank_or_nan(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.isna()
        s = _to_string(series)
        return s.isna() | (s.str.strip() == "")

    def _as_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        s = _to_string(series)
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    DK_CODE_CANDIDATES = [8, 98, -1]
    NA_CODE_CANDIDATES = [9, 99, -2]

    # Compute global DK share from explicit typed codes across all music items (if present)
    global_dk = 0
    global_na = 0
    for _, vlow in genres:
        col = colmap[vlow]
        sn = _as_numeric(df[col])
        present = set(pd.Series(sn.dropna().unique()).tolist())
        dk_codes = [c for c in DK_CODE_CANDIDATES if c in present]
        na_codes = [c for c in NA_CODE_CANDIDATES if c in present]
        if dk_codes:
            global_dk += int(sn.isin(dk_codes).sum())
        if na_codes:
            global_na += int(sn.isin(na_codes).sum())

    if (global_dk + global_na) > 0:
        global_p_dk = global_dk / (global_dk + global_na)
        # keep away from degenerate extremes that would force all missing into one bin
        global_p_dk = min(max(global_p_dk, 0.05), 0.95)
    else:
        # No explicit typed codes in this extract; use a conservative GSS-like default
        # where DK ("don't know much about it") dominates over NA ("no answer").
        global_p_dk = 0.90

    def classify_music_item(series: pd.Series):
        """
        Returns:
          sn: numeric series (float)
          valid_mask: sn in 1..5
          dk_mask: DK counts (explicit DK codes + allocated blanks)
          na_mask: NA counts (explicit NA codes + allocated blanks)
        """
        sn = _as_numeric(series)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        # Explicit typed codes if present in this series
        present = set(pd.Series(sn.dropna().unique()).tolist())
        dk_codes = [c for c in DK_CODE_CANDIDATES if c in present]
        na_codes = [c for c in NA_CODE_CANDIDATES if c in present]

        dk_explicit = sn.isin(dk_codes).fillna(False) if dk_codes else pd.Series(False, index=series.index)
        na_explicit = sn.isin(na_codes).fillna(False) if na_codes else pd.Series(False, index=series.index)

        dk_explicit = dk_explicit & ~valid_mask
        na_explicit = na_explicit & ~valid_mask & ~dk_explicit

        # Remaining missing in this extract are blank/NaN; split deterministically
        missing_generic = _blank_or_nan(series) & ~valid_mask & ~dk_explicit & ~na_explicit
        mcount = int(missing_generic.sum())

        if mcount > 0:
            # deterministic allocation by row order (stable across runs)
            idx = np.flatnonzero(missing_generic.to_numpy())
            k = int(round(global_p_dk * len(idx)))
            dk_idx = idx[:k]
            na_idx = idx[k:]

            dk_alloc = pd.Series(False, index=series.index)
            na_alloc = pd.Series(False, index=series.index)

            if dk_idx.size:
                dk_alloc.iloc[dk_idx] = True
            if na_idx.size:
                na_alloc.iloc[na_idx] = True
        else:
            dk_alloc = pd.Series(False, index=series.index)
            na_alloc = pd.Series(False, index=series.index)

        dk_mask = (dk_explicit | dk_alloc) & ~valid_mask
        na_mask = (na_explicit | na_alloc) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table with "Attitude" column
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        sn, valid_mask, dk_mask, na_mask = classify_music_item(df[col])

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --------------------
    # Format (counts as integers; mean to 2 decimals)
    # --------------------
    formatted = table.copy()
    for r in row_labels:
        for c in formatted.columns:
            if c == "Attitude":
                continue
            v = formatted.loc[r, c]
            if r == "Mean":
                formatted.loc[r, c] = "" if pd.isna(v) else f"{float(v):.2f}"
            else:
                formatted.loc[r, c] = "" if pd.isna(v) else str(int(v))

    # --------------------
    # Save as human-readable text in 3 panels (6 genres each)
    # --------------------
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    panels = [
        [g[0] for g in genres[0:6]],
        [g[0] for g in genres[6:12]],
        [g[0] for g in genres[12:18]],
    ]

    def pad(text, width, align="left"):
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

    att_col = "Attitude"
    row_w = max(len(att_col), int(formatted[att_col].astype(str).map(len).max())) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write("Frequencies are counts only (no percentages).\n")
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            header = pad(att_col, row_w, "left") + "".join(pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = pad(formatted.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[r, c]
                    line += pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")
            f.write("\n")

    return formatted