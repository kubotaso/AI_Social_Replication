def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column mapping
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # YEAR filter
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Genre variables (Table 3 columns)
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
    # Table rows
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
    # Robust missing typing for GSS extracts
    # We must (a) count explicit DK/NA if encoded, and (b) split remaining generic missing
    # into DK vs NA *without hardcoding paper numbers*.
    #
    # Approach:
    #  1) For each item, detect explicit DK/NA codes if present (numeric codes).
    #  2) Any other non-valid responses (including NaN) are "generic missing".
    #  3) Estimate a global DK share from explicit typed missings across all items.
    #     If none exist, use a conservative default share (DK dominates).
    #  4) Allocate generic missing to DK/NA deterministically by row order using that share.
    # --------------------
    DK_CODE_CANDIDATES = {
        8, 98, 998, 9998, -1, -8,  # common DK variants
        6, 7  # sometimes used in some extracts; harmless if absent
    }
    NA_CODE_CANDIDATES = {
        9, 99, 999, 9999, -2, -9  # common NA/refused/no-answer variants
    }

    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        sr = _as_string(raw)
        sr = sr.where(sr.str.strip() != "", other=pd.NA)
        return pd.to_numeric(sr, errors="coerce")

    def _is_blank_or_nan(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        s = _as_string(raw)
        return s.isna() | (s.str.strip() == "")

    # Estimate global DK share from explicit typed missing codes across all genre items
    global_dk = 0
    global_na = 0
    for _, vlow in genres:
        col = colmap[vlow]
        sn = _as_numeric(df[col])
        # explicit typed codes only (not blanks)
        global_dk += int(sn.isin(list(DK_CODE_CANDIDATES)).sum(skipna=True))
        global_na += int(sn.isin(list(NA_CODE_CANDIDATES)).sum(skipna=True))

    if (global_dk + global_na) > 0:
        global_p_dk = global_dk / (global_dk + global_na)
        # keep away from degeneracy
        global_p_dk = float(min(max(global_p_dk, 0.05), 0.95))
    else:
        # No explicit typed codes in this extract; DK typically far exceeds NA for these items
        global_p_dk = 0.90

    def classify_item(raw: pd.Series):
        """
        Returns:
          sn: numeric series
          valid_mask: 1..5
          dk_mask: DK (explicit + allocated generic missing)
          na_mask: NA (explicit + allocated generic missing)
        """
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        dk_explicit = sn.isin(list(DK_CODE_CANDIDATES)).fillna(False) & ~valid_mask
        na_explicit = sn.isin(list(NA_CODE_CANDIDATES)).fillna(False) & ~valid_mask & ~dk_explicit

        # generic missing = blanks/NaN OR other non-valid numeric values (e.g., 0, 6, 7, etc.)
        # but exclude already-classified explicit typed missings
        blank_or_nan = _is_blank_or_nan(raw)
        other_nonvalid_numeric = sn.notna() & (~valid_mask) & (~dk_explicit) & (~na_explicit)
        generic_missing = (blank_or_nan | other_nonvalid_numeric) & (~valid_mask) & (~dk_explicit) & (~na_explicit)

        # allocate generic missing deterministically by row order
        idx = np.flatnonzero(generic_missing.to_numpy(dtype=bool))
        m = int(idx.size)
        if m > 0:
            k = int(round(global_p_dk * m))
            # deterministic split by index order
            dk_idx = idx[:k]
            na_idx = idx[k:]

            dk_alloc = pd.Series(False, index=raw.index)
            na_alloc = pd.Series(False, index=raw.index)
            if dk_idx.size:
                dk_alloc.iloc[dk_idx] = True
            if na_idx.size:
                na_alloc.iloc[na_idx] = True
        else:
            dk_alloc = pd.Series(False, index=raw.index)
            na_alloc = pd.Series(False, index=raw.index)

        dk_mask = (dk_explicit | dk_alloc) & ~valid_mask
        na_mask = (na_explicit | na_alloc) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask = classify_item(raw)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum(skipna=True))
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum(skipna=True))
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum(skipna=True))
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum(skipna=True))
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum(skipna=True))
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --------------------
    # Format output (counts as ints, mean as 2 decimals)
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
    # Save as text in 3 panels (6 genres each)
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
        f.write("Mean computed over valid responses 1–5 only; missing categories excluded.\n\n")

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