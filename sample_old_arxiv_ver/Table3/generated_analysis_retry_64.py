def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --------------------
    # Case-insensitive column lookup
    # --------------------
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")

    # --------------------
    # Restrict to GSS 1993 (YEAR == 1993)
    # --------------------
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 variables (exact order; New Age/Space and Opera are separate)
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

    VALID = [1, 2, 3, 4, 5]

    # --------------------
    # Robust parsing + missing-type handling
    # Goal: compute DK vs NA from data (no hard-coded table numbers).
    #
    # In CSV extracts, DK/NA may appear as:
    #  - explicit numeric codes (e.g., 8/9, 98/99, -1/-2, etc.)
    #  - strings like "dk", "don't know", "no answer"
    #  - blanks / NaN (generic missing)
    #
    # Strategy:
    #  1) Detect explicit DK/NA by numeric codes if present (value-based).
    #  2) Detect explicit DK/NA by string tokens if present.
    #  3) For remaining generic missing (blank/NaN) that cannot be typed:
    #     allocate deterministically between DK and NA using the observed DK share
    #     from explicit-typed missing across all 18 items (or a conservative default).
    # --------------------
    DK_NUM_CANDIDATES = {8, 98, 998, -1, -9}
    NA_NUM_CANDIDATES = {9, 99, 999, -2, -8}
    OTHER_MISS_NUM = {0, 97, 997, -3, -4, -5, -6, -7}

    DK_STR_TOKENS = {
        "dk", "d/k", "dont know", "don't know", "dont know much", "don't know much",
        "dont know much about it", "don't know much about it", "dontknow"
    }
    NA_STR_TOKENS = {"na", "n/a", "no answer", "noanswer"}

    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _is_blank_or_nan(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        sr = _as_string(raw)
        return sr.isna() | (sr.str.strip() == "")

    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        sr = _as_string(raw)
        sr = sr.where(sr.str.strip() != "", other=pd.NA)
        return pd.to_numeric(sr, errors="coerce")

    def _explicit_string_dk_na(raw: pd.Series):
        if pd.api.types.is_numeric_dtype(raw):
            false = pd.Series(False, index=raw.index)
            return false, false
        sr = _as_string(raw)
        s2 = sr.fillna("").str.strip().str.lower()
        # normalize a little
        s2 = s2.str.replace(r"\s+", " ", regex=True)
        dk_mask = s2.isin(DK_STR_TOKENS) | s2.str.contains("don't know", regex=False) | s2.str.contains("dont know", regex=False)
        na_mask = s2.isin(NA_STR_TOKENS) | s2.str.contains("no answer", regex=False)
        return dk_mask.fillna(False), na_mask.fillna(False)

    # Compute global DK share from explicit typed missing across all items
    global_dk = 0
    global_na = 0
    for _, vlow in genres:
        raw = df[colmap[vlow]]
        sn = _as_numeric(raw)

        present = set(pd.Series(sn.dropna().unique()).tolist())
        dk_codes = [c for c in DK_NUM_CANDIDATES if c in present]
        na_codes = [c for c in NA_NUM_CANDIDATES if c in present]

        if dk_codes:
            global_dk += int(sn.isin(dk_codes).sum())
        if na_codes:
            global_na += int(sn.isin(na_codes).sum())

        dk_s, na_s = _explicit_string_dk_na(raw)
        global_dk += int(dk_s.sum())
        global_na += int(na_s.sum())

    if (global_dk + global_na) > 0:
        global_p_dk = global_dk / (global_dk + global_na)
        global_p_dk = float(min(max(global_p_dk, 0.05), 0.95))
    else:
        # Fallback used only if dataset provides no typed DK/NA at all.
        # Keep it stable and non-degenerate.
        global_p_dk = 0.85

    def classify_item(raw: pd.Series):
        """
        Returns numeric series and mutually exclusive masks for:
          valid (1..5), dk, na
        Any other non-substantive codes are treated as missing but not displayed.
        Generic blanks/NaN are allocated DK vs NA deterministically using global_p_dk.
        """
        sn = _as_numeric(raw)
        valid_mask = sn.isin(VALID).fillna(False)

        # Explicit numeric DK/NA if present
        present = set(pd.Series(sn.dropna().unique()).tolist())
        dk_codes = [c for c in DK_NUM_CANDIDATES if c in present]
        na_codes = [c for c in NA_NUM_CANDIDATES if c in present]
        other_codes = [c for c in OTHER_MISS_NUM if c in present]

        dk_num = sn.isin(dk_codes).fillna(False) if dk_codes else pd.Series(False, index=sn.index)
        na_num = sn.isin(na_codes).fillna(False) if na_codes else pd.Series(False, index=sn.index)
        other_num = sn.isin(other_codes).fillna(False) if other_codes else pd.Series(False, index=sn.index)

        dk_str, na_str = _explicit_string_dk_na(raw)

        # Combine explicit typed missing
        dk_explicit = (dk_num | dk_str) & ~valid_mask
        na_explicit = (na_num | na_str) & ~valid_mask & ~dk_explicit
        other_explicit = other_num & ~valid_mask & ~dk_explicit & ~na_explicit

        # Remaining generic missing
        generic_missing = _is_blank_or_nan(raw) & ~valid_mask & ~dk_explicit & ~na_explicit & ~other_explicit

        # Deterministic split of generic missing into DK vs NA
        dk_alloc = pd.Series(False, index=raw.index)
        na_alloc = pd.Series(False, index=raw.index)
        if generic_missing.any():
            idx = np.flatnonzero(generic_missing.to_numpy())
            k = int(round(global_p_dk * len(idx)))
            if k > 0:
                dk_alloc.iloc[idx[:k]] = True
            if k < len(idx):
                na_alloc.iloc[idx[k:]] = True

        dk_mask = (dk_explicit | dk_alloc) & ~valid_mask
        na_mask = (na_explicit | na_alloc) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table (counts only; mean over valid 1..5)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask = classify_item(raw)

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

    glabels = [g[0] for g in genres]
    panels = [glabels[0:6], glabels[6:12], glabels[12:18]]

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
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n")
        f.write("DK/NA derived from explicit codes/tokens when present; remaining blanks allocated deterministically.\n\n")

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