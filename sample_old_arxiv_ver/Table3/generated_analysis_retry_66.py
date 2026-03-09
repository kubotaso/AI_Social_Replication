def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- restrict to 1993 ---
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --- genre variables (exact order/labels) ---
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

    # --- table structure ---
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

    # --- robust DK/NA decoding ---
    # GSS-style special codes are often high positives or negatives; extracts may vary.
    # We detect them from value labels if available; otherwise from common code sets.
    DK_NUM_CODES_DEFAULT = {8, 98, 998, 9998, -1, -8, -9}
    NA_NUM_CODES_DEFAULT = {9, 99, 999, 9999, -2}

    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        sr = _as_string(raw)
        sr = sr.where(sr.str.strip() != "", other=pd.NA)
        return pd.to_numeric(sr, errors="coerce")

    def _blank_mask(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        sr = _as_string(raw)
        return sr.isna() | (sr.str.strip() == "")

    def _get_label_maps(series: pd.Series):
        """
        Try to infer DK/NA codes from pandas categorical/value labels if present.
        Returns (dk_codes, na_codes) as sets.
        """
        dk_codes, na_codes = set(), set()

        # pandas Categorical
        if pd.api.types.is_categorical_dtype(series):
            # If categorical with categories that are numeric-like labels, not reliable.
            # But if categories are numeric and there is a mapping elsewhere, cannot access.
            return dk_codes, na_codes

        # If it's an Int/Float with metadata, pandas doesn't store labels. So return empty.
        return dk_codes, na_codes

    def classify_item(raw: pd.Series):
        sn = _as_numeric(raw)

        valid_mask = sn.isin(VALID).fillna(False)

        # explicit DK/NA by string tokens (rare in numeric extracts, but safe)
        dk_str = pd.Series(False, index=raw.index)
        na_str = pd.Series(False, index=raw.index)
        if not pd.api.types.is_numeric_dtype(raw):
            s = _as_string(raw).fillna("").str.strip().str.lower()
            dk_str = s.str.contains("don't know", regex=False) | s.str.contains("dont know", regex=False) | s.eq("dk") | s.eq("d/k")
            na_str = s.str.contains("no answer", regex=False) | s.eq("na") | s.eq("n/a")

        # infer numeric DK/NA codes (use defaults plus any inferred)
        dk_codes_inf, na_codes_inf = _get_label_maps(raw)
        dk_codes = set(DK_NUM_CODES_DEFAULT) | set(dk_codes_inf)
        na_codes = set(NA_NUM_CODES_DEFAULT) | set(na_codes_inf)

        dk_num = sn.isin(list(dk_codes)).fillna(False)
        na_num = sn.isin(list(na_codes)).fillna(False)

        dk_explicit = (dk_num | dk_str) & ~valid_mask
        na_explicit = (na_num | na_str) & ~valid_mask & ~dk_explicit

        # Remaining missing: blanks/NaN that are not valid and not explicit DK/NA.
        # We cannot invent DK vs NA from blanks; however, the GSS extract typically uses
        # blanks for "don't know much about it" and reserves very small NA. We can
        # distinguish using within-item pattern: if an item has ANY explicit NA codes,
        # treat blanks as DK; else treat blanks as DK by default (instrument-specific).
        # This avoids pushing all blanks into NA.
        generic_missing = _blank_mask(raw) & ~valid_mask & ~dk_explicit & ~na_explicit

        has_any_explicit_na = bool(na_explicit.any())
        has_any_explicit_dk = bool(dk_explicit.any())

        # If there are explicit DK codes, blanks are almost certainly DK too.
        # If there are explicit NA codes but no DK codes, blanks are still more consistent with DK for these items.
        # So default blanks -> DK for these music items.
        dk_from_blank = generic_missing
        na_from_blank = pd.Series(False, index=raw.index)

        dk_mask = dk_explicit | dk_from_blank
        na_mask = na_explicit | na_from_blank

        # Ensure disjointness
        na_mask = na_mask & ~dk_mask & ~valid_mask
        dk_mask = dk_mask & ~valid_mask

        return sn, valid_mask, dk_mask, na_mask

    # --- build table (counts only; mean over valid 1..5) ---
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

    # --- format: counts as ints; mean to 2 decimals ---
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

    # --- save as text in 3 panels (6 genres each) ---
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