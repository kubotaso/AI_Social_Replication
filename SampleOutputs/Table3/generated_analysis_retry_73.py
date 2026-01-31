def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --------------------
    # Restrict to GSS 1993
    # --------------------
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order and separate New Age/Space and Opera)
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
    # Rows (exact order)
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
    # Missing-category handling
    #
    # Requirement: compute DK/NA from raw data (no fabricated splitting).
    #
    # Strategy:
    # 1) Identify any explicit DK/NA numeric codes if present.
    # 2) Also identify explicit DK/NA text labels if present.
    # 3) If the file does not distinguish DK vs NA (e.g., both are blank/NaN),
    #    we still must produce DK and NA rows. In that case we fall back to
    #    GSS-style common codes if they appear (7/8/9, 97/98/99, -1/-2),
    #    otherwise treat:
    #       - DK/NA as missing by code if present,
    #       - blanks remain "undifferentiated missing" and are counted as "No answer"
    #         (a deterministic, non-splitting rule).
    # This avoids the prior runtime issues and avoids inventing a DK/NA split.
    # --------------------
    DK_NUM_CODES = {7, 8, 97, 98, -1}
    NA_NUM_CODES = {9, 99, -2}

    DK_TEXT_TOKENS = [
        "don't know", "dont know", "don’t know", "dk", "d/k", "dontknow",
        "don't know much", "dont know much", "don’t know much",
    ]
    NA_TEXT_TOKENS = [
        "no answer", "na", "n/a", "not ascertained", "not available",
        "refused", "refuse", "iap", "declined",
    ]

    def _as_text(raw: pd.Series) -> pd.Series:
        s = raw.astype("string")
        s = s.where(~s.isna(), other=pd.NA)
        return s.str.strip().str.lower()

    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        s = raw.astype("string")
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def _mask_any_token(text_series: pd.Series, tokens) -> pd.Series:
        mask = pd.Series(False, index=text_series.index)
        for t in tokens:
            mask = mask | text_series.str.contains(t, na=False)
        return mask

    def classify_item(raw: pd.Series):
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        # numeric-coded DK/NA
        dk_mask_num = sn.isin(list(DK_NUM_CODES)).fillna(False) & (~valid_mask)
        na_mask_num = sn.isin(list(NA_NUM_CODES)).fillna(False) & (~valid_mask) & (~dk_mask_num)

        # text-coded DK/NA (for non-valid and not already captured by numeric)
        st = _as_text(raw)
        dk_mask_txt = _mask_any_token(st, DK_TEXT_TOKENS) & (~valid_mask) & (~dk_mask_num) & (~na_mask_num)
        na_mask_txt = _mask_any_token(st, NA_TEXT_TOKENS) & (~valid_mask) & (~dk_mask_num) & (~na_mask_num) & (~dk_mask_txt)

        dk_mask = dk_mask_num | dk_mask_txt
        na_mask = na_mask_num | na_mask_txt

        # remaining missing that is not explicitly DK/NA: blanks/NaN or other non-valid codes
        # deterministic rule: count as "No answer" (not DK), because DK is a substantive missing type.
        nonvalid_nonclassified = (~valid_mask) & (~dk_mask) & (~na_mask)
        # Identify "blank-like" missing from the raw column
        if pd.api.types.is_numeric_dtype(raw):
            blank_like = raw.isna()
        else:
            st_raw = raw.astype("string")
            blank_like = st_raw.isna() | (st_raw.str.strip() == "")
        undifferentiated_missing = nonvalid_nonclassified & blank_like

        na_mask = na_mask | undifferentiated_missing

        # Any other strange codes (non-blank, non-valid, non DK/NA) are treated as missing but not shown.
        # (Table shows only DK and No answer among missing types.)
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
    # Format: counts as integers; mean to 2 decimals
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
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n")
        f.write("DK/NA counts are taken from explicit codes/labels when available; otherwise blanks are counted as 'No answer'.\n\n")

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