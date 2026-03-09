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

    # --- genre variables (exact order) ---
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

    # --- rows (exact order) ---
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

    # --- helpers ---
    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        s = raw.astype("string")
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def _is_blank(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        s = raw.astype("string")
        return s.isna() | (s.str.strip() == "")

    # The GSS sometimes uses 8/9 or 98/99 for DK/NA; handle that if present.
    DK_CODES = {8, 98}
    NA_CODES = {9, 99}

    def classify_item(raw: pd.Series):
        """
        Returns:
          sn: numeric series
          valid_mask: sn in {1..5}
          dk_mask: DK (explicit codes and/or "don't know" strings)
          na_mask: No answer (explicit codes and/or "no answer"/refused strings)
          other_miss_mask: missing/unusable that is neither DK nor NA (kept out of displayed DK/NA)
        Note:
          We do NOT guess/split blanks into DK vs NA; blanks remain "other missing"
          unless explicitly coded as DK/NA.
        """
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        # numeric-coded DK/NA
        dk_mask = sn.isin(list(DK_CODES)).fillna(False) & (~valid_mask)
        na_mask = sn.isin(list(NA_CODES)).fillna(False) & (~valid_mask) & (~dk_mask)

        # text-coded DK/NA (rare, but safe)
        st = raw.astype("string").str.strip().str.lower()
        text_dk = st.str.contains("don't know", na=False) | st.str.contains("dont know", na=False) | st.str.contains("don’t know", na=False)
        text_na = (
            st.str.contains("no answer", na=False)
            | st.str.fullmatch(r"n/?a", na=False)
            | st.str.contains("refused", na=False)
        )

        dk_mask = dk_mask | (text_dk & (~valid_mask) & (~na_mask))
        na_mask = na_mask | (text_na & (~valid_mask) & (~dk_mask))

        # other missing/unusable (including blanks/NaN, and any other non-1..5 codes not DK/NA)
        blank_like = _is_blank(raw)
        other_nonvalid = (~valid_mask) & (~dk_mask) & (~na_mask)
        other_miss_mask = other_nonvalid & (blank_like | sn.notna())

        return sn, valid_mask, dk_mask, na_mask, other_miss_mask

    # --- build table (counts only; no percentages) ---
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask, _other_miss = classify_item(raw)

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

    # --- write text output (3 panels of 6 genres each) ---
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