def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 (YEAR == 1993), case-insensitive column handling ---
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --- Table 3: genres in exact order (18 columns) ---
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

    # --- Rows (exact labels/order) ---
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

    # Common GSS-style missing codes (numeric extracts) + string tokens
    DK_NUM_CODES = {8, 98, -1}
    NA_NUM_CODES = {9, 99, -2}

    DK_STR_TOKENS = {
        "na(d)", "na(dk)", "dk", "d", ".d",
        "don't know", "dont know", "don’t know",
        "don't know much about it", "dont know much about it", "don’t know much about it",
    }
    NA_STR_TOKENS = {"na(n)", "na(na)", "no answer", "n", ".n"}

    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _as_string(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    def _blank_or_nan(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        st = _as_string(raw)
        return st.isna() | (st.str.strip() == "")

    def classify_item(raw: pd.Series):
        """
        Returns:
          sn: numeric series (float)
          valid_mask: 1..5
          dk_mask: DK (explicit codes/strings OR, if no explicit typed codes exist for this item,
                   treat remaining missing as DK by default)
          na_mask: NA (explicit codes/strings only)
        Rationale:
          The provided CSV extract often stores DK/NA as blank. Table 3 requires DK and NA rows.
          In this extract, "No answer" is typically explicitly coded (if present at all), while
          DK is more often collapsed into blanks. To avoid inventing NA, we:
            - Count explicit NA only
            - Count explicit DK plus (if item lacks explicit DK/NA encodings) all remaining blanks as DK
          This matches the table structure without forcing arbitrary DK/NA splits.
        """
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        st = _as_string(raw).str.strip().str.lower()

        # explicit numeric codes
        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False) & (~valid_mask)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False) & (~valid_mask) & (~dk_num)

        # explicit string codes/phrases
        dk_text = (
            st.isin(list(DK_STR_TOKENS))
            | st.str.contains("don’t know", na=False)
            | st.str.contains("don't know", na=False)
            | st.str.contains("dont know", na=False)
        ) & (~valid_mask)

        na_text = (
            st.isin(list(NA_STR_TOKENS))
            | st.str.contains("no answer", na=False)
        ) & (~valid_mask)

        dk_explicit = (dk_num | dk_text) & (~valid_mask)
        na_explicit = (na_num | na_text) & (~valid_mask) & (~dk_explicit)

        # remaining missing (blank/NaN) not otherwise classified
        generic_missing = _blank_or_nan(raw) & (~valid_mask) & (~dk_explicit) & (~na_explicit)

        # If this item shows no explicit DK/NA encodings at all, treat generic missing as DK.
        has_any_typed = bool(dk_explicit.any() or na_explicit.any() or dk_num.any() or na_num.any() or dk_text.any() or na_text.any())
        if has_any_typed:
            dk_mask = dk_explicit
            na_mask = na_explicit
        else:
            dk_mask = dk_explicit | generic_missing
            na_mask = na_explicit  # do not invent NA

        return sn, valid_mask, dk_mask, na_mask

    # --- Build summary table ---
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

    # --- Format: counts as integers, mean to 2 decimals ---
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

    # --- Save as text in 3 panels (6 genres each) ---
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
        f.write("DK/NA counted from explicit codes/strings; if an item has no typed missing encodings,\n")
        f.write("blank/NaN are treated as 'Don’t know much about it' to match Table 3 structure.\n\n")

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