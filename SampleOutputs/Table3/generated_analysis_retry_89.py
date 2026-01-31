def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Read data
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column access
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --------------------
    # Restrict to GSS 1993 (YEAR == 1993)
    # --------------------
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order/mapping)
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
    # Missing-code handling (robust, non-guessy)
    # - We count explicit DK/NA codes/tokens when present
    # - For truly blank/NaN entries, we classify them as "No answer" (NA(n))
    #   because DK is substantively a response option that is typically coded, not blank.
    # --------------------
    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _norm_text(s: pd.Series) -> pd.Series:
        st = _as_string(s)
        return st.str.strip().str.lower()

    def _as_numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _as_string(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    # Common GSS-style special codes seen across extracts
    DK_NUM = {8, 98, 998, -1}
    NA_NUM = {9, 99, 999, -2}

    DK_TOKENS = {
        "na(d)", "na (d)", ".d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
    }
    NA_TOKENS = {
        "na(n)", "na (n)", ".n", "no answer",
    }

    def _typed_missing_masks(raw: pd.Series):
        sn = _as_numeric(raw)
        st = _norm_text(raw)

        present = set(pd.Series(sn.dropna().unique()).astype(float).tolist())
        dk_codes = [c for c in DK_NUM if float(c) in present]
        na_codes = [c for c in NA_NUM if float(c) in present]

        dk_num = sn.isin(dk_codes).fillna(False) if dk_codes else pd.Series(False, index=raw.index)
        na_num = sn.isin(na_codes).fillna(False) if na_codes else pd.Series(False, index=raw.index)

        dk_txt = st.isin(DK_TOKENS) | st.str.contains("don’t know", na=False) | st.str.contains("don't know", na=False) | st.str.contains("dont know", na=False)
        na_txt = st.isin(NA_TOKENS) | st.str.contains("no answer", na=False)

        dk = (dk_num | dk_txt).fillna(False)
        na = (na_num | na_txt).fillna(False) & (~dk)
        return dk, na

    def _classify_item(raw: pd.Series):
        sn = _as_numeric(raw)
        valid = sn.isin(list(VALID)).fillna(False)

        dk_typed, na_typed = _typed_missing_masks(raw)

        dk = (dk_typed & (~valid)).fillna(False)
        na = (na_typed & (~valid) & (~dk)).fillna(False)

        st = _norm_text(raw)
        blank = st.isna() | (st == "")
        generic_missing = (blank | sn.isna()) & (~valid) & (~dk) & (~na)
        generic_missing = generic_missing.fillna(False)

        # Treat blank/NaN as "No answer" (instrument DK is a coded response, not a blank)
        na = (na | generic_missing) & (~valid) & (~dk)

        return sn, valid, dk, na

    # --------------------
    # Build output table (counts only; mean over 1-5)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask = _classify_item(raw)

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
    # Format for display: counts as integers; mean to 2 decimals
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
        f.write("DK/NA handling: counts explicit DK/NA codes/tokens when present; blank/NaN treated as 'No answer'.\n\n")

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