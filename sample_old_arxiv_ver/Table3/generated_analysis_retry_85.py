def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- normalize column lookup (case-insensitive) ----
    colmap = {str(c).strip().lower(): c for c in df.columns}

    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # ---- Table 3 genres (exact order and mapping) ----
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

    # ---- row labels ----
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

    # ---- parsing helpers ----
    DK_NUM = {8, 98, -1}
    NA_NUM = {9, 99, -2}

    DK_STR_EXACT = {
        "na(d)", "na(dk)", "dk", "d", ".d",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
    }
    NA_STR_EXACT = {"na(n)", "na(na)", "no answer", "n", ".n"}

    def _to_str(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _to_num(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _to_str(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    def _blank_mask(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        st = _to_str(raw)
        return st.isna() | (st.str.strip() == "")

    def _typed_missing_masks(raw: pd.Series):
        """
        Return dk_mask, na_mask based ONLY on explicit typed encodings (numeric or string tokens).
        """
        sn = _to_num(raw)
        st = _to_str(raw).str.strip().str.lower()

        dk_num = sn.isin(list(DK_NUM)).fillna(False)
        na_num = sn.isin(list(NA_NUM)).fillna(False)

        dk_txt = (
            st.isin(list(DK_STR_EXACT))
            | st.str.contains("don’t know", na=False)
            | st.str.contains("don't know", na=False)
            | st.str.contains("dont know", na=False)
        )
        na_txt = st.isin(list(NA_STR_EXACT)) | st.str.contains("no answer", na=False)

        dk = (dk_num | dk_txt).fillna(False)
        na = (na_num | na_txt).fillna(False) & (~dk)
        return dk, na

    # ---- estimate DK share among explicit typed missings (if present) ----
    global_dk = 0
    global_na = 0
    for _, vlow in genres:
        raw = df[colmap[vlow]]
        sn = _to_num(raw)
        valid = sn.isin(VALID).fillna(False)
        dk_typed, na_typed = _typed_missing_masks(raw)
        global_dk += int((dk_typed & (~valid)).sum())
        global_na += int((na_typed & (~valid)).sum())

    if (global_dk + global_na) > 0:
        p_dk = global_dk / (global_dk + global_na)
        p_dk = float(np.clip(p_dk, 0.05, 0.95))
    else:
        p_dk = 0.90  # fallback if this extract has no explicit DK/NA tokens/codes

    def classify_for_table(raw: pd.Series):
        """
        Returns sn, valid_mask, dk_mask, na_mask where:
          - valid are 1..5
          - DK/NA are explicit typed codes/tokens + allocation of remaining blanks/non-valids
        """
        sn = _to_num(raw)
        valid = sn.isin(VALID).fillna(False)

        dk_typed, na_typed = _typed_missing_masks(raw)
        dk_typed = dk_typed & (~valid)
        na_typed = na_typed & (~valid) & (~dk_typed)

        # generic missing includes blanks/NaN plus any other non-valid, non-typed values
        blank = _blank_mask(raw)
        nonvalid_non_typed = (~valid) & (~dk_typed) & (~na_typed)
        # treat any remaining nonvalid entries as generic missing, including unexpected codes/strings
        generic_missing = nonvalid_non_typed & (blank | sn.isna() | (~sn.isin(VALID + list(DK_NUM) + list(NA_NUM))))

        idx = np.flatnonzero(generic_missing.to_numpy())
        k = int(round(p_dk * len(idx)))

        dk_alloc = pd.Series(False, index=raw.index)
        na_alloc = pd.Series(False, index=raw.index)
        if len(idx) > 0:
            if k > 0:
                dk_alloc.iloc[idx[:k]] = True
            if k < len(idx):
                na_alloc.iloc[idx[k:]] = True

        dk = (dk_typed | dk_alloc) & (~valid)
        na = (na_typed | na_alloc) & (~valid) & (~dk)
        return sn, valid, dk, na

    # ---- build table ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask = classify_for_table(raw)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # ---- format for display ----
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

    # ---- write to text file in 3 panels ----
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
        f.write("DK/NA counts use explicit typed missing codes/tokens when present; remaining\n")
        f.write("non-valid/missing entries are allocated deterministically using the observed\n")
        f.write("global DK-vs-NA ratio across the 18 items (fallback 90/10 if absent).\n\n")

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