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
    # Helper parsing
    # --------------------
    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        sr = _as_string(raw)
        sr = sr.where(sr.str.strip() != "", other=pd.NA)
        return pd.to_numeric(sr, errors="coerce")

    # --------------------
    # DK/NA decoding
    # Goal: count DK vs NA without guessing.
    #
    # Strategy:
    # 1) Detect explicit typed missing codes if present (common GSS conventions).
    # 2) If explicit codes not present, use the fact that this extract often encodes
    #    nonresponse as NA (blank) but the "No answer" category is not directly observable.
    #    In that case, we CANNOT infer DK vs NA from blanks alone.
    #
    # To remain faithful and non-fabricating:
    # - We count explicit DK codes as DK and explicit NA codes as NA.
    # - Remaining non-valid, non-coded values are treated as DK if (and only if) they match
    #   a DK string token; otherwise they are treated as system missing and reported as DK
    #   for these items only if the dataset provides a separate "na" mechanism.
    #
    # Practically, for this file the DK/NA are encoded as numeric codes (not blanks),
    # so the explicit-code path should be used and will reproduce the table.
    # --------------------
    DK_CODES = {8, 98, 998, 9998, -1, -8}   # "don't know" style codes seen in GSS extracts
    NA_CODES = {9, 99, 999, 9999, -2, -9}   # "no answer/refused" style codes

    def classify_item(raw: pd.Series):
        sn = _as_numeric(raw)

        valid_mask = sn.isin(list(VALID)).fillna(False)

        # string tokens (safety for non-numeric extracts)
        dk_str = pd.Series(False, index=raw.index)
        na_str = pd.Series(False, index=raw.index)
        if not pd.api.types.is_numeric_dtype(raw):
            s = _as_string(raw).fillna("").str.strip().str.lower()
            dk_str = (
                s.eq("dk")
                | s.eq("d/k")
                | s.str.contains("dont know", regex=False)
                | s.str.contains("don't know", regex=False)
            )
            na_str = s.eq("na") | s.eq("n/a") | s.str.contains("no answer", regex=False)

        dk_num = sn.isin(list(DK_CODES)).fillna(False)
        na_num = sn.isin(list(NA_CODES)).fillna(False)

        dk_mask = (dk_num | dk_str) & ~valid_mask
        na_mask = (na_num | na_str) & ~valid_mask & ~dk_mask

        # Any remaining non-valid entries not captured above are treated as missing-but-uncategorized.
        # Table 3 only reports DK and No answer; we do not reallocate uncategorized missing between them.
        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        raw = df[colmap[var_lower]]
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