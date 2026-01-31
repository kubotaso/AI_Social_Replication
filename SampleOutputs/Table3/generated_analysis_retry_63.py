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
    # Restrict to GSS 1993
    # --------------------
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 variables (exact order + correct split for New Age/Space and Opera)
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
    # Helpers: parse numeric + determine DK/NA from actual codes in the data
    # (No imputation/splitting of blanks; blanks remain missing but not attributed to DK/NA.)
    # --------------------
    def _as_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        s = series.astype("string")
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def _pick_code_for_label(series_num: pd.Series, label_substrings) -> int | None:
        """
        Attempt to find the numeric code whose value label indicates DK or NA.
        Works if Series has pandas.Categorical with categories, or if Stata/SPSS
        imported labels are not present, returns None.
        """
        # pandas categorical: categories may carry labels, but numeric codes are in .cat.codes;
        # not reliable for typical CSV. So primarily attempt to use attributes if provided.
        # CSV usually loses labels -> fall back to common GSS conventions by inspecting values.
        return None

    # Common GSS missing-type numeric codes for attitude items in many extracts
    DK_CANDIDATES = [8, 98, 998, -1, -9]
    NA_CANDIDATES = [9, 99, 999, -2, -8]
    OTHER_MISSING = [0, 97, -3, -4, -5, -6, -7]

    def classify(series_raw: pd.Series):
        sn = _as_numeric(series_raw)

        valid_mask = sn.isin(VALID)

        present = set(pd.Series(sn.dropna().unique()).tolist())

        dk_codes = [c for c in DK_CANDIDATES if c in present]
        na_codes = [c for c in NA_CANDIDATES if c in present]
        other_codes = [c for c in OTHER_MISSING if c in present]

        dk_mask = sn.isin(dk_codes) if dk_codes else pd.Series(False, index=sn.index)
        na_mask = sn.isin(na_codes) if na_codes else pd.Series(False, index=sn.index)
        other_mask = sn.isin(other_codes) if other_codes else pd.Series(False, index=sn.index)

        # Ensure exclusivity and exclude any accidental overlap with valid
        dk_mask = dk_mask & ~valid_mask
        na_mask = na_mask & ~valid_mask & ~dk_mask
        other_mask = other_mask & ~valid_mask & ~dk_mask & ~na_mask

        return sn, valid_mask, dk_mask, na_mask, other_mask

    # --------------------
    # Build table (counts only; mean over 1..5)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        raw = df[colmap[var_lower]]
        sn, valid_mask, dk_mask, na_mask, other_mask = classify(raw)

        # 1..5 counts
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum(skipna=True))
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum(skipna=True))
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum(skipna=True))
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum(skipna=True))
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum(skipna=True))

        # Missing-type counts shown in the table: DK and NA only, based on actual codes
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --------------------
    # Format: counts as ints; mean to 2 decimals
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
        f.write("DK/NA counts are computed only when corresponding numeric codes are present in the data.\n\n")

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