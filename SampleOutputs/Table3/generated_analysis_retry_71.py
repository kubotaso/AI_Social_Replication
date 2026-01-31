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
    # Table 3 genre variables (exact order and distinct labels)
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

    VALID = [1, 2, 3, 4, 5]

    # --------------------
    # Helpers: robust missing code handling for GSS extracts
    # Strategy:
    # - Treat 1..5 as valid.
    # - Detect explicit DK/NA codes if present (common: 8/9, 98/99, -1/-2).
    # - Otherwise, if value labels are present, use them.
    # - Otherwise, treat all non-1..5 as missing; split into DK vs NA only when
    #   explicit codes are found. (No fabricated split.)
    # This avoids incorrect DK/NA distributions.
    # --------------------
    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        s = raw.astype("string")
        s = s.where(s.str.strip() != "", other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    # Candidate sets (observed in different GSS exports)
    DK_CANDIDATES = {8, 98, -1}
    NA_CANDIDATES = {9, 99, -2}

    # Discover which of these actually appear in the data (pooled across items)
    pooled_codes = set()
    for _, vlow in genres:
        sn = _as_numeric(df[colmap[vlow]])
        u = set(pd.Series(sn.dropna().unique()).astype(int).tolist())
        pooled_codes |= u

    DK_CODES = sorted(list(DK_CANDIDATES & pooled_codes))
    NA_CODES = sorted(list(NA_CANDIDATES & pooled_codes))

    def classify_item(raw: pd.Series):
        sn = _as_numeric(raw)
        valid_mask = sn.isin(VALID).fillna(False)

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        if DK_CODES:
            dk_mask = sn.isin(DK_CODES).fillna(False) & (~valid_mask)
        if NA_CODES:
            na_mask = sn.isin(NA_CODES).fillna(False) & (~valid_mask) & (~dk_mask)

        # Any remaining nonvalid values (including blanks/NaN) are missing but not split
        # into DK vs NA (to avoid inventing categories).
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
    # Ensure headers never concatenate (the prior "New Age/Space Opera" issue).
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
        f.write("Mean computed over valid responses 1–5 only; missing excluded.\n")
        if DK_CODES or NA_CODES:
            f.write(f"Detected DK code(s): {DK_CODES if DK_CODES else 'none'}; NA code(s): {NA_CODES if NA_CODES else 'none'}.\n")
        else:
            f.write("No explicit DK/NA codes detected in this CSV extract; DK/NA rows count only explicit codes.\n")
        f.write("\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            # Write header with explicit separators to avoid accidental concatenation
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