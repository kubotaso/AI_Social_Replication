def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---- Table 3 genre variables (exact set/order) ----
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

    # resolve columns case-insensitively
    missing = [v for _, v in genres if v not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # ---- Row labels (exact order) ----
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

    # ---- Missing code handling (GSS-style) ----
    # IMPORTANT FIX:
    # In this extracted CSV, DK/NA are commonly represented as NaN (blank) rather than 8/9.
    # To match Table 3, count:
    #   DK = explicit DK codes (8,98,...) + NaN/blanks
    #   NA = explicit no-answer codes (9,99,...) only
    # This aligns with the observed pattern: DK is large; NA is small but nonzero.
    DK_CODES = {8, 98, -1}
    NA_CODES = {9, 99, -2}

    def _blank_mask(s: pd.Series) -> pd.Series:
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            st = s.astype("string")
            return st.isna() | (st.str.strip() == "")
        return pd.Series(False, index=s.index)

    def classify_music_item(series: pd.Series):
        s = series
        sn = pd.to_numeric(s, errors="coerce")

        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        # explicit numeric DK/NA codes
        dk_mask = sn.isin(list(DK_CODES)).fillna(False) & ~valid_mask
        na_mask = sn.isin(list(NA_CODES)).fillna(False) & ~valid_mask & ~dk_mask

        # Treat NaN/blank as DK for these items (the CSV frequently stores DK as blank)
        nan_or_blank = (sn.isna() | _blank_mask(s)).fillna(False)
        dk_mask = (dk_mask | (nan_or_blank & ~valid_mask & ~na_mask))

        return sn, valid_mask, dk_mask, na_mask

    # ---- Build the table ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=float)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]

        sn, valid_mask, dk_mask, na_mask = classify_music_item(s)

        table.loc["(1) Like very much", genre_label] = float((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = float((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = float((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = float((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = float((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = float(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = float(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = float(mean_val) if pd.notna(mean_val) else np.nan

    # ---- Format for output/return ----
    formatted = pd.DataFrame(index=row_labels, columns=["Attitude"] + [g[0] for g in genres], dtype=object)
    formatted["Attitude"] = formatted.index

    for genre_label, _ in genres:
        col = genre_label
        out = []
        for r in row_labels:
            v = table.loc[r, col]
            if r == "Mean":
                out.append("" if pd.isna(v) else f"{float(v):.2f}")
            else:
                out.append("" if pd.isna(v) else str(int(round(float(v)))))
        formatted[col] = out

    # ---- Save as human-readable text file with 3 panels (6 genres each) ----
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    panels = [
        [g[0] for g in genres[0:6]],
        [g[0] for g in genres[6:12]],
        [g[0] for g in genres[12:18]],
    ]

    def _pad(text, width, align="left"):
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

    attitude_col = "Attitude"
    row_w = max(len(attitude_col), max(len(str(x)) for x in formatted[attitude_col])) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(c), max_cell_len) + 4

            header = _pad(attitude_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for i in range(len(formatted)):
                att = formatted.iloc[i][attitude_col]
                line = _pad(att, row_w, "left")
                for c in panel_cols:
                    val = formatted.iloc[i][c]
                    line += _pad(val, widths[c], "center" if att == "Mean" else "right")
                f.write(line + "\n")

    return formatted