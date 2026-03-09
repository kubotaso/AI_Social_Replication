def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---------- Restrict to GSS 1993 ----------
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected a YEAR/year column in the dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---------- Genre variables (Table 3 columns) ----------
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

    colmap = {str(c).strip().lower(): c for c in df.columns}
    missing = [v for _, v in genres if v not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # ---------- Table rows (exact order) ----------
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

    # ---------- Missing code handling ----------
    # Standard GSS-style: 8=don't know, 9=no answer. Also tolerate 98/99.
    # IMPORTANT: Do NOT reallocate NaN/blanks into DK/NA; keep them missing (excluded from table M rows).
    DK_CODES = {8, 98}
    NA_CODES = {9, 99}

    def classify_music_item(series: pd.Series):
        # numeric coercion; non-numeric becomes NaN (not counted in DK/NA unless explicitly coded)
        sn = pd.to_numeric(series, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5])
        dk_mask = sn.isin(list(DK_CODES)) & ~valid_mask
        na_mask = sn.isin(list(NA_CODES)) & ~valid_mask & ~dk_mask
        return sn, valid_mask, dk_mask, na_mask

    # ---------- Build table (counts + mean) ----------
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=object)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]

        sn, valid_mask, dk_mask, na_mask = classify_music_item(s)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # ---------- Format for output (strings) ----------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )
        else:
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else str(int(x))
            )

    # ---------- Save as human-readable text with panels (3 panels of 6 genres) ----------
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    panels = [
        [g[0] for g in genres[0:6]],
        [g[0] for g in genres[6:12]],
        [g[0] for g in genres[12:18]],
    ]

    def _panel_widths(cols):
        widths = {}
        for c in cols:
            max_cell_len = int(formatted[c].astype(str).map(len).max())
            widths[c] = max(len(str(c)), max_cell_len) + 4
        return widths

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

    row_w = max(len("Attitude"), max(len(str(idx)) for idx in formatted.index)) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            widths = _panel_widths(panel_cols)

            f.write("\n")
            f.write(f"Panel {p_idx}\n")
            header = _pad("Attitude", row_w, "left") + "".join(
                _pad(c, widths[c], "center") for c in panel_cols
            )
            f.write(header + "\n")

            for idx in formatted.index:
                line = _pad(idx, row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[idx, c]
                    line += _pad(val, widths[c], "center" if idx == "Mean" else "right")
                f.write(line + "\n")

    return formatted