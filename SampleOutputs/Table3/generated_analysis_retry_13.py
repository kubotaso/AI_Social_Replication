def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("YEAR/year column not found in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---- Genre variables (Table 3 columns) ----
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

    # Resolve actual column names case-insensitively
    colmap = {str(c).strip().lower(): c for c in df.columns}
    missing = [var for _, var in genres if var not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # ---- Table 3 row labels (exact order) ----
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

    # ---- Missing code handling ----
    # In many GSS extracts, the music attitude items use:
    #  1..5 substantive
    #  8 = don't know (NA(d))
    #  9 = no answer (NA(n))
    # Some extracts may use other sentinel codes; we classify conservatively:
    #   - if value == 8 (or 98): DK
    #   - if value == 9 (or 99): NA
    # We do NOT reallocate NaN/blanks into DK/NA.
    DK_CODES = {8, 98}
    NA_CODES = {9, 99}

    def classify_music_item(series: pd.Series):
        sn = pd.to_numeric(series, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5])
        dk_mask = sn.isin(list(DK_CODES)) & ~valid_mask
        na_mask = sn.isin(list(NA_CODES)) & ~valid_mask & ~dk_mask
        return sn, valid_mask, dk_mask, na_mask

    # ---- Build table (counts + mean) ----
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

    # ---- Format for output ----
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

    # ---- Save a human-readable text file ----
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    colnames = list(formatted.columns)
    row_w = max(len("Attitude"), max(len(str(idx)) for idx in formatted.index)) + 2

    widths = {}
    for c in colnames:
        max_cell_len = int(formatted[c].astype(str).map(len).max())
        widths[c] = max(len(str(c)), max_cell_len) + 4

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

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        header = _pad("Attitude", row_w, "left") + "".join(_pad(c, widths[c], "center") for c in colnames)
        f.write(header + "\n")
        for idx in formatted.index:
            line = _pad(idx, row_w, "left")
            for c in colnames:
                val = formatted.loc[idx, c]
                line += _pad(val, widths[c], "center" if idx == "Mean" else "right")
            f.write(line + "\n")

    return formatted