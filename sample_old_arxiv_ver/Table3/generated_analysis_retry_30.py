def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---------- Resolve columns case-insensitively ----------
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # ---------- Restrict to GSS 1993 ----------
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---------- Table 3 genre variables (exact order) ----------
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

    # ---------- Row labels (exact order) ----------
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

    # ---------- Missing-code handling ----------
    # In this provided CSV, the two missing categories are not preserved as distinct codes
    # (they arrive as NaN), so we must use the published-code convention used for these
    # music items in GSS:
    #   1-5 = substantive
    #   8   = Don't know much about it (NA(d))
    #   9   = No answer (NA(n))
    #
    # If the extract uses 98/99 (or other common variants), we honor those too.
    VALID = {1, 2, 3, 4, 5}
    DK_CODES = {8, 98}
    NA_CODES = {9, 99}

    def _counts_for_item(series: pd.Series):
        s_num = pd.to_numeric(series, errors="coerce")

        c = {k: int((s_num == k).sum()) for k in [1, 2, 3, 4, 5]}
        dk = int(s_num.isin(list(DK_CODES)).sum())
        na = int(s_num.isin(list(NA_CODES)).sum())

        # If there are string tokens (unlikely here), count them as well without reallocating NaNs
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            low = series.astype("string").str.strip().str.lower()
            numeric_nan = s_num.isna()
            dk_tokens = {
                "d", "dk",
                "dont know", "don't know", "don’t know",
                "dont know much", "don't know much", "don’t know much",
                "dont know much about it", "don't know much about it", "don’t know much about it",
                "dont know enough", "don't know enough", "don’t know enough",
                "dont know enough about it", "don't know enough about it", "don’t know enough about it",
            }
            na_tokens = {"n", "na", "no answer", "noanswer"}
            dk += int((numeric_nan & low.isin(dk_tokens)).sum())
            na += int((numeric_nan & low.isin(na_tokens)).sum())

        mean_val = s_num.where(s_num.isin(list(VALID))).mean()
        return c[1], c[2], c[3], c[4], c[5], dk, na, mean_val

    # ---------- Build table with Attitude as first column ----------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk, na, mean_val = _counts_for_item(s)

        table.loc["(1) Like very much", genre_label] = c1
        table.loc["(2) Like it", genre_label] = c2
        table.loc["(3) Mixed feelings", genre_label] = c3
        table.loc["(4) Dislike it", genre_label] = c4
        table.loc["(5) Dislike very much", genre_label] = c5
        table.loc["(M) Don’t know much about it", genre_label] = dk
        table.loc["(M) No answer", genre_label] = na
        table.loc["Mean", genre_label] = mean_val

    # ---------- Format (counts as ints; mean as 2 decimals) ----------
    formatted = table.copy()
    for r in row_labels:
        if r == "Mean":
            for c in formatted.columns:
                if c == "Attitude":
                    continue
                v = formatted.loc[r, c]
                formatted.loc[r, c] = "" if pd.isna(v) else f"{float(v):.2f}"
        else:
            for c in formatted.columns:
                if c == "Attitude":
                    continue
                v = formatted.loc[r, c]
                formatted.loc[r, c] = "" if pd.isna(v) else str(int(v))

    # ---------- Save as human-readable text file with 3 panels (6 genres each) ----------
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

    att_col = "Attitude"
    row_w = max(len(att_col), max(len(str(x)) for x in formatted[att_col].tolist())) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("Counts shown for 1–5 plus (M) Don’t know much about it and (M) No answer. Mean computed over 1–5 only.\n")
        f.write("Missing categories are counted using standard GSS music-item codes (DK=8/98, NA=9/99); plain NaN cannot be split.\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            header = _pad(att_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = _pad(formatted.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[r, c]
                    line += _pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")

    return formatted