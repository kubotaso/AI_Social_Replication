def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 ---
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # --- Table 3 genre variables (exact set/order) ---
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
    missing = [v for _, v in genres if v not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # --- Row labels (exact order) ---
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

    # --- Missing code handling ---
    # Key correction vs prior attempts:
    # Do NOT allocate blanks/NaNs between DK and NA; leave them missing-but-not-displayed.
    # Only count explicit DK/NA codes.
    DK_NUM = {8, 98, -1}
    NA_NUM = {9, 99, -2}

    DK_STR = {
        "dk", "d",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_STR = {"na", "n", "no answer", "noanswer"}

    def classify_music_item(series: pd.Series):
        """
        Returns:
          sn: numeric series (NaN where non-numeric/blank)
          valid_mask: sn in 1..5
          dk_mask: explicit "don't know much about it" codes only
          na_mask: explicit "no answer" codes only
        """
        s = series
        sn = pd.to_numeric(s, errors="coerce")

        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        dk_num = sn.isin(list(DK_NUM)).fillna(False) & ~valid_mask
        na_num = sn.isin(list(NA_NUM)).fillna(False) & ~valid_mask

        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            low = s.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR).fillna(False) & ~valid_mask
            na_str = low.isin(NA_STR).fillna(False) & ~valid_mask
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        dk_mask = (dk_num | dk_str) & ~valid_mask
        na_mask = (na_num | na_str) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # --- Build numeric table (counts + mean) ---
    numeric = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=float)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]

        sn, valid_mask, dk_mask, na_mask = classify_music_item(s)

        numeric.loc["(1) Like very much", genre_label] = float((sn == 1).sum())
        numeric.loc["(2) Like it", genre_label] = float((sn == 2).sum())
        numeric.loc["(3) Mixed feelings", genre_label] = float((sn == 3).sum())
        numeric.loc["(4) Dislike it", genre_label] = float((sn == 4).sum())
        numeric.loc["(5) Dislike very much", genre_label] = float((sn == 5).sum())
        numeric.loc["(M) Don’t know much about it", genre_label] = float(dk_mask.sum())
        numeric.loc["(M) No answer", genre_label] = float(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        numeric.loc["Mean", genre_label] = float(mean_val) if pd.notna(mean_val) else np.nan

    # --- Format return table with Attitude first column ---
    formatted = pd.DataFrame(index=row_labels, columns=["Attitude"] + [g[0] for g in genres], dtype=object)
    formatted["Attitude"] = row_labels

    for genre_label, _ in genres:
        vals = []
        for r in row_labels:
            v = numeric.loc[r, genre_label]
            if r == "Mean":
                vals.append("" if pd.isna(v) else f"{float(v):.2f}")
            else:
                vals.append("" if pd.isna(v) else str(int(v)))
        formatted[genre_label] = vals

    # --- Save as human-readable text file with 3 panels (6 genres each) ---
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