def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    df = df.loc[pd.to_numeric(df[colmap["year"]], errors="coerce") == 1993].copy()

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
    missing_vars = [v for _, v in genres if v not in colmap]
    if missing_vars:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing_vars}")

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

    valid_vals = [1, 2, 3, 4, 5]

    # ---- Infer numeric codes for DK vs NA from data (robust, no paper numbers) ----
    # Many GSS extracts use distinct special numeric codes (e.g., 8/9, 98/99, etc.).
    # We'll:
    #  1) gather all non-1..5 numeric codes across these 18 items
    #  2) consider only codes appearing in multiple items
    #  3) assign DK = most frequent special code; NA = second most frequent
    #     (DK is typically much more common than NA on these "don't know much about it" options)
    special_total = {}
    special_items = {}

    for _, var_lower in genres:
        s = pd.to_numeric(df[colmap[var_lower]], errors="coerce")
        specials = s[s.notna() & ~s.isin(valid_vals)]
        if specials.empty:
            continue
        vc = specials.value_counts()
        for code, cnt in vc.items():
            # normalize numeric keys
            if pd.isna(code):
                continue
            code_key = float(code)
            special_total[code_key] = special_total.get(code_key, 0) + int(cnt)
            special_items[code_key] = special_items.get(code_key, 0) + 1

    # Candidate codes must appear across multiple items to avoid stray misc codes.
    # Using >= 6 (one third of items) is a good stability threshold.
    candidates = [c for c in special_total if special_items.get(c, 0) >= 6]
    ranked = sorted(candidates, key=lambda c: (-special_total[c], c))

    dk_code = None
    na_code = None
    if len(ranked) >= 2:
        dk_code, na_code = ranked[0], ranked[1]
    elif len(ranked) == 1:
        dk_code, na_code = ranked[0], None

    def compute_item(series, dk_code_val, na_code_val):
        sn = pd.to_numeric(series, errors="coerce")

        counts_1_5 = {k: int((sn == k).sum()) for k in valid_vals}

        # DK/NA counts strictly from the inferred numeric codes
        dk_cnt = int((sn == dk_code_val).sum()) if dk_code_val is not None else 0
        na_cnt = int((sn == na_code_val).sum()) if na_code_val is not None else 0

        mean_val = sn.where(sn.isin(valid_vals)).mean()
        return counts_1_5, dk_cnt, na_cnt, mean_val

    # ---- Build numeric table ----
    numeric = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=float)

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        counts, dk_cnt, na_cnt, mean_val = compute_item(s, dk_code, na_code)

        numeric.loc["(1) Like very much", genre_label] = counts[1]
        numeric.loc["(2) Like it", genre_label] = counts[2]
        numeric.loc["(3) Mixed feelings", genre_label] = counts[3]
        numeric.loc["(4) Dislike it", genre_label] = counts[4]
        numeric.loc["(5) Dislike very much", genre_label] = counts[5]
        numeric.loc["(M) Don’t know much about it", genre_label] = dk_cnt
        numeric.loc["(M) No answer", genre_label] = na_cnt
        numeric.loc["Mean", genre_label] = float(mean_val) if pd.notna(mean_val) else np.nan

    # ---- Format return table with Attitude first column ----
    formatted = pd.DataFrame(index=row_labels, columns=["Attitude"] + [g[0] for g in genres], dtype=object)
    formatted["Attitude"] = row_labels

    for genre_label, _ in genres:
        col_out = []
        for r in row_labels:
            v = numeric.loc[r, genre_label]
            if r == "Mean":
                col_out.append("" if pd.isna(v) else f"{float(v):.2f}")
            else:
                col_out.append("" if pd.isna(v) else str(int(round(float(v)))))
        formatted[genre_label] = col_out

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

    att_col = "Attitude"
    row_w = max(len(att_col), max(len(str(x)) for x in formatted[att_col])) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write(
            f"(Missing-code inference from data: DK_code={dk_code if dk_code is not None else 'None'}, "
            f"NA_code={na_code if na_code is not None else 'None'})\n"
        )

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(c), max_cell_len) + 4

            header = _pad(att_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for i in range(len(formatted)):
                att = formatted.iloc[i][att_col]
                line = _pad(att, row_w, "left")
                for c in panel_cols:
                    val = formatted.iloc[i][c]
                    line += _pad(val, widths[c], "center" if att == "Mean" else "right")
                f.write(line + "\n")

    return formatted