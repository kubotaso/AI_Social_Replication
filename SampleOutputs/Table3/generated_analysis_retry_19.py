def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 ---
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    df = df.loc[pd.to_numeric(df[colmap["year"]], errors="coerce") == 1993].copy()

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

    valid_vals = {1, 2, 3, 4, 5}

    # --- Determine DK vs NA codes from the data (no hard-coded paper numbers) ---
    # Prefer numeric codes that are common across many items; DK should be much more frequent than NA.
    special_total = {}
    special_items = {}
    for _, var_lower in genres:
        s = pd.to_numeric(df[colmap[var_lower]], errors="coerce")
        specials = s[s.notna() & ~s.isin(list(valid_vals))]
        if specials.empty:
            continue
        vc = specials.value_counts()
        for code, cnt in vc.items():
            code = float(code)
            special_total[code] = special_total.get(code, 0) + int(cnt)
            special_items[code] = special_items.get(code, 0) + 1

    # candidates must appear across multiple items (helps avoid stray misc codes)
    candidates = [c for c in special_total if special_items.get(c, 0) >= 6]
    ranked = sorted(candidates, key=lambda c: (-special_total[c], c))

    dk_code = None
    na_code = None
    if len(ranked) >= 2:
        # DK is expected to be the most frequent special code across these "know about it" items
        dk_code, na_code = ranked[0], ranked[1]
    elif len(ranked) == 1:
        dk_code, na_code = ranked[0], None
    else:
        # If the extract has no explicit non-1..5 codes, DK/NA can't be distinguished reliably.
        # Keep as None (counts will be 0), but the table will still compute correctly for 1..5 and mean.
        dk_code, na_code = None, None

    def compute_item(series, dk_code_val, na_code_val):
        sn = pd.to_numeric(series, errors="coerce")

        counts_1_5 = {k: int((sn == k).sum()) for k in [1, 2, 3, 4, 5]}
        dk_cnt = int((sn == dk_code_val).sum()) if dk_code_val is not None else 0
        na_cnt = int((sn == na_code_val).sum()) if na_code_val is not None else 0

        mean_val = sn.where(sn.isin([1, 2, 3, 4, 5])).mean()
        return counts_1_5, dk_cnt, na_cnt, mean_val

    # --- Build numeric table (rows x genres) ---
    numeric = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=float)

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        counts, dk_cnt, na_cnt, mean_val = compute_item(df[col], dk_code, na_code)

        numeric.loc["(1) Like very much", genre_label] = counts[1]
        numeric.loc["(2) Like it", genre_label] = counts[2]
        numeric.loc["(3) Mixed feelings", genre_label] = counts[3]
        numeric.loc["(4) Dislike it", genre_label] = counts[4]
        numeric.loc["(5) Dislike very much", genre_label] = counts[5]
        numeric.loc["(M) Don’t know much about it", genre_label] = dk_cnt
        numeric.loc["(M) No answer", genre_label] = na_cnt
        numeric.loc["Mean", genre_label] = float(mean_val) if pd.notna(mean_val) else np.nan

    # --- Format return table with Attitude first column ---
    formatted = pd.DataFrame(index=row_labels, columns=["Attitude"] + [g[0] for g in genres], dtype=object)
    formatted["Attitude"] = row_labels
    for genre_label, _ in genres:
        out_col = []
        for r in row_labels:
            v = numeric.loc[r, genre_label]
            if r == "Mean":
                out_col.append("" if pd.isna(v) else f"{float(v):.2f}")
            else:
                out_col.append("" if pd.isna(v) else str(int(round(float(v)))))
        formatted[genre_label] = out_col

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