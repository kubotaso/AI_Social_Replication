def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 ---
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
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

    # --- Missing code handling (robust, data-driven) ---
    # GSS-style special codes vary by extract. We infer DK vs NA from the *distribution*
    # of non-substantive codes across the 18 music items:
    #   - DK should be the larger missing category for these items
    #   - NA should be the smaller missing category
    #
    # Algorithm:
    # 1) Collect candidate special codes from each item: values not in {1..5}, excluding NaN.
    # 2) Keep only codes appearing in >= 2 items and with total frequency > 0.
    # 3) Choose the two most common candidate codes overall; map the more frequent -> DK, other -> NA.
    # 4) Count DK/NA using those inferred codes only. (Blank/NaN remain missing but not displayed.)
    #
    # This avoids hard-coding paper numbers and avoids mis-allocating NaN/blanks into DK/NA.

    valid_values = {1, 2, 3, 4, 5}

    # Build a combined frequency of "special" codes across all items
    special_counts = {}
    appear_in_items = {}

    for _, var_lower in genres:
        col = colmap[var_lower]
        sn = pd.to_numeric(df[col], errors="coerce")
        specials = sn[~sn.isin(list(valid_values)) & sn.notna()]
        vc = specials.value_counts(dropna=True)
        for code, cnt in vc.items():
            code_key = float(code)
            special_counts[code_key] = special_counts.get(code_key, 0) + int(cnt)
            appear_in_items[code_key] = appear_in_items.get(code_key, 0) + 1

    # Candidate codes: appear in >=2 items and are not valid 1..5
    candidates = [
        code for code, total in special_counts.items()
        if (code not in valid_values) and (appear_in_items.get(code, 0) >= 2) and (total > 0)
    ]

    # Pick DK/NA codes
    dk_code = None
    na_code = None
    if len(candidates) >= 2:
        ranked = sorted(candidates, key=lambda c: (-special_counts.get(c, 0), c))
        top2 = ranked[:2]
        # DK is typically more frequent than NA for these "know about it" items
        if special_counts[top2[0]] >= special_counts[top2[1]]:
            dk_code, na_code = top2[0], top2[1]
        else:
            dk_code, na_code = top2[1], top2[0]
    elif len(candidates) == 1:
        # If only one special code exists consistently, treat it as DK and leave NA as none
        dk_code = candidates[0]
        na_code = None
    else:
        # No explicit special codes found; fall back to common GSS conventions
        # (still not paper numbers; just a coding fallback)
        dk_code, na_code = 8.0, 9.0

    def compute_counts_and_mean(series: pd.Series, dk_code_val, na_code_val):
        sn = pd.to_numeric(series, errors="coerce")

        counts = {k: int((sn == k).sum()) for k in [1, 2, 3, 4, 5]}

        dk_cnt = int((sn == dk_code_val).sum()) if dk_code_val is not None else 0
        na_cnt = int((sn == na_code_val).sum()) if na_code_val is not None else 0

        # Mean on 1..5 only
        mean_val = sn.where(sn.isin([1, 2, 3, 4, 5])).mean()
        return counts, dk_cnt, na_cnt, mean_val

    # --- Build numeric table ---
    numeric = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=float)

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        counts, dk_cnt, na_cnt, mean_val = compute_counts_and_mean(df[col], dk_code, na_code)

        numeric.loc["(1) Like very much", genre_label] = float(counts[1])
        numeric.loc["(2) Like it", genre_label] = float(counts[2])
        numeric.loc["(3) Mixed feelings", genre_label] = float(counts[3])
        numeric.loc["(4) Dislike it", genre_label] = float(counts[4])
        numeric.loc["(5) Dislike very much", genre_label] = float(counts[5])
        numeric.loc["(M) Don’t know much about it", genre_label] = float(dk_cnt)
        numeric.loc["(M) No answer", genre_label] = float(na_cnt)
        numeric.loc["Mean", genre_label] = float(mean_val) if pd.notna(mean_val) else np.nan

    # --- Format return table with Attitude first column ---
    formatted = pd.DataFrame(index=row_labels, columns=["Attitude"] + [g[0] for g in genres], dtype=object)
    formatted["Attitude"] = row_labels

    for genre_label, _ in genres:
        col_vals = []
        for r in row_labels:
            v = numeric.loc[r, genre_label]
            if r == "Mean":
                col_vals.append("" if pd.isna(v) else f"{float(v):.2f}")
            else:
                col_vals.append("" if pd.isna(v) else str(int(round(float(v)))))
        formatted[genre_label] = col_vals

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
        f.write(f"(Inferred missing codes: DK={dk_code if dk_code is not None else 'None'}, NA={na_code if na_code is not None else 'None'})\n")

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