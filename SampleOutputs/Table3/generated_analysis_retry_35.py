def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 (case-insensitive column resolution) ---
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --- Genre variables (exact Table 3 order/labels) ---
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

    # --- Table 3 rows (exact order) ---
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

    # --- Robust code handling for DK/NA ---
    # IMPORTANT: Do not reallocate generic NaN/blanks; only count explicit missing types.
    VALID_CODES = {1, 2, 3, 4, 5}

    DK_TOKENS = {
        "dk", "d",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
        "don't know much about", "don’t know much about",
        "don't know enough about", "don’t know enough about",
    }
    NA_TOKENS = {"na", "n", "no answer", "noanswer"}

    def _pick_present_codes(sn: pd.Series, candidates):
        present = set(pd.Series(sn.dropna().unique()).astype(float).tolist()) if sn.notna().any() else set()
        out = []
        for c in candidates:
            if float(c) in present:
                out.append(float(c))
        return set(out)

    def _infer_dk_na_codes(sn: pd.Series):
        """
        Infer DK/NA numeric codes from those present in the data.
        Common patterns in GSS extracts: DK {8,98}, NA {9,99}, sometimes 0/6/7 used elsewhere.
        We only count codes we actually observe.
        """
        dk_candidates = [8, 98]
        na_candidates = [9, 99]

        dk = _pick_present_codes(sn, dk_candidates)
        na = _pick_present_codes(sn, na_candidates)

        # Some extracts may use 6/7 for DK/NA; only use if present AND not a valid 1-5
        # Prefer 8/9/98/99 when present.
        if not dk:
            dk |= _pick_present_codes(sn, [6, 7])
        if not na:
            na |= _pick_present_codes(sn, [0])

        # Ensure disjoint and not overlapping valid codes
        dk = {c for c in dk if int(c) not in VALID_CODES}
        na = {c for c in na if int(c) not in VALID_CODES and c not in dk}
        return dk, na

    def _count_item(series: pd.Series):
        s = series

        # Numeric parse
        sn = pd.to_numeric(s, errors="coerce")

        # Valid response counts (1-5)
        counts = {k: int((sn == k).sum()) for k in [1, 2, 3, 4, 5]}

        # Infer and count numeric DK/NA codes actually present
        dk_codes, na_codes = _infer_dk_na_codes(sn)
        dk = int(sn.isin(list(dk_codes)).sum()) if dk_codes else 0
        na = int(sn.isin(list(na_codes)).sum()) if na_codes else 0

        # Token-based DK/NA if stored as strings
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            low = s.astype("string").str.strip().str.lower()
            # only consider non-empty, non-numeric representations
            nonnum = sn.isna() & low.notna() & (low != "")
            dk += int((nonnum & low.isin(DK_TOKENS)).sum())
            na += int((nonnum & low.isin(NA_TOKENS)).sum())

        # Mean over valid 1-5 only
        mean_val = sn.where(sn.isin(list(VALID_CODES))).mean()

        return counts[1], counts[2], counts[3], counts[4], counts[5], dk, na, mean_val

    # --- Build table with Attitude first column (fix unlabeled rows issue) ---
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk, na, mean_val = _count_item(s)

        table.loc["(1) Like very much", genre_label] = c1
        table.loc["(2) Like it", genre_label] = c2
        table.loc["(3) Mixed feelings", genre_label] = c3
        table.loc["(4) Dislike it", genre_label] = c4
        table.loc["(5) Dislike very much", genre_label] = c5
        table.loc["(M) Don’t know much about it", genre_label] = dk
        table.loc["(M) No answer", genre_label] = na
        table.loc["Mean", genre_label] = mean_val

    # --- Format for display (counts as ints, means to 2 decimals) ---
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

    # --- Save a human-readable text file in 3 panels (6 genres each) ---
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
        f.write(title + "\n\n")
        f.write("Counts shown for responses 1–5 plus (M) Don’t know much about it and (M) No answer.\n")
        f.write("Mean computed over valid responses 1–5 only; (M) categories excluded from mean.\n\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
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
            f.write("\n")

    return formatted