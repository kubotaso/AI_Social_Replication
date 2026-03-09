def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Resolve columns case-insensitively ----
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # ---- Restrict to GSS 1993 ----
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
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

    # ---- Missing-code handling: infer DK vs NA from the data (no guessing/splitting) ----
    # We must distinguish the two "M" categories correctly.
    # Approach:
    #  1) Substantive: 1..5
    #  2) Among non-substantive numeric codes (and any other non-1..5 values), determine
    #     which code corresponds to DK vs NA by how frequently it appears:
    #       - DK is typically much more frequent than NA in this battery.
    #     If only one non-substantive code exists, treat it as DK and set NA=0.
    #  3) Also honor common conventions when present (8/98 for DK, 9/99 for NA) by seeding.

    DK_SEED = {8, 98, -1}
    NA_SEED = {9, 99, -2}

    def _compute_item_counts(series: pd.Series):
        s_num = pd.to_numeric(series, errors="coerce")
        valid = s_num.isin([1, 2, 3, 4, 5])

        # Counts for substantive categories
        c = {k: int((s_num == k).sum()) for k in [1, 2, 3, 4, 5]}

        # Candidate missing codes: numeric values not in 1..5
        non_sub = s_num[~valid & s_num.notna()]
        codes, freqs = np.unique(non_sub.to_numpy(), return_counts=True) if len(non_sub) else (np.array([]), np.array([]))

        code_counts = {float(code): int(cnt) for code, cnt in zip(codes, freqs)}

        dk_codes = set()
        na_codes = set()

        # Seed with conventional codes if present
        for code in list(code_counts.keys()):
            if code in DK_SEED:
                dk_codes.add(code)
            if code in NA_SEED:
                na_codes.add(code)

        # Ensure disjoint: if any overlap (unlikely), remove from NA
        na_codes -= dk_codes

        # Remaining unknown codes (not seeded)
        unknown = [code for code in code_counts.keys() if (code not in dk_codes and code not in na_codes)]

        # If both DK and NA already identified, leave unknown as "other missing" (not displayed)
        if not (dk_codes and na_codes):
            # If exactly one of DK/NA seeded, try to identify the other from unknown codes
            if dk_codes and (not na_codes):
                # Pick the least frequent unknown as NA (if any), else NA remains empty
                if unknown:
                    least = min(unknown, key=lambda x: code_counts.get(x, 0))
                    na_codes.add(least)
                    unknown = [u for u in unknown if u != least]
            elif na_codes and (not dk_codes):
                # Pick the most frequent unknown as DK (if any), else DK remains empty
                if unknown:
                    most = max(unknown, key=lambda x: code_counts.get(x, 0))
                    dk_codes.add(most)
                    unknown = [u for u in unknown if u != most]
            else:
                # Neither seeded: choose two codes by frequency if possible
                if len(unknown) >= 2:
                    # DK = most frequent, NA = least frequent (common pattern)
                    most = max(unknown, key=lambda x: code_counts.get(x, 0))
                    least = min([u for u in unknown if u != most], key=lambda x: code_counts.get(x, 0))
                    dk_codes.add(most)
                    na_codes.add(least)
                    unknown = [u for u in unknown if u not in (most, least)]
                elif len(unknown) == 1:
                    # Only one missing code: treat as DK
                    dk_codes.add(unknown[0])
                    unknown = []

        dk_cnt = int(s_num.isin(list(dk_codes)).sum()) if dk_codes else 0
        na_cnt = int(s_num.isin(list(na_codes)).sum()) if na_codes else 0

        mean_val = s_num.where(valid).mean()

        return c[1], c[2], c[3], c[4], c[5], dk_cnt, na_cnt, mean_val

    # ---- Build table with Attitude as first column ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk_cnt, na_cnt, mean_val = _compute_item_counts(s)

        table.loc["(1) Like very much", genre_label] = c1
        table.loc["(2) Like it", genre_label] = c2
        table.loc["(3) Mixed feelings", genre_label] = c3
        table.loc["(4) Dislike it", genre_label] = c4
        table.loc["(5) Dislike very much", genre_label] = c5
        table.loc["(M) Don’t know much about it", genre_label] = dk_cnt
        table.loc["(M) No answer", genre_label] = na_cnt
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for output (counts as integers; mean to 2 decimals) ----
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
    row_w = max(len(att_col), max(len(str(x)) for x in formatted[att_col].tolist())) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("Counts shown for 1–5 plus (M) DK and (M) No answer. Mean computed over 1–5 only.\n")
        f.write("DK/NA are inferred from non-1..5 numeric codes per item (DK assumed most frequent missing code; NA least frequent; 8/98 and 9/99 honored when present).\n")

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