def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Resolve columns case-insensitively ---
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --- Restrict to GSS 1993 ---
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

    missing_vars = [v for _, v in genres if v not in colmap]
    if missing_vars:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing_vars}")

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

    def _to_num(series: pd.Series) -> pd.Series:
        # robust numeric conversion (strings like "4.0" become 4.0)
        return pd.to_numeric(series, errors="coerce")

    def _compute_item_counts(series: pd.Series):
        """
        Compute Table 3 counts + mean for a single music item.
        Assumptions consistent with GSS for this battery:
          - 1..5 are substantive responses
          - 8 = Don't know much about it
          - 9 = No answer
        Any other values (including NaN) are treated as missing and folded into "No answer"
        for display (Table 3 only shows DK and No answer among missings).
        """
        sn = _to_num(series)

        # substantive counts
        c1 = int((sn == 1).sum())
        c2 = int((sn == 2).sum())
        c3 = int((sn == 3).sum())
        c4 = int((sn == 4).sum())
        c5 = int((sn == 5).sum())

        # missing-type counts (DO NOT swap; do not infer)
        dk_cnt = int((sn == 8).sum())
        na_cnt = int((sn == 9).sum())

        # Any other non-1..5 numeric codes + NaN are treated as "No answer" for display
        other_special = sn.notna() & ~sn.isin([1, 2, 3, 4, 5, 8, 9])
        na_cnt += int(other_special.sum())
        na_cnt += int(sn.isna().sum())

        mean_val = sn.where(sn.isin(list(valid_vals))).mean()

        return c1, c2, c3, c4, c5, dk_cnt, na_cnt, mean_val

    # --- Build table with Attitude as first column (explicit row labels) ---
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk_cnt, na_cnt, mean_val = _compute_item_counts(s)

        table.loc["(1) Like very much", genre_label] = str(c1)
        table.loc["(2) Like it", genre_label] = str(c2)
        table.loc["(3) Mixed feelings", genre_label] = str(c3)
        table.loc["(4) Dislike it", genre_label] = str(c4)
        table.loc["(5) Dislike very much", genre_label] = str(c5)
        table.loc["(M) Don’t know much about it", genre_label] = str(dk_cnt)
        table.loc["(M) No answer", genre_label] = str(na_cnt)
        table.loc["Mean", genre_label] = "" if pd.isna(mean_val) else f"{float(mean_val):.2f}"

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
    row_w = max(len(att_col), max(len(str(x)) for x in table[att_col].tolist())) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("Coding used for missing-type rows: 8=Don’t know much about it; 9=No answer. Mean computed over 1–5 only.\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(table[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            header = _pad(att_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = _pad(table.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = table.loc[r, c]
                    line += _pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")

    return table