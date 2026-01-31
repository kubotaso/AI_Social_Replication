def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column mapping
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --------------------
    # Restrict to YEAR == 1993
    # --------------------
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order, exact headers)
    # --------------------
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

    # --------------------
    # Row labels (exact order)
    # --------------------
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

    # --------------------
    # Missing handling (FIX)
    # Use explicit GSS-style codes if present:
    #   Don't know much about it -> 8 (also allow 98, -1 defensively)
    #   No answer               -> 9 (also allow 99, -2 defensively)
    #
    # IMPORTANT: Do NOT "allocate" NaNs/blanks across DK/NA. If the extract uses NaN for
    # missing but doesn't encode type, we cannot recover DK vs NA; the published table
    # relies on typed missing codes. Here we count only explicit DK/NA codes as those rows.
    # Other missing (NaN/blank/other special codes) are excluded from the mean but not shown.
    # --------------------
    VALID = {1, 2, 3, 4, 5}
    DK_CODES = {8, 98, -1}
    NA_CODES = {9, 99, -2}

    def to_num(s: pd.Series) -> pd.Series:
        # Handles numeric columns and string columns with blanks
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        ss = s.astype("string")
        ss = ss.where(ss.str.strip() != "", other=pd.NA)
        return pd.to_numeric(ss, errors="coerce")

    # --------------------
    # Build table: explicit Attitude column + 18 genre columns
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        sn = to_num(df[col])

        # Frequencies for 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # Typed missing categories (explicit codes only)
        dk_mask = sn.isin(list(DK_CODES)).fillna(False)
        na_mask = sn.isin(list(NA_CODES)).fillna(False)
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        # Mean over valid 1..5 only
        valid_mask = sn.isin(list(VALID)).fillna(False)
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = float(mean_val) if not pd.isna(mean_val) else np.nan

    # --------------------
    # Format for display (counts as ints; mean 2 decimals)
    # --------------------
    formatted = table.copy()
    for r in row_labels:
        for c in formatted.columns:
            if c == "Attitude":
                continue
            v = formatted.loc[r, c]
            if r == "Mean":
                formatted.loc[r, c] = "" if pd.isna(v) else f"{float(v):.2f}"
            else:
                formatted.loc[r, c] = "" if pd.isna(v) else str(int(v))

    # --------------------
    # Save as human-readable text in 3 panels (6 genres each)
    # --------------------
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    panels = [
        [g[0] for g in genres[0:6]],
        [g[0] for g in genres[6:12]],
        [g[0] for g in genres[12:18]],
    ]

    def pad(text, width, align="left"):
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
    row_w = max(len(att_col), int(formatted[att_col].astype(str).map(len).max())) + 2

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

            header = pad(att_col, row_w, "left") + "".join(pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = pad(formatted.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[r, c]
                    line += pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")
            f.write("\n")

    return formatted