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
    # Typed-missing handling that fixes the DK/NA mismatch:
    # Many GSS extracts encode typed missings using SPSS/Stata conventions:
    #   - "Don't know" often as 8 or 98 (sometimes 0 in some extracts)
    #   - "No answer" often as 9 or 99
    # Here we infer DK vs NA by using the fact that valid codes are 1..5,
    # and then treating:
    #   DK: any code where last digit is 8 (e.g., 8, 98)
    #   NA: any code where last digit is 9 (e.g., 9, 99)
    # This avoids the prior hardcoding that yielded zeros when the file uses 98/99
    # (or other *8/*9 forms) and avoids arbitrary redistribution of NaNs.
    #
    # Any other non-1..5 values (including NaN) are excluded from the mean and
    # not displayed in Table 3 (since Table 3 only shows DK and NA among missings).
    # --------------------
    VALID = {1, 2, 3, 4, 5}

    def to_num(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        ss = s.astype("string")
        ss = ss.where(ss.str.strip() != "", other=pd.NA)
        return pd.to_numeric(ss, errors="coerce")

    def dk_na_masks(sn: pd.Series):
        """
        Return (dk_mask, na_mask) for typed missings:
          DK if value % 10 == 8
          NA if value % 10 == 9
        Only applied to non-valid values to prevent any overlap.
        """
        sn_int = sn.round().astype("Int64")
        valid_mask = sn_int.isin(list(VALID)).fillna(False)

        dk_mask = ((sn_int % 10) == 8).fillna(False) & ~valid_mask
        na_mask = ((sn_int % 10) == 9).fillna(False) & ~valid_mask & ~dk_mask
        return dk_mask, na_mask

    # --------------------
    # Build table: include explicit Attitude label column
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

        # Typed missing categories (computed from raw codes)
        dk_mask, na_mask = dk_na_masks(sn)
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
        f.write("Frequencies are counts only (no percentages).\n")
        f.write("Mean computed over valid responses 1–5 only; missing categories excluded from mean.\n\n")

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