def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --------------------
    # Restrict to GSS 1993 (YEAR == 1993), case-insensitive
    # --------------------
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 variables (exact order)
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

    VALID = {1, 2, 3, 4, 5}

    # --------------------
    # Typed missing handling:
    # This dataset is a CSV extract; DK/NA may be blank (NaN) with no typed distinction.
    # To avoid inventing DK vs NA from blanks, we:
    #   - Count explicit DK/NA codes/tokens if present.
    #   - Otherwise, leave DK/NA as 0 and report all other missing as "No answer"
    #     ONLY if we can detect a "no answer" token/code. If neither DK nor NA is
    #     identifiable, we assign all missing to "No answer" (a conventional
    #     nonresponse bucket) and DK to 0. This is deterministic and avoids
    #     arbitrary splitting that previously caused mismatches.
    #
    # NOTE: If the source file encodes DK/NA via specific numeric codes, this will
    # correctly compute both from raw data.
    # --------------------
    DK_NUM_CODES = {8, 98, 998, -1, -9}
    NA_NUM_CODES = {9, 99, 999, -2, -8}
    OTHER_MISS_NUM = {0, 97, 997, -3, -4, -5, -6, -7}

    DK_STR_TOKENS = {
        "dk", "d/k", "dont know", "don't know", "dontknow",
        "dont know much", "don't know much",
        "dont know much about it", "don't know much about it",
    }
    NA_STR_TOKENS = {"na", "n/a", "no answer", "noanswer"}

    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        sr = _as_string(raw)
        sr = sr.where(sr.str.strip() != "", other=pd.NA)
        return pd.to_numeric(sr, errors="coerce")

    def _is_blank_or_nan(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        sr = _as_string(raw)
        return sr.isna() | (sr.str.strip() == "")

    def _explicit_string_dk_na(raw: pd.Series):
        if pd.api.types.is_numeric_dtype(raw):
            false = pd.Series(False, index=raw.index)
            return false, false
        sr = _as_string(raw).fillna("")
        s2 = sr.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        dk = s2.isin(DK_STR_TOKENS) | s2.str.contains("don't know", regex=False) | s2.str.contains("dont know", regex=False)
        na = s2.isin(NA_STR_TOKENS) | s2.str.contains("no answer", regex=False)
        return dk.fillna(False), na.fillna(False)

    def classify_item(raw: pd.Series):
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        present = set(pd.Series(sn.dropna().unique()).tolist())
        dk_codes = [c for c in DK_NUM_CODES if c in present]
        na_codes = [c for c in NA_NUM_CODES if c in present]
        other_codes = [c for c in OTHER_MISS_NUM if c in present]

        dk_num = sn.isin(dk_codes).fillna(False) if dk_codes else pd.Series(False, index=sn.index)
        na_num = sn.isin(na_codes).fillna(False) if na_codes else pd.Series(False, index=sn.index)
        other_num = sn.isin(other_codes).fillna(False) if other_codes else pd.Series(False, index=sn.index)

        dk_str, na_str = _explicit_string_dk_na(raw)

        dk_explicit = (dk_num | dk_str) & ~valid_mask
        na_explicit = (na_num | na_str) & ~valid_mask & ~dk_explicit
        other_explicit = other_num & ~valid_mask & ~dk_explicit & ~na_explicit

        generic_missing = _is_blank_or_nan(raw) & ~valid_mask & ~dk_explicit & ~na_explicit & ~other_explicit

        # If we cannot distinguish DK/NA in blanks, do NOT split arbitrarily.
        # Put generic missing into "No answer" by default; DK remains only explicit DK.
        dk_mask = dk_explicit
        na_mask = na_explicit | generic_missing

        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table (counts only; mean over valid 1..5)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask = classify_item(raw)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --------------------
    # Format (counts as integers; mean to 2 decimals)
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

    glabels = [g[0] for g in genres]
    panels = [glabels[0:6], glabels[6:12], glabels[12:18]]

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
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n")
        f.write("DK/NA counted from explicit codes/tokens when present; otherwise blanks treated as No answer.\n\n")

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