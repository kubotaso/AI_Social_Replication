def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # --------------------
    # Load data
    # --------------------
    df = pd.read_csv(data_source)

    # Case-insensitive column resolution
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # Restrict to YEAR == 1993
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

    # --------------------
    # Missingness classification
    # Key fix vs prior attempts:
    #  - Do NOT fabricate/redistribute NaNs into DK/NA.
    #  - Compute DK and NA ONLY from explicit raw codes/tokens.
    #  - Do NOT assume 8/9; detect which codes are used in the file.
    # --------------------
    VALID = {1, 2, 3, 4, 5}

    # Common GSS-style "missing type" code pairs: (DK, NA)
    CANDIDATE_PAIRS = [
        (8, 9),
        (98, 99),
        (-1, -2),
        (0, 9),   # sometimes 0 used as DK/Not applicable in extracts; kept as fallback
        (7, 9),   # another occasional pair in coded surveys; fallback only
    ]

    DK_STR_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
    }
    NA_STR_TOKENS = {"n", "na", "no answer", "noanswer"}

    def _to_numeric_preserve_strings(series: pd.Series) -> pd.Series:
        # Convert blanks to NA first, then numeric coercion
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        s = series.astype("string")
        s = s.where(~(s.str.strip() == ""), other=pd.NA)
        return pd.to_numeric(s, errors="coerce")

    def _detect_dk_na_codes(sn: pd.Series):
        """
        Detect DK and NA numeric codes used in this variable in THIS dataset.
        Chooses the candidate pair that is most supported by observed codes.
        If none present, returns empty sets (counts will come from string tokens only).
        """
        present = set(sn.dropna().unique().tolist())
        # remove valid codes from consideration
        present_maybe_missing = {x for x in present if x not in VALID}

        best_pair = None
        best_score = -1
        for dk, na in CANDIDATE_PAIRS:
            score = int(dk in present_maybe_missing) + int(na in present_maybe_missing)
            if score > best_score:
                best_score = score
                best_pair = (dk, na)

        dk_codes, na_codes = set(), set()
        if best_pair is not None:
            dk, na = best_pair
            if dk in present_maybe_missing:
                dk_codes.add(dk)
            if na in present_maybe_missing:
                na_codes.add(na)

        return dk_codes, na_codes

    def _classify(series: pd.Series):
        """
        Returns:
          sn: numeric series (float, NaN where non-numeric)
          valid_mask: sn in 1..5
          dk_mask: explicit "don't know much about it"
          na_mask: explicit "no answer"
        Note: NaNs/blanks that are not explicitly coded are NOT forced into DK/NA.
        """
        sn = _to_numeric_preserve_strings(series)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        dk_codes, na_codes = _detect_dk_na_codes(sn)
        dk_num = sn.isin(list(dk_codes)).fillna(False) if dk_codes else pd.Series(False, index=series.index)
        na_num = sn.isin(list(na_codes)).fillna(False) if na_codes else pd.Series(False, index=series.index)

        if (series.dtype == "object") or str(series.dtype).startswith("string"):
            low = series.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR_TOKENS).fillna(False)
            na_str = low.isin(NA_STR_TOKENS).fillna(False)
        else:
            dk_str = pd.Series(False, index=series.index)
            na_str = pd.Series(False, index=series.index)

        # Ensure disjoint, and exclude valid
        dk_mask = (dk_num | dk_str) & ~valid_mask
        na_mask = (na_num | na_str) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build the table (with explicit Attitude column)
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        s = df[col]
        sn, valid_mask, dk_mask, na_mask = _classify(s)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = mean_val

    # --------------------
    # Format for display (counts ints; means 2 decimals)
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