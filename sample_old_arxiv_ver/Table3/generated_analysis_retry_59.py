def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- normalize column lookup (case-insensitive) ----
    colmap = {str(c).strip().lower(): c for c in df.columns}

    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # ---- Table 3 genre variables (exact order/labels) ----
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

    VALID = [1, 2, 3, 4, 5]

    # ---- helpers ----
    def _series_to_numeric(s: pd.Series) -> pd.Series:
        # robust numeric conversion; blanks -> NA
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = s.astype("string")
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    def _is_blank_or_na(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s.isna()
        st = s.astype("string")
        return st.isna() | (st.str.strip() == "")

    # ---- determine DK/NA coding from data (no guessing/splitting) ----
    # Identify explicit codes for:
    #   DK = "don't know much about it"
    #   NA = "no answer"
    #
    # We learn the codes by looking for values >5 that occur across multiple genre items.
    # Then we assign:
    #   DK code = most common value among {6,7,8} seen across items (prefers 6/7/8)
    #   NA code = most common value among {9} seen across items; otherwise among {8,9}
    #
    # If nothing found, DK/NA counts will be 0 for that item (we do not fabricate).
    candidate_pool = []
    for _, v in genres:
        sn = _series_to_numeric(df[colmap[v]])
        vals = sn.dropna().astype(int)
        vals = vals[(vals >= 6) & (vals <= 99)]
        candidate_pool.extend(vals.tolist())

    counts = pd.Series(candidate_pool).value_counts() if len(candidate_pool) else pd.Series(dtype=int)

    # prefer typical GSS codes if present
    dk_candidates_pref = [6, 7, 8, 98]
    na_candidates_pref = [9, 99]

    def pick_code(preferred, fallback):
        for c in preferred:
            if c in counts.index:
                return int(c)
        for c in fallback:
            if c in counts.index:
                return int(c)
        return None

    DK_CODE = pick_code(dk_candidates_pref, fallback=[8, 7, 6])
    NA_CODE = pick_code(na_candidates_pref, fallback=[9, 8, 99])

    # If DK_CODE and NA_CODE collapse to same value, treat NA_CODE as next best distinct
    if DK_CODE is not None and NA_CODE is not None and DK_CODE == NA_CODE:
        for c in na_candidates_pref + [9, 99, 8, 7, 6]:
            if c != DK_CODE and c in counts.index:
                NA_CODE = int(c)
                break
        else:
            NA_CODE = None  # avoid double counting

    # ---- build table (counts + mean) ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        col = colmap[var_lower]
        raw = df[col]
        sn = _series_to_numeric(raw)

        # frequencies for 1..5
        for v, rl in zip(VALID, row_labels[:5]):
            table.loc[rl, genre_label] = int((sn == v).sum())

        # typed missing categories only if explicit codes exist; no redistribution of blanks
        dk_mask = pd.Series(False, index=sn.index)
        na_mask = pd.Series(False, index=sn.index)

        if DK_CODE is not None:
            dk_mask = (sn == DK_CODE)
        if NA_CODE is not None:
            na_mask = (sn == NA_CODE)

        # ensure we don't accidentally count valid 1..5 as missing
        valid_mask = sn.isin(VALID).fillna(False)
        dk_mask = dk_mask.fillna(False) & ~valid_mask
        na_mask = na_mask.fillna(False) & ~valid_mask & ~dk_mask

        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        # mean over valid 1..5 only
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

        # Note: blanks/NaN in CSV are treated as generic missing but are not split into DK vs NA.
        # This prevents deterministic fabrication and matches "compute from raw data" requirement.

    # ---- format for display ----
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

    # ---- write human-readable text in 3 panels (6 genres each) ----
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
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n")
        f.write(f"Detected DK code: {DK_CODE if DK_CODE is not None else 'None'}; "
                f"Detected NA code: {NA_CODE if NA_CODE is not None else 'None'}.\n\n")

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