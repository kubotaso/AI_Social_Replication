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

    # ---- Determine DK/NA codes robustly from observed values in the music items ----
    # GSS commonly uses:
    #   DK: 8 or 98 ; NA: 9 or 99
    # Some extracts may use 6/7/8/9 variants, but we will infer from labels if present,
    # otherwise infer from value distribution among 6..99 across the music items.
    DK_PREF = [8, 98, 6, 7]
    NA_PREF = [9, 99]

    # Pool candidates across all genre items (values outside 1..5)
    pool = []
    for _, v in genres:
        sn = _series_to_numeric(df[colmap[v]])
        vals = sn.dropna()
        vals = vals[~vals.isin(VALID)]
        vals = vals[(vals >= 6) & (vals <= 99)]
        pool.extend(vals.astype(int).tolist())

    vc = pd.Series(pool).value_counts() if pool else pd.Series(dtype=int)

    def _pick(pref_list, vc_index):
        for code in pref_list:
            if code in vc_index:
                return int(code)
        return None

    DK_CODE = _pick(DK_PREF, vc.index)
    NA_CODE = _pick(NA_PREF, vc.index)

    # If neither explicit DK/NA code appears in this CSV, we still must compute DK/NA
    # counts from raw data. In this extract, DK/NA are stored as system-missing (blank/NaN),
    # but are distinguishable in the original GSS by type. Since the CSV does not retain
    # typed missingness, we cannot truthfully split blanks into DK vs NA without external
    # information. Therefore, we:
    #   - Use explicit codes if present.
    #   - Otherwise, classify blanks as generic missing and report them under DK with NA=0,
    #     BUT ALSO record a note in the output file that typed missingness was not present.
    #
    # This avoids fabricating DK vs NA counts while still keeping total Ns correct.
    typed_missing_available = (DK_CODE is not None) or (NA_CODE is not None)

    # ---- build table (counts + mean) ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        raw = df[colmap[var_lower]]
        sn = _series_to_numeric(raw)
        valid_mask = sn.isin(VALID).fillna(False)

        # frequencies for 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # missing categories
        if typed_missing_available:
            dk_mask = pd.Series(False, index=sn.index)
            na_mask = pd.Series(False, index=sn.index)

            if DK_CODE is not None:
                dk_mask = (sn == DK_CODE)
            if NA_CODE is not None:
                na_mask = (sn == NA_CODE)

            dk_mask = dk_mask.fillna(False) & ~valid_mask
            na_mask = na_mask.fillna(False) & ~valid_mask & ~dk_mask

            # also include blanks as generic missing; they are not typed in this extract
            blank_mask = _is_blank_or_na(raw) & ~valid_mask & ~dk_mask & ~na_mask

            # Put untyped blanks into DK to keep totals correct without inventing NA.
            dk_mask = dk_mask | blank_mask

            table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
            table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        else:
            # No explicit typed DK/NA codes exist in this CSV; count blanks as missing.
            blank_mask = _is_blank_or_na(raw) & ~valid_mask
            table.loc["(M) Don’t know much about it", genre_label] = int(blank_mask.sum())
            table.loc["(M) No answer", genre_label] = 0

        # mean over valid 1..5 only
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

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
        f.write("Mean computed over valid responses 1–5 only; missing excluded from mean.\n")
        if typed_missing_available:
            f.write(
                f"Detected missing codes in this extract: DK={DK_CODE if DK_CODE is not None else 'None'}, "
                f"NA={NA_CODE if NA_CODE is not None else 'None'}; "
                "untyped blanks (if any) are counted under DK to preserve total N.\n\n"
            )
        else:
            f.write(
                "This CSV extract contains no explicit typed DK/NA codes; blanks/NaN are counted under DK, "
                "and NA is reported as 0 (typed split not recoverable from this file).\n\n"
            )

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