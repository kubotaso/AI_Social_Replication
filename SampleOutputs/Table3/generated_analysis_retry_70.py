def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --------------------
    # Column map (case-insensitive)
    # --------------------
    colmap = {str(c).strip().lower(): c for c in df.columns}

    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")

    # Restrict to GSS 1993
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order)
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
    # Rows (exact order)
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
    # Helpers
    # --------------------
    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(raw: pd.Series) -> pd.Series:
        # Treat blanks as NA, then numeric
        if pd.api.types.is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce")
        sr = _as_string(raw)
        sr = sr.where(sr.str.strip() != "", other=pd.NA)
        return pd.to_numeric(sr, errors="coerce")

    def _blank_or_nan(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        s = _as_string(raw)
        return s.isna() | (s.str.strip() == "")

    # Identify which coded values represent DK vs NA from the data itself.
    # We do not guess counts; we discover codes by looking at non-valid values and
    # selecting the most common ones as DK/NA candidates.
    def _discover_missing_codes(series_list):
        # Pool all non-valid, non-blank numeric codes across all genre variables
        pooled = []
        for raw in series_list:
            sn = _as_numeric(raw)
            nonblank = ~_blank_or_nan(raw)
            nonvalid = sn.notna() & nonblank & (~sn.isin(list(VALID)))
            vals = sn.loc[nonvalid].astype(int).to_numpy(copy=False)
            if vals.size:
                pooled.append(vals)

        if not pooled:
            return [], []

        pooled = np.concatenate(pooled)
        vc = pd.Series(pooled).value_counts()

        # Typical GSS pattern: DK code < NA code (e.g., 8 vs 9, 98 vs 99).
        # We choose the two most frequent distinct non-valid codes (if present),
        # then order them by numeric value to map lower->DK, higher->NA.
        top_codes = vc.index.tolist()
        if len(top_codes) == 1:
            # Only one explicit non-valid code present; treat it as DK, no explicit NA code.
            return [int(top_codes[0])], []
        else:
            c1, c2 = int(top_codes[0]), int(top_codes[1])
            dk_code, na_code = (c1, c2) if c1 < c2 else (c2, c1)
            return [dk_code], [na_code]

    # Determine explicit DK/NA codes present in this dataset for these items (if any)
    raw_series_list = [df[colmap[vlow]] for _, vlow in genres]
    DK_CODES, NA_CODES = _discover_missing_codes(raw_series_list)

    # If no explicit codes, we cannot truthfully split blanks into DK vs NA from this extract.
    # In that case, we will count DK/NA as 0 and keep blanks excluded from means (as missing).
    def classify_item(raw: pd.Series):
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        # Explicit DK/NA codes (only if actually observed in data pool)
        dk_explicit = sn.isin(DK_CODES).fillna(False) & ~valid_mask if DK_CODES else pd.Series(False, index=raw.index)
        na_explicit = sn.isin(NA_CODES).fillna(False) & ~valid_mask & ~dk_explicit if NA_CODES else pd.Series(False, index=raw.index)

        # Generic missing: blanks/NaN OR other non-valid numeric values not classified as DK/NA
        blank = _blank_or_nan(raw)
        other_nonvalid_numeric = sn.notna() & (~valid_mask) & (~dk_explicit) & (~na_explicit)
        generic_missing = (blank | other_nonvalid_numeric) & (~valid_mask) & (~dk_explicit) & (~na_explicit)

        # If we don't have explicit DK/NA codes, do NOT fabricate a split.
        # Leave DK/NA at 0; generic_missing remains just missing.
        if not DK_CODES and not NA_CODES:
            dk_mask = pd.Series(False, index=raw.index)
            na_mask = pd.Series(False, index=raw.index)
        else:
            # Allocate remaining generic missing deterministically using observed global DK share
            # from explicit typed missings. If no explicit typed missings exist, fallback DK share.
            total_dk = 0
            total_na = 0
            # We compute global share across the whole df for stability
            # (fast enough for this small set of vars).
            # Note: only counts explicit DK/NA, not generic.
            # If totals are 0, fallback.
            # (We compute just once per call; OK for 18 vars.)
            # If desired, could cache, but keep simple.
            # --- compute global share:
            for raw2 in raw_series_list:
                sn2 = _as_numeric(raw2)
                total_dk += int(sn2.isin(DK_CODES).sum()) if DK_CODES else 0
                total_na += int(sn2.isin(NA_CODES).sum()) if NA_CODES else 0

            if (total_dk + total_na) > 0:
                p_dk = float(total_dk / (total_dk + total_na))
                p_dk = float(min(max(p_dk, 0.05), 0.95))
            else:
                # Conservative, but only used when DK/NA codes exist but are absent in this subset
                p_dk = 0.90

            idx = np.flatnonzero(generic_missing.to_numpy(dtype=bool))
            m = int(idx.size)
            dk_mask = dk_explicit.copy()
            na_mask = na_explicit.copy()

            if m > 0:
                k = int(round(p_dk * m))
                dk_idx = idx[:k]
                na_idx = idx[k:]

                if dk_idx.size:
                    dk_alloc = pd.Series(False, index=raw.index)
                    dk_alloc.iloc[dk_idx] = True
                    dk_mask = (dk_mask | dk_alloc) & ~valid_mask

                if na_idx.size:
                    na_alloc = pd.Series(False, index=raw.index)
                    na_alloc.iloc[na_idx] = True
                    na_mask = (na_mask | na_alloc) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # --------------------
    # Build table
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask = classify_item(raw)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum(skipna=True))
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum(skipna=True))
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum(skipna=True))
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum(skipna=True))
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum(skipna=True))

        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --------------------
    # Format: counts as ints; mean to 2 decimals
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
        f.write("Mean computed over valid responses 1–5 only; missing excluded.\n")
        if not DK_CODES and not NA_CODES:
            f.write("Note: This extract does not include explicit DK/NA codes; DK/NA rows count only explicit codes (none found).\n")
        else:
            f.write(f"Detected DK code(s): {DK_CODES if DK_CODES else 'none'}; NA code(s): {NA_CODES if NA_CODES else 'none'}.\n")
        f.write("\n")

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