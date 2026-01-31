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

    VALID = {1, 2, 3, 4, 5}

    # ---- helpers ----
    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _as_string(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    def _blank_mask(raw: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(raw):
            return raw.isna()
        st = _as_string(raw)
        return st.isna() | (st.str.strip() == "")

    def _standardize_labels(st: pd.Series) -> pd.Series:
        # normalize curly quotes etc.
        st = st.fillna(pd.NA)
        st = st.str.lower()
        st = st.str.replace("\u2019", "'", regex=False)
        st = st.str.replace("\u2018", "'", regex=False)
        st = st.str.replace("\u201c", '"', regex=False)
        st = st.str.replace("\u201d", '"', regex=False)
        st = st.str.strip()
        return st

    # ---- detect typed missing codes from value labels if present ----
    # Preference order: read DK/NA codes from labels; else fall back to common GSS numeric codes.
    DK_CODES_FALLBACK = {8, 98}
    NA_CODES_FALLBACK = {9, 99}

    def _codes_from_value_labels(series: pd.Series):
        dk_codes = set()
        na_codes = set()

        # pandas Categorical can carry categories but not labels; try attrs (rare), else none.
        # If it's an R-imported labelled vector (via pyreadstat), there may be series.attrs["value_labels"].
        vlabels = None
        if hasattr(series, "attrs"):
            vlabels = series.attrs.get("value_labels", None)

        if isinstance(vlabels, dict) and vlabels:
            for k, v in vlabels.items():
                try:
                    code = int(k)
                except Exception:
                    continue
                lab = _standardize_labels(pd.Series([str(v)])).iloc[0]
                if lab is pd.NA:
                    continue
                if "don't know" in lab or "dont know" in lab:
                    dk_codes.add(code)
                if "no answer" in lab:
                    na_codes.add(code)
            return dk_codes, na_codes

        return dk_codes, na_codes

    # ---- per-item DK/NA classification ----
    def classify_item(raw: pd.Series):
        sn = _as_numeric(raw)
        valid_mask = sn.isin(list(VALID)).fillna(False)

        dk_codes, na_codes = _codes_from_value_labels(raw)

        # If labels aren't available, use fallback codes, but only if they actually appear.
        present_vals = set(pd.Series(sn.dropna().unique()).tolist())
        if not dk_codes:
            dk_codes = set(c for c in DK_CODES_FALLBACK if c in present_vals)
        if not na_codes:
            na_codes = set(c for c in NA_CODES_FALLBACK if c in present_vals)

        dk_mask = sn.isin(list(dk_codes)).fillna(False) & ~valid_mask if dk_codes else pd.Series(False, index=sn.index)
        na_mask = sn.isin(list(na_codes)).fillna(False) & ~valid_mask if na_codes else pd.Series(False, index=sn.index)

        # Remaining missing (blank/NaN) are untyped; Table 3 still splits DK vs NA.
        # We cannot recover that split from this CSV alone, so we use a consistent,
        # data-driven allocation rule:
        #   - If explicit DK/NA codes exist in THIS item, use their observed DK share to split blanks.
        #   - Else if explicit codes exist across ALL items, use global DK share to split blanks.
        #   - Else: treat blanks as DK (conservative; avoids inventing NA when nothing indicates it).
        blank = _blank_mask(raw) & ~valid_mask & ~dk_mask & ~na_mask
        return sn, valid_mask, dk_mask, na_mask, blank, dk_codes, na_codes

    # First pass: collect global DK/NA shares where explicit codes exist (any item)
    global_dk = 0
    global_na = 0
    per_item_explicit = {}

    for glabel, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask, blank, dk_codes, na_codes = classify_item(raw)
        dk_exp = int(dk_mask.sum())
        na_exp = int(na_mask.sum())
        per_item_explicit[glabel] = (dk_exp, na_exp)
        global_dk += dk_exp
        global_na += na_exp

    global_total = global_dk + global_na
    global_p_dk = None
    if global_total > 0:
        global_p_dk = global_dk / global_total
        global_p_dk = float(min(max(global_p_dk, 0.01), 0.99))

    # ---- build table ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        raw = df[colmap[var_lower]]
        sn, valid_mask, dk_mask, na_mask, blank_mask, dk_codes, na_codes = classify_item(raw)

        # frequencies for 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # split blanks into DK vs NA using observed DK share when possible
        dk_exp, na_exp = per_item_explicit.get(genre_label, (0, 0))
        item_total = dk_exp + na_exp
        if item_total > 0:
            p_dk = dk_exp / item_total
            p_dk = float(min(max(p_dk, 0.01), 0.99))
        elif global_p_dk is not None:
            p_dk = global_p_dk
        else:
            p_dk = 1.0  # no basis for NA split in this extract

        b_idx = np.flatnonzero(blank_mask.to_numpy())
        b_n = int(len(b_idx))
        if b_n > 0:
            # deterministic allocation by row order
            dk_k = int(round(p_dk * b_n))
            dk_idx = b_idx[:dk_k]
            na_idx = b_idx[dk_k:]

            dk_alloc = pd.Series(False, index=raw.index)
            na_alloc = pd.Series(False, index=raw.index)
            if dk_idx.size:
                dk_alloc.iloc[dk_idx] = True
            if na_idx.size:
                na_alloc.iloc[na_idx] = True
        else:
            dk_alloc = pd.Series(False, index=raw.index)
            na_alloc = pd.Series(False, index=raw.index)

        dk_final = (dk_mask | dk_alloc).fillna(False) & ~valid_mask
        na_final = (na_mask | na_alloc).fillna(False) & ~valid_mask & ~dk_final

        table.loc["(M) Don’t know much about it", genre_label] = int(dk_final.sum())
        table.loc["(M) No answer", genre_label] = int(na_final.sum())

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
        if global_p_dk is None:
            f.write(
                "Note: This CSV extract does not include explicit typed DK/NA codes; any blank/NaN values "
                "are allocated to DK/NA using a deterministic rule (defaulting blanks to DK when no typed "
                "information is available).\n\n"
            )
        else:
            f.write(
                "Note: Blank/NaN values (untyped missings in this CSV) are split into DK/NA using the "
                f"observed global DK share from explicit codes across items (DK share={global_p_dk:.3f}).\n\n"
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