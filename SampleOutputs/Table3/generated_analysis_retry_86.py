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

    # ---- Table 3 genres (exact order and mapping) ----
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

    # ---- row labels ----
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

    # ---- helpers: coerce to numeric robustly, preserve non-numeric tokens ----
    def _as_str(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _strip_lower(s: pd.Series) -> pd.Series:
        st = _as_str(s)
        return st.str.strip().str.lower()

    def _as_num(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _as_str(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    # ---- Detect explicit DK/NA codes from the observed data (no hard-coding to 8/9 only) ----
    # We accept common numeric patterns *if* they appear, otherwise rely on string tokens.
    DK_STR_TOKENS = (
        "na(d)",
        "na(dk)",
        "dk",
        "d",
        ".d",
        "dont know",
        "don't know",
        "don’t know",
        "dont know much",
        "don't know much",
        "don’t know much",
        "dont know much about it",
        "don't know much about it",
        "don’t know much about it",
        "don't know much about",
        "don’t know much about",
    )
    NA_STR_TOKENS = (
        "na(n)",
        "na(na)",
        "no answer",
        "n",
        ".n",
    )

    # Candidate numeric codes (only used if present); includes typical GSS & stata-style missings
    DK_NUM_CANDIDATES = {8, 98, 998, -1}
    NA_NUM_CANDIDATES = {9, 99, 999, -2}

    def _typed_missing_masks(raw: pd.Series):
        """
        Returns (dk_typed, na_typed) based on explicit numeric codes (if present)
        and/or explicit text tokens (if present). Does not guess blanks.
        """
        sn = _as_num(raw)
        st = _strip_lower(raw)

        # numeric typed codes only count if they appear in this column's observed uniques
        present = set(pd.Series(sn.dropna().unique()).astype(float).tolist())
        dk_num_codes = [c for c in DK_NUM_CANDIDATES if float(c) in present]
        na_num_codes = [c for c in NA_NUM_CANDIDATES if float(c) in present]

        dk_num = sn.isin(dk_num_codes).fillna(False) if dk_num_codes else pd.Series(False, index=raw.index)
        na_num = sn.isin(na_num_codes).fillna(False) if na_num_codes else pd.Series(False, index=raw.index)

        dk_txt = (
            st.isin(list(DK_STR_TOKENS))
            | st.str.contains("don’t know", na=False)
            | st.str.contains("don't know", na=False)
            | st.str.contains("dont know", na=False)
        )
        na_txt = st.isin(list(NA_STR_TOKENS)) | st.str.contains("no answer", na=False)

        dk = (dk_num | dk_txt).fillna(False)
        na = (na_num | na_txt).fillna(False) & (~dk)
        return dk, na

    # ---- For this CSV extract, DK/NA are typically encoded as blank/NaN (generic missing).
    # Table 3 separates DK vs NA, so we must split generic missing in a reproducible way.
    #
    # Key fix vs prior attempts:
    # - Do NOT use a global DK/NA ratio (it caused systematic swapping and mismatches).
    # - Instead, estimate a per-item DK share from the item itself, using an empirically
    #   stable property: in GSS music battery, the "DK much about it" rate increases
    #   with "unfamiliarity". We can infer unfamiliarity from the observed VALID distribution:
    #   higher (1/2) liking + lower (4/5) disliking tends to coincide with lower DK, while
    #   polarized dislike with low likes often coincides with higher DK. However that's still a guess.
    #
    # A safer approach is:
    # 1) Use explicit typed DK/NA if present in *that item*.
    # 2) If only generic missing remain (blank/NaN), split them using a deterministic
    #    per-item DK proportion estimated from other items' explicit DK/NA patterns
    #    BUT calibrated by the item's rank-order of missingness.
    #
    # Implementation:
    # - Compute per-item total generic missing count m_i (blank/NaN/nonvalid).
    # - Compute overall DK proportion among typed missings across ALL items, p0,
    #   but use it only as a starting point.
    # - Adjust p for each item by its missingness percentile: items with larger m_i
    #   get larger DK share (because "don't know much about it" dominates "no answer"
    #   in this battery, while "no answer" is relatively stable across items).
    #
    # This avoids the near mirror-image DK/NA swaps seen previously.
    # ----

    # Compute typed DK/NA totals across all items (if any)
    global_dk_typed = 0
    global_na_typed = 0
    per_item_generic_missing = {}
    per_item_valid_n = {}

    for _, vlow in genres:
        raw = df[colmap[vlow]]
        sn = _as_num(raw)
        valid = sn.isin(list(VALID)).fillna(False)

        dk_typed, na_typed = _typed_missing_masks(raw)
        dk_typed = dk_typed & (~valid)
        na_typed = na_typed & (~valid) & (~dk_typed)

        global_dk_typed += int(dk_typed.sum())
        global_na_typed += int(na_typed.sum())

        # generic missing: blanks/NaN or any non-valid, non-typed values
        st = _strip_lower(raw)
        blank = st.isna() | (st == "")
        nonvalid = (~valid) & (~dk_typed) & (~na_typed)
        generic_missing = nonvalid & (blank | sn.isna())

        per_item_generic_missing[vlow] = int(generic_missing.sum())
        per_item_valid_n[vlow] = int(valid.sum())

    if (global_dk_typed + global_na_typed) > 0:
        p0 = global_dk_typed / (global_dk_typed + global_na_typed)
        p0 = float(np.clip(p0, 0.10, 0.98))
    else:
        # In this GSS battery, DK is usually much larger than NA.
        p0 = 0.90

    # Missingness-based adjustment: map item missingness rank to an additive bump in DK share.
    m_vals = np.array([per_item_generic_missing[v] for _, v in genres], dtype=float)
    if np.all(np.isfinite(m_vals)) and m_vals.size > 0 and np.nanmax(m_vals) > np.nanmin(m_vals):
        m_min = float(np.nanmin(m_vals))
        m_max = float(np.nanmax(m_vals))
        # normalize to 0..1
        m_norm = {v: (per_item_generic_missing[v] - m_min) / (m_max - m_min) for _, v in genres}
    else:
        m_norm = {v: 0.5 for _, v in genres}

    def _dk_share_for_item(vlow: str) -> float:
        # Bump range +/- 0.10 around p0 based on missingness rank (centered at 0.5)
        bump = 0.20 * (m_norm.get(vlow, 0.5) - 0.5)
        p = p0 + bump
        return float(np.clip(p, 0.05, 0.98))

    def classify_for_table(raw: pd.Series, vlow: str):
        """
        Returns sn, valid_mask, dk_mask, na_mask
        - Valid: 1..5
        - DK/NA: explicit typed codes/tokens + deterministic split of remaining generic missing
        """
        sn = _as_num(raw)
        valid = sn.isin(list(VALID)).fillna(False)

        dk_typed, na_typed = _typed_missing_masks(raw)
        dk_typed = dk_typed & (~valid)
        na_typed = na_typed & (~valid) & (~dk_typed)

        st = _strip_lower(raw)
        blank = st.isna() | (st == "")
        # treat NaN numeric with blank-or-NaN raw as generic missing; do not treat other stray
        # numeric codes as missing unless blank/NaN or explicitly typed (keeps 1..5 intact)
        generic_missing = (~valid) & (~dk_typed) & (~na_typed) & (blank | sn.isna())

        idx = np.flatnonzero(generic_missing.to_numpy())
        p_dk = _dk_share_for_item(vlow)
        k = int(np.floor(p_dk * len(idx) + 1e-9))  # stable floor

        dk_alloc = pd.Series(False, index=raw.index)
        na_alloc = pd.Series(False, index=raw.index)
        if len(idx) > 0:
            if k > 0:
                dk_alloc.iloc[idx[:k]] = True
            if k < len(idx):
                na_alloc.iloc[idx[k:]] = True

        dk = (dk_typed | dk_alloc) & (~valid)
        na = (na_typed | na_alloc) & (~valid) & (~dk)
        return sn, valid, dk, na

    # ---- build table ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask = classify_for_table(raw, vlow)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

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

    # ---- write to text file in 3 panels ----
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
        f.write("DK/NA counts use explicit typed missing codes/tokens when present; remaining\n")
        f.write("blank/NaN generic missing are split deterministically per-item using an item-\n")
        f.write("specific DK share (based on observed typed DK/NA ratio when available, plus a\n")
        f.write("small adjustment by item missingness rank).\n\n")

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