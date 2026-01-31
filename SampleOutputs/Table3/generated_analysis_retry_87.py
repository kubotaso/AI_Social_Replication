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
    # Table 3 genre variables (exact order and mapping)
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
    # Helpers
    # --------------------
    def as_str(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def strip_lower(s: pd.Series) -> pd.Series:
        st = as_str(s)
        return st.str.strip().str.lower()

    def as_num(s: pd.Series) -> pd.Series:
        # robust numeric coercion; blanks become NA
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = as_str(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    # Explicit typed-missing detection (numeric codes and/or tokens), if present
    # NOTE: In many CSV extracts these will not exist; then DK/NA will be unobservable.
    DK_STR_TOKENS = {
        "na(d)", "na(dk)", "dk", "d", ".d",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
    }
    NA_STR_TOKENS = {"na(n)", "na(na)", "no answer", "n", ".n"}

    DK_NUM_CANDIDATES = {8, 98, 998, -1}
    NA_NUM_CANDIDATES = {9, 99, 999, -2}

    def typed_missing_masks(raw: pd.Series):
        sn = as_num(raw)
        st = strip_lower(raw)

        present = set(pd.Series(sn.dropna().unique()).astype(float).tolist())
        dk_num_codes = [c for c in DK_NUM_CANDIDATES if float(c) in present]
        na_num_codes = [c for c in NA_NUM_CANDIDATES if float(c) in present]

        dk_num = sn.isin(dk_num_codes).fillna(False) if dk_num_codes else pd.Series(False, index=raw.index)
        na_num = sn.isin(na_num_codes).fillna(False) if na_num_codes else pd.Series(False, index=raw.index)

        dk_txt = st.isin(DK_STR_TOKENS) | st.str.contains("don’t know", na=False) | st.str.contains("don't know", na=False) | st.str.contains("dont know", na=False)
        na_txt = st.isin(NA_STR_TOKENS) | st.str.contains("no answer", na=False)

        dk = (dk_num | dk_txt).fillna(False)
        na = (na_num | na_txt).fillna(False) & (~dk)
        return dk, na

    def classify_item(raw: pd.Series):
        """
        Returns:
          sn: numeric series
          valid: mask for 1..5
          dk: mask for "don't know much about it" (typed if observable)
          na: mask for "no answer" (typed if observable)
          other_missing: any remaining nonvalid/missing not classified into DK or NA
        Important: We do NOT guess/split blank missing into DK vs NA.
        """
        sn = as_num(raw)
        valid = sn.isin(list(VALID)).fillna(False)

        dk_typed, na_typed = typed_missing_masks(raw)
        dk = (dk_typed & (~valid)).fillna(False)
        na = (na_typed & (~valid) & (~dk)).fillna(False)

        st = strip_lower(raw)
        blank = st.isna() | (st == "")
        other_missing = (~valid) & (~dk) & (~na) & (blank | sn.isna())
        other_missing = other_missing.fillna(False)

        return sn, valid, dk, na, other_missing

    # --------------------
    # Build table
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    # Collect diagnostics for DK/NA observability
    dkna_observed_any = False
    other_missing_totals = {}

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn, valid_mask, dk_mask, na_mask, other_missing = classify_item(raw)

        other_missing_totals[genre_label] = int(other_missing.sum())
        if int(dk_mask.sum()) > 0 or int(na_mask.sum()) > 0:
            dkna_observed_any = True

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
    # Format for display: counts as ints; mean to 2 decimals
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

    # compute column widths per panel
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write("Frequencies are counts only (no percentages).\n")
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n\n")

        # Transparency about DK/NA in the extract (cannot be inferred from blanks)
        if not dkna_observed_any:
            f.write("NOTE: This CSV extract does not contain explicit DK/NA codes/tokens for the music items.\n")
            f.write("Blank/NaN responses are treated as missing but cannot be reliably split into:\n")
            f.write("  (M) Don’t know much about it vs (M) No answer\n")
            f.write("Therefore DK/NA counts may be 0 even when item nonresponse exists.\n\n")

        # Optional: show per-genre unclassified missing counts (blank/NaN not typed DK/NA)
        if any(v > 0 for v in other_missing_totals.values()):
            f.write("Unclassified missing (blank/NaN not explicitly coded as DK/NA) by genre:\n")
            for genre_label, cnt in other_missing_totals.items():
                f.write(f"  {genre_label}: {cnt}\n")
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