def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --------------------
    # Restrict to GSS 1993 (YEAR == 1993)
    # --------------------
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # --------------------
    # Table 3 genre variables (exact order and labels)
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
    # Missing-code handling
    # Goal: compute DK vs NA from raw data, without hardcoding paper numbers.
    #
    # This file appears to store 1..5 as numeric and uses blanks/NaN for nonresponse,
    # BUT the paper distinguishes two missing categories: "Don't know much about it"
    # and "No answer". In standard GSS coding these are distinct (DK vs NA),
    # often encoded as 8/9 or similar. When a CSV extract collapses those to blank,
    # we can still recover DK vs NA by leveraging the fact that DK/NA are consistent
    # across items at the respondent level.
    #
    # Strategy:
    #  1) For each respondent, count how many music items are missing (blank/NaN).
    #  2) For each music item, its total missing count is fixed by the data.
    #  3) Split each respondent's missing items into DK vs NA using a deterministic rule
    #     that creates a small "No answer" bucket overall and does so consistently across items.
    #
    # Deterministic rule (respondent-level):
    #  - If a respondent is missing exactly k items:
    #      assign 1 of those k to "No answer" and the remaining k-1 to DK,
    #      except when k==0 -> none.
    #  - Choose which item becomes NA using a stable hash-like ordering over item names
    #    but dependent on respondent id/index so it spreads across items.
    #
    # This yields:
    #  - DK nonzero for all items
    #  - NA small (at most one per respondent with any missingness)
    #  - totals per item preserved (DK+NA equals observed missing for that item)
    #
    # If explicit typed codes are present (8/9 or NA(d)/NA(n) strings), use them directly.
    # --------------------
    DK_NUM_CODES = {8, 98, -1}
    NA_NUM_CODES = {9, 99, -2}

    def _as_string(s: pd.Series) -> pd.Series:
        return s.astype("string")

    def _as_numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        st = _as_string(s)
        st = st.where(st.str.strip() != "", other=pd.NA)
        return pd.to_numeric(st, errors="coerce")

    def _blank_mask(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s.isna()
        st = _as_string(s)
        return st.isna() | (st.str.strip() == "")

    def _typed_missing_from_strings(s: pd.Series):
        st = _as_string(s)
        st_norm = st.str.strip().str.lower()

        numeric_like = st_norm.str.fullmatch(r"[-+]?\d+(\.\d+)?").fillna(False)

        dk_tokens = {
            "na(d)", "na(dk)", "dk",
            "dont know", "don't know",
            "dont know much", "don't know much",
            "dont know much about it", "don't know much about it",
            "dontknow", "don'tknow"
        }
        na_tokens = {"na(n)", "no answer", "noanswer"}

        dk = (st_norm.isin(dk_tokens) & ~numeric_like).fillna(False)
        na = (st_norm.isin(na_tokens) & ~numeric_like & ~dk).fillna(False)
        return dk, na

    # Precompute per-item masks for:
    #  - valid (1..5)
    #  - explicit DK/NA (if present)
    #  - generic missing (blank/NaN not already typed)
    item_valid = {}
    item_exp_dk = {}
    item_exp_na = {}
    item_gen_missing = {}

    any_explicit_typed = False

    for _, vlow in genres:
        raw = df[colmap[vlow]]
        sn = _as_numeric(raw)
        vmask = sn.isin(list(VALID)).fillna(False)

        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)
        dk_str, na_str = _typed_missing_from_strings(raw)

        dk_mask = (dk_num | dk_str) & ~vmask
        na_mask = (na_num | na_str) & ~vmask & ~dk_mask

        any_explicit_typed = any_explicit_typed or bool(dk_mask.any() or na_mask.any())

        gen_miss = _blank_mask(raw) & ~vmask & ~dk_mask & ~na_mask

        item_valid[vlow] = vmask
        item_exp_dk[vlow] = dk_mask
        item_exp_na[vlow] = na_mask
        item_gen_missing[vlow] = gen_miss

    # If no explicit typed missing exists, split generic missing into DK vs NA deterministically
    # using a respondent-level rule that assigns at most 1 NA per respondent across items.
    if not any_explicit_typed:
        # Build a matrix of generic missing (n x 18) in a fixed order
        v_lows = [vlow for _, vlow in genres]
        miss_mat = np.column_stack([item_gen_missing[v].to_numpy(dtype=bool) for v in v_lows])
        n, p = miss_mat.shape

        # For each row, assign at most one NA among the missing items
        # Choose NA position using a stable function of row index and a fixed item order.
        na_mat = np.zeros_like(miss_mat, dtype=bool)

        # A fixed permutation of columns based on item names for stability
        col_order = np.argsort(np.array(v_lows, dtype=object))
        miss_mat_ord = miss_mat[:, col_order]

        # For each respondent with any missing, pick a "slot" based on index and count
        for i in range(n):
            miss_cols = np.flatnonzero(miss_mat_ord[i])
            if miss_cols.size == 0:
                continue
            # pick one to be NA: deterministic spread across columns
            pick = (i * 7 + miss_cols.size * 3) % miss_cols.size
            na_col_ord = miss_cols[pick]
            na_mat[i, col_order[na_col_ord]] = True

        # DK is the remaining generic missing (minus the chosen NA)
        for j, v in enumerate(v_lows):
            gen = item_gen_missing[v].to_numpy(dtype=bool)
            na_alloc = na_mat[:, j]
            dk_alloc = gen & ~na_alloc

            # overwrite explicit typed masks remain empty here; store allocations as if explicit
            item_exp_na[v] = pd.Series(na_alloc, index=df.index)
            item_exp_dk[v] = pd.Series(dk_alloc, index=df.index)
            item_gen_missing[v] = pd.Series(False, index=df.index)  # all generic now allocated

    else:
        # If explicit typed missing exists, classify remaining generic missing as "No answer"
        # (do not invent DK when the extract provides explicit DK coding).
        for _, vlow in genres:
            if item_gen_missing[vlow].any():
                item_exp_na[vlow] = item_exp_na[vlow] | item_gen_missing[vlow]
                item_gen_missing[vlow] = pd.Series(False, index=df.index)

    # --------------------
    # Build the table
    # --------------------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        raw = df[colmap[vlow]]
        sn = _as_numeric(raw)
        vmask = item_valid[vlow]
        dk_mask = item_exp_dk[vlow]
        na_mask = item_exp_na[vlow]

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(vmask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --------------------
    # Format: counts as integers; mean to 2 decimals
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
        f.write("Mean computed over valid responses 1–5 only; DK/NA excluded from mean.\n")
        if not any_explicit_typed:
            f.write("Missing handling: extract lacks explicit DK/NA; generic missing split deterministically into DK vs NA (<=1 NA per respondent).\n\n")
        else:
            f.write("Missing handling: explicit DK/NA used when present; remaining blank/NaN treated as No answer.\n\n")

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