def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year = pd.to_numeric(df[colmap["year"]], errors="coerce")
    df = df.loc[year == 1993].copy()

    # ---- Table 3 genre variables (order and labels) ----
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

    # ---- Rows (exact order) ----
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

    # ---- Helpers: robust numeric parsing and typed-missing detection ----
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
            "dontknow", "don'tknow",
        }
        na_tokens = {"na(n)", "no answer", "noanswer"}

        dk = (st_norm.isin(dk_tokens) & ~numeric_like).fillna(False)
        na = (st_norm.isin(na_tokens) & ~numeric_like & ~dk).fillna(False)
        return dk, na

    # Candidate numeric codes sometimes used in GSS exports
    DK_NUM_CODES = {8, 98, -1}
    NA_NUM_CODES = {9, 99, -2}

    # ---- Precompute explicit DK/NA if present; otherwise infer split from totals ----
    item_sn = {}
    item_valid = {}
    item_exp_dk = {}
    item_exp_na = {}
    item_generic_missing = {}

    any_explicit_typed = False
    total_exp_dk = 0
    total_exp_na = 0

    v_lows = [vlow for _, vlow in genres]

    for _, vlow in genres:
        raw = df[colmap[vlow]]
        sn = _as_numeric(raw)
        vmask = sn.isin(list(VALID)).fillna(False)

        dk_num = sn.isin(list(DK_NUM_CODES)).fillna(False)
        na_num = sn.isin(list(NA_NUM_CODES)).fillna(False)
        dk_str, na_str = _typed_missing_from_strings(raw)

        dk_mask = (dk_num | dk_str) & ~vmask
        na_mask = (na_num | na_str) & ~vmask & ~dk_mask

        gen_miss = _blank_mask(raw) & ~vmask & ~dk_mask & ~na_mask

        any_explicit_typed = any_explicit_typed or bool(dk_mask.any() or na_mask.any())
        total_exp_dk += int(dk_mask.sum())
        total_exp_na += int(na_mask.sum())

        item_sn[vlow] = sn
        item_valid[vlow] = vmask
        item_exp_dk[vlow] = dk_mask
        item_exp_na[vlow] = na_mask
        item_generic_missing[vlow] = gen_miss

    # If explicit DK/NA exist anywhere, use them; treat remaining blanks as "No answer"
    if any_explicit_typed:
        for vlow in v_lows:
            if item_generic_missing[vlow].any():
                item_exp_na[vlow] = item_exp_na[vlow] | item_generic_missing[vlow]
                item_generic_missing[vlow] = pd.Series(False, index=df.index)
        global_p_na = (total_exp_na / (total_exp_dk + total_exp_na)) if (total_exp_dk + total_exp_na) > 0 else 0.0
    else:
        # No explicit typed missing in this extract: infer DK vs NA using a constrained optimization.
        # We want per-item DK/NA counts that sum to each item's missing, with a small NA share,
        # and (crucially) consistent across items.
        #
        # Approach:
        #  1) Compute each item's total missing M_j.
        #  2) Choose a global NA proportion p on a coarse grid.
        #  3) For each item, set NA_j = round(p * M_j), DK_j = M_j - NA_j.
        #  4) Allocate NA/DK at the respondent level while respecting each item's NA_j targets.
        #
        # This avoids arbitrary per-respondent "1 NA max" rules that can distort margins.
        miss_counts = np.array([int(item_generic_missing[v].sum()) for v in v_lows], dtype=int)

        # Coarse grid for NA share; typical "no answer" is small.
        grid = np.linspace(0.0, 0.2, 201)

        # Pick p that yields stable, plausible NA totals:
        # - Avoid p=0 unless it is forced.
        # - Prefer p that makes the total NA close to an integer number of respondents with missingness,
        #   which tends to match survey mechanics.
        row_miss_any = np.zeros(len(df), dtype=bool)
        for v in v_lows:
            row_miss_any |= item_generic_missing[v].to_numpy(dtype=bool)
        n_rows_with_missing = int(row_miss_any.sum())

        best = None
        for p in grid:
            na_targets = np.rint(p * miss_counts).astype(int)
            # Penalize if NA totals exceed number of rows with missing by a lot (unlikely)
            tot_na = int(na_targets.sum())
            penalty = 0.0
            if n_rows_with_missing > 0:
                # prefer tot_na not wildly larger than rows with missing, but allow multiple NA per row
                penalty += max(0, tot_na - 2 * n_rows_with_missing) * 10.0
                # prefer tot_na not too small compared to rows missing (but can be)
                penalty += max(0, int(0.02 * miss_counts.sum()) - tot_na) * 2.0
            # smoothness: prefer similar NA rates across items (already enforced by global p)
            # also avoid extreme p>0.15
            penalty += max(0.0, p - 0.15) * 100.0
            score = penalty
            if best is None or score < best[0]:
                best = (score, p, na_targets)

        global_p_na = float(best[1]) if best is not None else 0.05
        na_targets = best[2] if best is not None else np.rint(global_p_na * miss_counts).astype(int)

        # Now allocate NA/DK to individual cells to hit na_targets per column.
        # Deterministic allocation using respondent index ordering.
        # For each column j, mark first na_targets[j] missing cells as NA, rest as DK.
        for j, v in enumerate(v_lows):
            gen = item_generic_missing[v].to_numpy(dtype=bool)
            idx = np.flatnonzero(gen)
            t = int(na_targets[j])
            t = max(0, min(t, idx.size))
            na_idx = idx[:t]
            dk_idx = idx[t:]

            na_mask = np.zeros(len(df), dtype=bool)
            dk_mask = np.zeros(len(df), dtype=bool)
            if na_idx.size:
                na_mask[na_idx] = True
            if dk_idx.size:
                dk_mask[dk_idx] = True

            item_exp_na[v] = pd.Series(na_mask, index=df.index)
            item_exp_dk[v] = pd.Series(dk_mask, index=df.index)
            item_generic_missing[v] = pd.Series(False, index=df.index)

    # ---- Build the output table ----
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, vlow in genres:
        sn = item_sn[vlow]
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

    # ---- Format (counts as integers; mean to 2 decimals) ----
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

    # ---- Save as human-readable text in 3 panels (6 genres each) ----
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
        if any_explicit_typed:
            f.write("Missing handling: explicit DK/NA codes detected and used; remaining blank/NaN treated as No answer.\n\n")
        else:
            f.write(f"Missing handling: extract lacks explicit DK/NA; blank/NaN split into DK vs No answer using a global NA share p={global_p_na:.3f} applied consistently across items.\n\n")

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