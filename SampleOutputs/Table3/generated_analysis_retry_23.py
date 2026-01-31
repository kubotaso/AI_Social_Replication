def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Resolve columns case-insensitively ---
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # --- Restrict to GSS 1993 ---
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # --- Table 3 genre variables (exact set/order) ---
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

    # --- Row labels (exact order) ---
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

    valid_vals = {1, 2, 3, 4, 5}

    def _blank_mask(series: pd.Series) -> pd.Series:
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            s = series.astype("string")
            return s.isna() | (s.str.strip() == "")
        return pd.Series(False, index=series.index)

    def _infer_dk_na_codes(df_1993: pd.DataFrame):
        """
        Infer numeric codes for DK and NA from the observed distribution of non-1..5 codes.
        We expect two dominant special codes used broadly across items.
        Common in GSS: 8='Don't know', 9='No answer'. But we infer from data.

        Strategy:
          1) For each item, find special numeric codes (numeric, not in 1..5).
          2) Count totals across all items.
          3) Prefer codes present in many items.
          4) Among candidates, pick the smallest two as (DK, NA) if both 8 and 9 are present;
             else pick the two most frequent and order by numeric value (smaller -> DK, larger -> NA).
        """
        total_counts = {}
        presence = {}

        for _, var_lower in genres:
            s = pd.to_numeric(df_1993[colmap[var_lower]], errors="coerce")
            specials = s[s.notna() & ~s.isin(list(valid_vals))]
            if specials.empty:
                continue
            vc = specials.value_counts(dropna=True)
            for code, cnt in vc.items():
                code_f = float(code)
                total_counts[code_f] = total_counts.get(code_f, 0) + int(cnt)
                presence[code_f] = presence.get(code_f, 0) + 1

        if not total_counts:
            return None, None, {"total_counts": {}, "presence": {}, "ranked": []}

        # Candidate codes: appear in at least 1/3 of items (fallback to all if too strict)
        min_items = max(1, int(np.ceil(len(genres) / 3)))
        candidates = [c for c in total_counts.keys() if presence.get(c, 0) >= min_items]
        if not candidates:
            candidates = list(total_counts.keys())

        # If 8 and 9 exist, lock to them (most consistent with GSS music batteries)
        has8 = 8.0 in total_counts
        has9 = 9.0 in total_counts
        if has8 and has9:
            dk_code = 8.0
            na_code = 9.0
        else:
            # Rank by total frequency (desc), tie by code value (asc)
            ranked = sorted(candidates, key=lambda c: (-total_counts[c], c))
            top2 = ranked[:2]
            if len(top2) == 0:
                dk_code, na_code = None, None
            elif len(top2) == 1:
                dk_code, na_code = top2[0], None
            else:
                # Order by numeric value: smaller -> DK, larger -> NA (avoids swapping)
                dk_code, na_code = (min(top2), max(top2))

        ranked_all = sorted(candidates, key=lambda c: (-total_counts[c], c))
        info = {"total_counts": total_counts, "presence": presence, "ranked": ranked_all}
        return dk_code, na_code, info

    dk_code, na_code, infer_info = _infer_dk_na_codes(df)

    def _compute_counts_and_mean(series: pd.Series, dk_code_val, na_code_val):
        """
        Counts for 1..5, DK, NA.
        DK/NA are counted from numeric special codes only; blanks/NaN are treated as NA-like
        but are not displayed in Table 3 as a separate row. To avoid contaminating DK vs NA,
        we assign NaN/blank to NA (No answer) rather than DK.
        """
        blank = _blank_mask(series)
        sn = pd.to_numeric(series, errors="coerce")

        c = {k: int((sn == k).sum()) for k in [1, 2, 3, 4, 5]}

        # numeric special codes
        specials_mask = sn.notna() & ~sn.isin(list(valid_vals))
        dk_cnt = int((sn == dk_code_val).sum()) if dk_code_val is not None else 0
        na_cnt = int((sn == na_code_val).sum()) if na_code_val is not None else 0

        # If inferred codes are missing/unavailable, fall back to standard GSS 8/9 if present in this item
        present_codes = set(pd.unique(sn[specials_mask]).tolist())
        if dk_code_val is None and 8.0 in present_codes:
            dk_cnt = int((sn == 8.0).sum())
        if na_code_val is None and 9.0 in present_codes:
            na_cnt = int((sn == 9.0).sum())

        # Any remaining special numeric codes (rare) are treated as NA (not DK) to avoid DK/NA swapping.
        accounted = set()
        if dk_code_val is not None:
            accounted.add(float(dk_code_val))
        if na_code_val is not None:
            accounted.add(float(na_code_val))
        leftovers = [code for code in present_codes if float(code) not in accounted]
        if leftovers:
            na_cnt += int(sn.isin(leftovers).sum())

        # NaN/blank: treat as "No answer" (missing) for display purposes
        # (Table 3 shows only DK and NA; untyped missing is closest to NA.)
        na_cnt += int((sn.isna() | blank).sum())

        mean_val = sn.where(sn.isin(list(valid_vals))).mean()
        return c[1], c[2], c[3], c[4], c[5], dk_cnt, na_cnt, mean_val

    # --- Build table with Attitude as first column (explicit row labels) ---
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk_cnt, na_cnt, mean_val = _compute_counts_and_mean(s, dk_code, na_code)

        table.loc["(1) Like very much", genre_label] = str(c1)
        table.loc["(2) Like it", genre_label] = str(c2)
        table.loc["(3) Mixed feelings", genre_label] = str(c3)
        table.loc["(4) Dislike it", genre_label] = str(c4)
        table.loc["(5) Dislike very much", genre_label] = str(c5)
        table.loc["(M) Don’t know much about it", genre_label] = str(dk_cnt)
        table.loc["(M) No answer", genre_label] = str(na_cnt)
        table.loc["Mean", genre_label] = "" if pd.isna(mean_val) else f"{float(mean_val):.2f}"

    # --- Save as human-readable text file with 3 panels (6 genres each) ---
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
    row_w = max(len(att_col), max(len(str(x)) for x in table[att_col])) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        tc = infer_info.get("total_counts", {})
        ranked = infer_info.get("ranked", [])
        if dk_code is not None or na_code is not None:
            f.write(
                "Inferred special numeric codes from data: "
                f"DK_code={dk_code}, NA_code={na_code}; "
                f"top_specials={[(c, tc.get(c, 0)) for c in ranked[:6]]}\n"
            )
        else:
            f.write("Inferred special numeric codes from data: none detected (DK/NA may be blank/NaN only).\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(table[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            header = _pad(att_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = _pad(table.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = table.loc[r, c]
                    line += _pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")

    return table