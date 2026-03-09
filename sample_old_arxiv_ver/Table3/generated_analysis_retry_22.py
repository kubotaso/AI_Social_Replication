def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 (case-insensitive column resolution) ---
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    df = df.loc[pd.to_numeric(df[colmap["year"]], errors="coerce") == 1993].copy()

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

    def _special_code_mask(sn: pd.Series):
        """Return mask of special (non 1-5) numeric codes, excluding NaN."""
        return sn.notna() & ~sn.isin(list(valid_vals))

    def infer_dk_na_codes(df_1993: pd.DataFrame):
        """
        Infer the numeric codes for:
          DK = 'Don't know much about it'
          NA = 'No answer'
        from the data across all 18 items.

        Key idea: the DK code is typically much more frequent than the NA code.
        We pick the two most frequent special codes overall, and set:
          DK = most frequent, NA = second most frequent.
        """
        total_counts = {}
        presence = {}

        for _, var_lower in genres:
            s = pd.to_numeric(df_1993[colmap[var_lower]], errors="coerce")
            specials = s[_special_code_mask(s)]
            if specials.empty:
                continue
            vc = specials.value_counts(dropna=True)
            for code, cnt in vc.items():
                code = float(code)
                total_counts[code] = total_counts.get(code, 0) + int(cnt)
                presence[code] = presence.get(code, 0) + 1

        if not total_counts:
            return None, None, {}

        # Keep codes that appear broadly (avoid one-off odd codes); if too strict, fall back.
        min_items = max(1, int(np.ceil(len(genres) * 0.5)))
        candidates = [c for c in total_counts.keys() if presence.get(c, 0) >= min_items]
        if not candidates:
            candidates = list(total_counts.keys())

        ranked = sorted(candidates, key=lambda c: (-total_counts[c], c))
        dk_code = ranked[0] if len(ranked) >= 1 else None
        na_code = ranked[1] if len(ranked) >= 2 else None

        info = {"total_counts": total_counts, "presence": presence, "ranked": ranked}
        return dk_code, na_code, info

    dk_code, na_code, infer_info = infer_dk_na_codes(df)

    def compute_counts_and_mean(series: pd.Series, dk_code_val, na_code_val):
        sn = pd.to_numeric(series, errors="coerce")

        c1 = int((sn == 1).sum())
        c2 = int((sn == 2).sum())
        c3 = int((sn == 3).sum())
        c4 = int((sn == 4).sum())
        c5 = int((sn == 5).sum())

        specials_mask = _special_code_mask(sn)
        specials = sn[specials_mask]

        # DK/NA counts based on inferred numeric codes
        dk_cnt = int((sn == dk_code_val).sum()) if dk_code_val is not None else 0
        na_cnt = int((sn == na_code_val).sum()) if na_code_val is not None else 0

        # If only one code inferred, assign all other special codes to DK (not NA),
        # because DK dominates and the published table shows DK >> NA.
        if specials_mask.any():
            if dk_code_val is not None and na_code_val is None:
                dk_cnt = int((specials == dk_code_val).sum()) + int((specials != dk_code_val).sum())
                na_cnt = 0
            elif dk_code_val is None and na_code_val is not None:
                # unlikely, but if it happens: treat everything that's not NA as DK
                na_cnt = int((specials == na_code_val).sum())
                dk_cnt = int((specials != na_code_val).sum())
            elif dk_code_val is None and na_code_val is None:
                dk_cnt = int(specials.shape[0])
                na_cnt = 0

        mean_val = sn.where(sn.isin(list(valid_vals))).mean()
        return c1, c2, c3, c4, c5, dk_cnt, na_cnt, mean_val

    # --- Build output table: include Attitude as first column (requested fix) ---
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk_cnt, na_cnt, mean_val = compute_counts_and_mean(s, dk_code, na_code)

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
        ranked = infer_info.get("ranked", [])
        tc = infer_info.get("total_counts", {})
        if dk_code is not None or na_code is not None:
            f.write(
                "Missing-code inference (computed from data): "
                f"DK_code={dk_code}, NA_code={na_code}; "
                f"special_code_ranking={[(c, tc.get(c, 0)) for c in ranked[:6]]}\n"
            )
        else:
            f.write("Missing-code inference (computed from data): no special numeric codes detected.\n")

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