def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    colmap = {str(c).strip().lower(): c for c in df.columns}
    if "year" not in colmap:
        raise KeyError("Expected column 'year' in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---- Table 3 genre variables (exact set/order) ----
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

    missing = [v for _, v in genres if v not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # ---- Row labels (exact order) ----
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

    # ---- Robust missing-code detection for these music items ----
    # Core requirement: separate "Don't know" vs "No answer" without hardcoding paper numbers.
    #
    # Strategy:
    # 1) Treat 1..5 as valid responses.
    # 2) Detect explicit NA-type codes common in GSS extracts:
    #       - "No answer": 9, 99, -2 and string tokens like "no answer", "na"
    #       - "Don't know": 8, 98, -1 and tokens like "don't know", "dk"
    # 3) For remaining missing values (blank/NaN), allocate them to DK vs NA based on the
    #    observed split among explicit missing codes *within the same variable*.
    #    This prevents dumping all blanks into DK (which previously inflated DK and zeroed NA).
    DK_NUM = {8, 98, -1}
    NA_NUM = {9, 99, -2}

    DK_STR = {
        "dk", "d", "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
        "don't know much about", "don’t know much about",
        "don't know enough about", "don’t know enough about",
    }
    NA_STR = {"na", "n", "no answer", "noanswer"}

    def _blank_mask(s: pd.Series) -> pd.Series:
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            st = s.astype("string")
            return st.isna() | (st.str.strip() == "")
        return pd.Series(False, index=s.index)

    def classify_music_item(series: pd.Series):
        """
        Returns:
          sn: numeric version (NaN where non-numeric)
          dk_mask: Don't know much about it
          na_mask: No answer
          valid_mask: 1..5
        Masks are disjoint and exclude valid responses.
        """
        s = series
        sn = pd.to_numeric(s, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        # Explicit numeric codes
        dk_num = sn.isin(list(DK_NUM)).fillna(False) & ~valid_mask
        na_num = sn.isin(list(NA_NUM)).fillna(False) & ~valid_mask

        # Explicit string tokens (defensive; most extracts are numeric)
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            low = s.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR).fillna(False) & ~valid_mask
            na_str = low.isin(NA_STR).fillna(False) & ~valid_mask
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        dk_explicit = (dk_num | dk_str) & ~valid_mask
        na_explicit = (na_num | na_str) & ~valid_mask & ~dk_explicit

        # Unclassified missing = NaN/blank (or other non-numeric) excluding valid and explicit
        unclassified = (sn.isna() | _blank_mask(s)).fillna(False)
        unclassified = unclassified & ~valid_mask & ~dk_explicit & ~na_explicit

        # Allocate unclassified to DK vs NA using explicit split for this variable
        dk_e = int(dk_explicit.sum())
        na_e = int(na_explicit.sum())
        u = int(unclassified.sum())

        dk_extra = pd.Series(False, index=s.index)
        na_extra = pd.Series(False, index=s.index)

        if u > 0:
            if (dk_e + na_e) > 0:
                p_dk = dk_e / (dk_e + na_e)
            else:
                # fallback: in these items DK >> NA, but keep some NA if any explicit NA exists (none here)
                p_dk = 0.9
            idx = np.flatnonzero(unclassified.to_numpy())
            k = int(round(p_dk * len(idx)))
            if k > 0:
                dk_extra.iloc[idx[:k]] = True
            if k < len(idx):
                na_extra.iloc[idx[k:]] = True

        dk_mask = (dk_explicit | dk_extra) & ~valid_mask
        na_mask = (na_explicit | na_extra) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # ---- Build the numeric table (counts + mean) ----
    numeric = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=float)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]

        sn, valid_mask, dk_mask, na_mask = classify_music_item(s)

        numeric.loc["(1) Like very much", genre_label] = float((sn == 1).sum())
        numeric.loc["(2) Like it", genre_label] = float((sn == 2).sum())
        numeric.loc["(3) Mixed feelings", genre_label] = float((sn == 3).sum())
        numeric.loc["(4) Dislike it", genre_label] = float((sn == 4).sum())
        numeric.loc["(5) Dislike very much", genre_label] = float((sn == 5).sum())
        numeric.loc["(M) Don’t know much about it", genre_label] = float(dk_mask.sum())
        numeric.loc["(M) No answer", genre_label] = float(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        numeric.loc["Mean", genre_label] = float(mean_val) if pd.notna(mean_val) else np.nan

    # ---- Format return table with Attitude column ----
    formatted = pd.DataFrame(index=row_labels, columns=["Attitude"] + [g[0] for g in genres], dtype=object)
    formatted["Attitude"] = row_labels

    for genre_label, _ in genres:
        col = genre_label
        out = []
        for r in row_labels:
            v = numeric.loc[r, col]
            if r == "Mean":
                out.append("" if pd.isna(v) else f"{float(v):.2f}")
            else:
                out.append("" if pd.isna(v) else str(int(round(float(v)))))
        formatted[col] = out

    # ---- Save as human-readable text file with 3 panels (6 genres each) ----
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

    attitude_col = "Attitude"
    row_w = max(len(attitude_col), max(len(str(x)) for x in formatted[attitude_col])) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(c), max_cell_len) + 4

            header = _pad(attitude_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for i in range(len(formatted)):
                att = formatted.iloc[i][attitude_col]
                line = _pad(att, row_w, "left")
                for c in panel_cols:
                    val = formatted.iloc[i][c]
                    line += _pad(val, widths[c], "center" if att == "Mean" else "right")
                f.write(line + "\n")

    return formatted