def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---------- Resolve columns case-insensitively ----------
    colmap = {str(c).strip().lower(): c for c in df.columns}

    # ---------- Restrict to GSS 1993 ----------
    if "year" not in colmap:
        raise KeyError("Expected column 'year' not found in dataset.")
    year_col = colmap["year"]
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---------- Table 3 genre variables (exact set/order) ----------
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

    # ---------- Row labels (exact order) ----------
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

    # ---------- Robust missing-code detection (computed from data; no hard-coded paper counts) ----------
    # We must count two specific missing categories:
    #   DK = "Don't know much about it"  (documentation: NA(d))
    #   NA = "No answer"                (documentation: NA(n))
    #
    # In many GSS extracts these are encoded as 8/9 or 98/99 (sometimes negative).
    # HOWEVER: in this provided CSV, the "M" categories may already be imported as actual NaN,
    # which would otherwise make both DK and NA appear as 0. To avoid that, we infer DK/NA
    # codes per item from observed non-1..5 codes when available, otherwise fall back to
    # standard conventions. We never "redistribute" NaNs between DK and NA.

    VALID = {1, 2, 3, 4, 5}

    # Candidates to consider as DK/NA codes (kept narrow to avoid misclassifying substantive codes)
    DK_CANDIDATES = [8, 98, 998, -1, -8, -98]
    NA_CANDIDATES = [9, 99, 999, -2, -9, -99]

    def _infer_codes(series_num: pd.Series):
        """
        Infer DK/NA numeric codes for a given item based on which candidate codes actually appear.
        Falls back to {8,98,-1} and {9,99,-2} when nothing appears.
        """
        present = set(pd.unique(series_num.dropna()))
        dk = {c for c in DK_CANDIDATES if c in present}
        na = {c for c in NA_CANDIDATES if c in present}

        # fallbacks if none detected (common GSS patterns)
        if not dk:
            dk = {8, 98, -1}
        if not na:
            na = {9, 99, -2}

        # ensure disjoint
        dk = set(dk) - set(na)
        na = set(na) - set(dk)
        return dk, na

    def _counts_for_item(series: pd.Series):
        # Keep original for possible string tokens, but primarily numeric in this dataset
        s_num = pd.to_numeric(series, errors="coerce")

        dk_codes, na_codes = _infer_codes(s_num)

        # Count 1..5 exactly
        c1 = int((s_num == 1).sum())
        c2 = int((s_num == 2).sum())
        c3 = int((s_num == 3).sum())
        c4 = int((s_num == 4).sum())
        c5 = int((s_num == 5).sum())

        # Count DK/NA explicitly by inferred numeric codes
        dk = int(s_num.isin(list(dk_codes)).sum())
        na = int(s_num.isin(list(na_codes)).sum())

        # If the item contains string tokens (rare here), count them too (without reallocating NaNs)
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            low = series.astype("string").str.strip().str.lower()
            dk_tokens = {
                "d", "dk",
                "dont know", "don't know", "don’t know",
                "dont know much", "don't know much", "don’t know much",
                "dont know much about it", "don't know much about it", "don’t know much about it",
                "dont know enough", "don't know enough", "don’t know enough",
                "dont know enough about it", "don't know enough about it", "don’t know enough about it",
            }
            na_tokens = {"n", "na", "no answer", "noanswer"}
            # Only count tokens where numeric is NaN (avoid double-counting if numeric code parsed)
            numeric_nan = s_num.isna()
            dk += int((numeric_nan & low.isin(dk_tokens)).sum())
            na += int((numeric_nan & low.isin(na_tokens)).sum())

        # Mean over valid 1..5 only
        mean_val = s_num.where(s_num.isin(list(VALID))).mean()
        return c1, c2, c3, c4, c5, dk, na, mean_val

    # ---------- Build table with Attitude as first column ----------
    out_cols = ["Attitude"] + [g[0] for g in genres]
    table = pd.DataFrame(index=row_labels, columns=out_cols, dtype=object)
    table["Attitude"] = row_labels

    for genre_label, var_lower in genres:
        s = df[colmap[var_lower]]
        c1, c2, c3, c4, c5, dk, na, mean_val = _counts_for_item(s)

        table.loc["(1) Like very much", genre_label] = c1
        table.loc["(2) Like it", genre_label] = c2
        table.loc["(3) Mixed feelings", genre_label] = c3
        table.loc["(4) Dislike it", genre_label] = c4
        table.loc["(5) Dislike very much", genre_label] = c5
        table.loc["(M) Don’t know much about it", genre_label] = dk
        table.loc["(M) No answer", genre_label] = na
        table.loc["Mean", genre_label] = mean_val

    # ---------- Format (counts as ints; mean as 2 decimals) ----------
    formatted = table.copy()
    for r in row_labels:
        if r == "Mean":
            for c in formatted.columns:
                if c == "Attitude":
                    continue
                v = formatted.loc[r, c]
                formatted.loc[r, c] = "" if pd.isna(v) else f"{float(v):.2f}"
        else:
            for c in formatted.columns:
                if c == "Attitude":
                    continue
                v = formatted.loc[r, c]
                formatted.loc[r, c] = "" if pd.isna(v) else str(int(v))

    # ---------- Save as human-readable text file with 3 panels (6 genres each) ----------
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
    row_w = max(len(att_col), max(len(str(x)) for x in formatted[att_col].tolist())) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("Counts shown for 1–5 plus (M) Don’t know much about it and (M) No answer. Mean computed over 1–5 only.\n")
        f.write("Note: DK/No-answer codes are inferred per item from observed non-1..5 codes among common GSS conventions.\n")

        for p_idx, panel_cols in enumerate(panels, start=1):
            f.write("\n")
            f.write(f"Panel {p_idx}\n")

            widths = {}
            for c in panel_cols:
                max_cell_len = int(formatted[c].astype(str).map(len).max())
                widths[c] = max(len(str(c)), max_cell_len) + 4

            header = _pad(att_col, row_w, "left") + "".join(_pad(c, widths[c], "center") for c in panel_cols)
            f.write(header + "\n")

            for r in row_labels:
                line = _pad(formatted.loc[r, att_col], row_w, "left")
                for c in panel_cols:
                    val = formatted.loc[r, c]
                    line += _pad(val, widths[c], "center" if r == "Mean" else "right")
                f.write(line + "\n")

    return formatted