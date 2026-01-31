def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # --- Restrict to GSS 1993 (robust to case) ---
    year_col = next((c for c in df.columns if c.strip().lower() == "year"), None)
    if year_col is None:
        raise KeyError("YEAR/year column not found in dataset.")
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # --- Genre variables (Table 3 columns): (display label, dataset field) ---
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

    # Resolve actual column names case-insensitively
    colmap = {c.lower(): c for c in df.columns}
    missing = [var for _, var in genres if var not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # --- Table 3 rows ---
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

    # --- Missing code handling: detect DK/NA reliably for this dataset ---
    # Common GSS-style numeric schemes in various exports:
    #   DK: 8, 98, -1 ; NA: 9, 99, -2
    # Some CSV extracts also store DK/NA as text.
    DK_STR_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
        "don't know much about", "don’t know much about",
    }
    NA_STR_TOKENS = {"n", "na", "no answer", "noanswer"}

    def classify_missing(series: pd.Series):
        """
        Return (dk_mask, na_mask, valid_mask) where valid_mask is substantive 1..5.
        DK/NA are detected using:
          - numeric: 8/9, 98/99, -1/-2 (and also treat pandas NaN as 'no answer' only if no explicit NA code exists? NO: keep NaN separate)
          - string tokens for DK/NA.
        Note: We do NOT count plain NaN as either DK or NA (paper distinguishes DK vs NA codes).
        """
        sn = pd.to_numeric(series, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        present = set(sn.dropna().unique().tolist())

        dk_codes = set()
        na_codes = set()

        # Prefer explicit missing pairs if present
        if 8 in present:
            dk_codes.add(8)
        if 9 in present:
            na_codes.add(9)

        if 98 in present:
            dk_codes.add(98)
        if 99 in present:
            na_codes.add(99)

        if -1 in present:
            dk_codes.add(-1)
        if -2 in present:
            na_codes.add(-2)

        dk_num = sn.isin(list(dk_codes)).fillna(False) if dk_codes else pd.Series(False, index=sn.index)
        na_num = sn.isin(list(na_codes)).fillna(False) if na_codes else pd.Series(False, index=sn.index)

        # Add string token detection (defensive)
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            low = series.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR_TOKENS).fillna(False)
            na_str = low.isin(NA_STR_TOKENS).fillna(False)
            dk_mask = dk_num | dk_str
            na_mask = na_num | na_str
        else:
            dk_mask = dk_num
            na_mask = na_num

        return dk_mask, na_mask, valid_mask

    # --- Build table (counts + mean) ---
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=object)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]
        sn = pd.to_numeric(s, errors="coerce")

        # Substantive counts 1..5
        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())

        # Missing categories shown in Table 3
        dk_mask, na_mask, valid_mask = classify_missing(s)
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        # Mean on valid 1..5 only
        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # --- Format for clean text export (fixed widths so headers don't run together) ---
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )
        else:
            formatted.loc[r, :] = formatted.loc[r, :].apply(
                lambda x: "" if pd.isna(x) else str(int(x))
            )

    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    # Compute column widths to prevent merged headers (e.g., "New Age/Space" and "Opera")
    colnames = list(formatted.columns)
    # width per column: max(header length, max cell length) + padding
    widths = {}
    for c in colnames:
        max_cell = int(formatted[c].astype(str).map(len).max())
        widths[c] = max(len(str(c)), max_cell) + 4  # padding

    def _pad(s, w, align="center"):
        s = str(s)
        if len(s) >= w:
            return s
        if align == "left":
            return s + " " * (w - len(s))
        if align == "right":
            return " " * (w - len(s)) + s
        # center
        left = (w - len(s)) // 2
        right = w - len(s) - left
        return " " * left + s + " " * right

    # Row label width
    row_w = max(len(idx) for idx in formatted.index) + 2

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")

        # Header row
        header = _pad("", row_w, "left") + "".join(_pad(c, widths[c], "center") for c in colnames)
        f.write(header + "\n")

        # Body rows
        for idx in formatted.index:
            line = _pad(idx, row_w, "left")
            for c in colnames:
                # right-align numbers, center means (formatted already)
                val = formatted.loc[idx, c]
                align = "right" if idx != "Mean" else "center"
                line += _pad(val, widths[c], align)
            f.write(line + "\n")

    return formatted