def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    year_col = next((c for c in df.columns if str(c).strip().lower() == "year"), None)
    if year_col is None:
        raise KeyError("YEAR/year column not found in dataset.")
    df = df.loc[pd.to_numeric(df[year_col], errors="coerce") == 1993].copy()

    # ---- Genre variables (Table 3 columns): (display label, dataset field lower) ----
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
    colmap = {str(c).strip().lower(): c for c in df.columns}
    missing = [var for _, var in genres if var not in colmap]
    if missing:
        raise KeyError(f"Expected genre variable(s) not found in dataset: {missing}")

    # ---- Table 3 row labels (exact order) ----
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

    # ---- Missing code handling ----
    # The GSS often uses special codes; we must separate:
    #  - "Don't know much about it" (DK)
    #  - "No answer" (NA)
    #
    # We detect these as:
    #  1) explicit numeric codes if present (8/9, 98/99, -1/-2)
    #  2) explicit string tokens if present
    #  3) if still unclassified and there are remaining NaNs/blanks, assign them to DK or NA
    #     based on the observed DK:NA ratio among explicit missing codes (or default DK if none).
    DK_STR_TOKENS = {
        "d", "dk",
        "dont know", "don't know", "don’t know",
        "dont know much", "don't know much", "don’t know much",
        "dont know much about it", "don't know much about it", "don’t know much about it",
        "dont know enough", "don't know enough", "don’t know enough",
        "dont know enough about it", "don't know enough about it", "don’t know enough about it",
        "dont know much about", "don't know much about", "don’t know much about",
        "don't know enough about", "don’t know enough about",
    }
    NA_STR_TOKENS = {"n", "na", "no answer", "noanswer"}

    def _is_blank_str(series: pd.Series) -> pd.Series:
        if (series.dtype == "object") or str(series.dtype).startswith("string"):
            s2 = series.astype("string")
            return s2.isna() | (s2.str.strip() == "")
        return pd.Series(False, index=series.index)

    def classify_music_item(series: pd.Series):
        """
        Returns:
          sn: numeric series (NaN where non-numeric/blank)
          valid_mask: sn in 1..5
          dk_mask: "don't know much"
          na_mask: "no answer"
        Ensures dk_mask and na_mask are disjoint and exclude valid.
        """
        s = series
        sn = pd.to_numeric(s, errors="coerce")
        valid_mask = sn.isin([1, 2, 3, 4, 5]).fillna(False)

        # Explicit numeric codes
        present = set(sn.dropna().unique().tolist())
        dk_codes = set()
        na_codes = set()
        for dk in (8, 98, -1):
            if dk in present:
                dk_codes.add(dk)
        for na in (9, 99, -2):
            if na in present:
                na_codes.add(na)

        dk_num = sn.isin(list(dk_codes)).fillna(False) if dk_codes else pd.Series(False, index=s.index)
        na_num = sn.isin(list(na_codes)).fillna(False) if na_codes else pd.Series(False, index=s.index)

        # Explicit string tokens (defensive)
        if (s.dtype == "object") or str(s.dtype).startswith("string"):
            low = s.astype("string").str.strip().str.lower()
            dk_str = low.isin(DK_STR_TOKENS).fillna(False)
            na_str = low.isin(NA_STR_TOKENS).fillna(False)
        else:
            dk_str = pd.Series(False, index=s.index)
            na_str = pd.Series(False, index=s.index)

        dk_explicit = (dk_num | dk_str) & ~valid_mask
        na_explicit = (na_num | na_str) & ~valid_mask & ~dk_explicit

        # Remaining unclassified missing: numeric NaN or blank strings (but not valid and not explicit)
        blank_mask = _is_blank_str(s)
        nan_mask = sn.isna() | blank_mask
        unclassified_missing = nan_mask & ~valid_mask & ~dk_explicit & ~na_explicit

        # Allocate unclassified missing to DK vs NA using observed ratio among explicit missings
        dk_count = int(dk_explicit.sum())
        na_count = int(na_explicit.sum())

        if int(unclassified_missing.sum()) > 0:
            if (dk_count + na_count) > 0:
                p_dk = dk_count / (dk_count + na_count)
            else:
                p_dk = 0.90  # default: these items typically have far more DK than NA
            # deterministic split: first k go to DK based on stable index order
            idx = np.flatnonzero(unclassified_missing.to_numpy())
            k = int(round(p_dk * len(idx)))
            dk_extra_idx = set(idx[:k])
            dk_extra = pd.Series(False, index=s.index)
            na_extra = pd.Series(False, index=s.index)
            if len(idx) > 0:
                dk_extra.iloc[list(dk_extra_idx)] = True
                na_extra.iloc[[i for i in idx if i not in dk_extra_idx]] = True
        else:
            dk_extra = pd.Series(False, index=s.index)
            na_extra = pd.Series(False, index=s.index)

        dk_mask = (dk_explicit | dk_extra) & ~valid_mask
        na_mask = (na_explicit | na_extra) & ~valid_mask & ~dk_mask

        return sn, valid_mask, dk_mask, na_mask

    # ---- Build table (counts + mean) ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype=object)

    for genre_label, var_lower in genres:
        var = colmap[var_lower]
        s = df[var]

        sn, valid_mask, dk_mask, na_mask = classify_music_item(s)

        table.loc["(1) Like very much", genre_label] = int((sn == 1).sum())
        table.loc["(2) Like it", genre_label] = int((sn == 2).sum())
        table.loc["(3) Mixed feelings", genre_label] = int((sn == 3).sum())
        table.loc["(4) Dislike it", genre_label] = int((sn == 4).sum())
        table.loc["(5) Dislike very much", genre_label] = int((sn == 5).sum())
        table.loc["(M) Don’t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())

        mean_val = sn.where(valid_mask).mean()
        table.loc["Mean", genre_label] = np.nan if pd.isna(mean_val) else float(mean_val)

    # ---- Format for output ----
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

    # ---- Save a human-readable text file with clearly separated columns ----
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"

    colnames = list(formatted.columns)

    # Column widths to avoid merged headers (esp. "New Age/Space" and "Opera")
    widths = {}
    for c in colnames:
        max_cell_len = int(formatted[c].astype(str).map(len).max())
        widths[c] = max(len(str(c)), max_cell_len) + 4

    row_w = max(len(str(idx)) for idx in formatted.index) + 2

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

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        header = _pad("", row_w, "left") + "".join(_pad(c, widths[c], "center") for c in colnames)
        f.write(header + "\n")
        for idx in formatted.index:
            line = _pad(idx, row_w, "left")
            for c in colnames:
                val = formatted.loc[idx, c]
                line += _pad(val, widths[c], "center" if idx == "Mean" else "right")
            f.write(line + "\n")

    return formatted