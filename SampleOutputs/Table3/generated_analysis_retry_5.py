def run_analysis(data_source):
    import os
    import re
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")
    df = df.loc[df["YEAR"].eq(1993)].copy()

    genre_map = [
        ("Latin/Salsa", "LATIN"),
        ("Jazz", "JAZZ"),
        ("Blues/R&B", "BLUES"),
        ("Show Tunes", "MUSICALS"),
        ("Oldies", "OLDIES"),
        ("Classical/Chamber", "CLASSICL"),
        ("Reggae", "REGGAE"),
        ("Swing/Big Band", "BIGBAND"),
        ("New Age/Space", "NEWAGE"),
        ("Opera", "OPERA"),
        ("Bluegrass", "BLUGRASS"),
        ("Folk", "FOLK"),
        ("Pop/Easy Listening", "MOODEASY"),
        ("Contemporary Rock", "CONROCK"),
        ("Rap", "RAP"),
        ("Heavy Metal", "HVYMETAL"),
        ("Country/Western", "COUNTRY"),
        ("Gospel", "GOSPEL"),
    ]

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

    # ---- helpers: robust parsing of GSS-style missing markers ----
    na_token_re = re.compile(r"NA\(\s*([A-Za-z])\s*\)", re.IGNORECASE)

    def _coerce_series_for_table(raw):
        """
        Returns:
          valid_num: numeric series with only 1..5, else NaN
          dk_mask: True where response is "Don't know much about it" (NA(d))
          na_mask: True where response is "No answer" (NA(n))
        Notes:
          - Works for bracketed strings like "[NA(d)]", plain "NA(d)", etc.
          - If data are numeric (e.g., 8/9 or 98/99), infers DK/NA codes from the two
            most common non-1..5 numeric codes (DK usually more frequent than NA).
          - Any other missing codes are excluded from mean and not counted in the two M rows,
            consistent with the published table showing only DK and No answer.
        """
        s = raw

        # String detection for explicit NA(d)/NA(n)
        s_str = s.astype("string")
        extracted = s_str.str.extract(na_token_re, expand=False)  # yields letter or <NA>
        letter = extracted.str.lower()

        dk_mask = letter.eq("d").fillna(False).to_numpy()
        na_mask = letter.eq("n").fillna(False).to_numpy()

        # Numeric parsing for substantive + possible numeric missing codes
        x = pd.to_numeric(s, errors="coerce")

        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If no explicit DK/NA tokens present, infer numeric codes (if any)
        has_explicit = bool(dk_mask.sum() + na_mask.sum())
        non_1_5 = x.notna() & (~x.isin([1, 2, 3, 4, 5]))

        if (not has_explicit) and bool(non_1_5.any()):
            vc = x.loc[non_1_5].value_counts(dropna=True)
            top = list(vc.index[:2])

            dk_code = None
            na_code = None

            if len(top) == 1:
                dk_code = top[0]
            elif len(top) >= 2:
                # assign the more frequent as DK, less frequent as NA
                c1, c2 = top[0], top[1]
                if vc.loc[c1] >= vc.loc[c2]:
                    dk_code, na_code = c1, c2
                else:
                    dk_code, na_code = c2, c1

            if dk_code is not None:
                dk_mask = dk_mask | (x.eq(dk_code) & non_1_5).to_numpy()
            if na_code is not None:
                na_mask = na_mask | (x.eq(na_code) & non_1_5).to_numpy()

        # prevent double-counting if something somehow matches both
        na_mask = na_mask & (~dk_mask)

        return valid_num, pd.Series(dk_mask, index=raw.index), pd.Series(na_mask, index=raw.index)

    # ---- build table ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        valid_num, dk_mask, na_mask = _coerce_series_for_table(df[var])

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- format for text output ----
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    # ---- save as three 6-column blocks ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = formatted.loc[:, cols].copy()
            block_df.index.name = "Attitude"
            f.write(block_df.to_string())
            f.write("\n\n")

    return table