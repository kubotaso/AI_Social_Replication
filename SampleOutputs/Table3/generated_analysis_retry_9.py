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

    def parse_na_tag(val):
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        s = str(val).strip()
        if not s:
            return None
        su = s.upper()
        # Match "[NA(d)]", "NA(d)", "[NA(D)]", etc.
        m = re.search(r"NA\(\s*([A-Z])\s*\)", su)
        return m.group(1) if m else None

    def is_numeric_like(x):
        try:
            float(str(x).strip())
            return True
        except Exception:
            return False

    def extract_valid_dk_na(series):
        """
        Returns:
          valid_num: numeric Series with values 1..5 else NaN
          dk_mask: boolean Series for NA(d) / dk category
          na_mask: boolean Series for NA(n) / no answer category
        """
        s = series.copy()

        # Detect tagged missing codes like "[NA(d)]"
        tags = s.map(parse_na_tag)

        dk_mask = tags.eq("D")
        na_mask = tags.eq("N")

        # Parse numeric values (works for floats/ints/strings like "3.0")
        x = pd.to_numeric(s, errors="coerce")

        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If no tagged DK/NA are present, but there are other non-1..5 numeric codes,
        # infer DK and NA from the two most common non-1..5 numeric codes.
        has_tagged = bool(dk_mask.fillna(False).any() or na_mask.fillna(False).any())
        non_valid_numeric = x.notna() & (~x.isin([1, 2, 3, 4, 5]))

        if (not has_tagged) and bool(non_valid_numeric.any()):
            vc = x.loc[non_valid_numeric].value_counts()
            top = list(vc.index[:2])

            dk_code = None
            na_code = None
            if len(top) >= 1:
                dk_code = top[0]
            if len(top) >= 2:
                na_code = top[1]

            # Prefer conventional GSS: 8=DK, 9=NA if present among candidates
            candidates = set(vc.index.tolist())
            if 8 in candidates:
                dk_code = 8
                if 9 in candidates:
                    na_code = 9
                else:
                    # keep inferred na_code if it exists and isn't dk_code
                    na_code = na_code if (na_code is not None and na_code != dk_code) else None
            elif 9 in candidates:
                # If only 9 present, treat as NA
                na_code = 9
                dk_code = dk_code if dk_code != na_code else None

            if dk_code is not None:
                dk_mask = dk_mask.fillna(False) | x.eq(dk_code)
            else:
                dk_mask = dk_mask.fillna(False)

            if na_code is not None:
                na_mask = na_mask.fillna(False) | x.eq(na_code)
            else:
                na_mask = na_mask.fillna(False)
        else:
            dk_mask = dk_mask.fillna(False)
            na_mask = na_mask.fillna(False)

        # Ensure no overlap
        na_mask = na_mask & (~dk_mask)

        return valid_num, dk_mask, na_mask

    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]
        valid_num, dk_mask, na_mask = extract_valid_dk_na(raw)

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

    # Format for human-readable output: counts as ints, mean to 2 decimals
    formatted = table.copy()
    for idx in formatted.index:
        if idx == "Mean":
            formatted.loc[idx] = formatted.loc[idx].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[idx] = formatted.loc[idx].map(lambda v: "" if pd.isna(v) else f"{int(round(float(v)))}")

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing (DK/NA).\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = formatted.loc[:, cols].copy()
            block_df.index.name = "Attitude"
            f.write(block_df.to_string())
            f.write("\n\n")

    return table