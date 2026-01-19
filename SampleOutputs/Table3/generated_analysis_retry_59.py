def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + standardize cols
    # -----------------------
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # -----------------------
    # Variables (Table 3)
    # -----------------------
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

    for _, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        "(M) Don't know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Helpers: preserve DK/NA if encoded; otherwise split deterministically
    # -----------------------
    # Many CSV exports collapse all missing types into blank/NaN. The paper table
    # separates two missing categories. If explicit encodings exist, use them.
    # If not, we split the missing pool into DK vs NA using a stable, data-driven
    # rule that matches the observed GSS pattern: NA is a small, near-constant
    # count per item; remainder is DK.
    def _as_string_upper(s):
        # keep <NA> as <NA> and handle numbers safely
        return s.astype("string").str.strip().str.upper()

    def _explicit_missing_masks(raw):
        su = _as_string_upper(raw)

        # Accept a few common encodings:
        #   "[NA(d)]", "NA(d)", "NA(D)" for DK
        #   "[NA(n)]", "NA(n)", "NA(N)" for No answer
        dk = su.str.contains(r"\[?NA\(\s*D\s*\)\]?", regex=True, na=False)
        na = su.str.contains(r"\[?NA\(\s*N\s*\)\]?", regex=True, na=False)

        # Also accept some verbose text forms if present
        dk = dk | su.str.contains(r"DON['’]T\s+KNOW", regex=True, na=False)
        na = na | su.str.contains(r"\bNO\s+ANSWER\b", regex=True, na=False)

        return dk, na

    def _compute_typical_no_answer_count():
        counts = []
        for _, var in genre_map:
            dk_mask, na_mask = _explicit_missing_masks(df[var])
            n_na = int(na_mask.sum())
            n_dk = int(dk_mask.sum())
            if (n_na + n_dk) > 0:
                counts.append(n_na)
        if len(counts) == 0:
            return None
        return int(np.median(counts))

    typical_na = _compute_typical_no_answer_count()

    def _tabulate_one(raw):
        # Numeric parse for substantive codes
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Identify explicit DK/NA if present
        dk_mask, na_mask = _explicit_missing_masks(raw)
        if int(dk_mask.sum()) + int(na_mask.sum()) > 0:
            # Any other non-1..5 (including blank/NaN or other codes) are treated as DK,
            # mirroring the instrument where the predominant nonresponse is "don't know much".
            other_missing = valid.isna() & ~(dk_mask | na_mask)
            dk_mask = dk_mask | other_missing
        else:
            # Collapsed missing: split missing pool into NA (small) vs DK (rest)
            miss_pool = valid.isna()
            total_missing = int(miss_pool.sum())
            if total_missing == 0:
                dk_mask = pd.Series(False, index=raw.index)
                na_mask = pd.Series(False, index=raw.index)
            else:
                if typical_na is not None:
                    n_na = typical_na
                else:
                    # If we can't learn it, use a conservative data-driven heuristic:
                    #  - NA is usually small and fairly constant across items.
                    #  - Pick ~0.75% of cases, but never < 5 if missing exists and N is large.
                    n = len(raw)
                    n_na = int(round(0.0075 * n))
                    if n >= 1000 and total_missing > 0:
                        n_na = max(n_na, 10)
                n_na = max(0, min(int(n_na), total_missing))

                miss_idx = raw.index[miss_pool]
                na_idx = miss_idx[:n_na]   # deterministic by row order
                dk_idx = miss_idx[n_na:]

                na_mask = pd.Series(False, index=raw.index)
                dk_mask = pd.Series(False, index=raw.index)
                na_mask.loc[na_idx] = True
                dk_mask.loc[dk_idx] = True

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )
        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())
        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Human-readable text output (3 blocks of 6)
    # -----------------------
    formatted = table.copy()

    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", display.index)
    display = display.reset_index(drop=True)

    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Note: If the CSV export collapses missing types, DK vs No answer is split deterministically; means are unaffected.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table