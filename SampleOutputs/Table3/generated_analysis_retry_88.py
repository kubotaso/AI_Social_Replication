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
    df = df.loc[df["YEAR"] == 1993].copy()

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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required column not found: {v}")

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

    # -----------------------
    # Missing code handling
    # -----------------------
    # We must compute DK vs NA from raw data, but many CSV exports collapse both into NaN.
    # Strategy:
    # 1) If explicit markers exist (strings like "[NA(d)]", "[NA(n)]", "DON'T KNOW", "NO ANSWER", etc.), use them.
    # 2) If not preserved (all missing are NaN), we cannot distinguish DK vs NA from raw data alone.
    #    In that case, we return the table with correct 1–5 counts and mean, and place all missing into DK,
    #    leaving NA as 0, while writing a note in the output file. This is still computed from raw data.
    #
    # This avoids runtime errors and respects "never hardcode paper numbers".

    def _to_str_series(s):
        # use pandas' nullable string dtype for safe .str ops
        return s.astype("string")

    def _explicit_missing_masks(raw):
        s = _to_str_series(raw).str.strip().str.upper()

        # DK tokens commonly seen in GSS extracts / labels
        dk_tokens = [
            "[NA(D)]",
            "NA(D)",
            "DONT KNOW",
            "DON'T KNOW",
            "DK",
            "DON’T KNOW",
            "DON'T KNOW MUCH ABOUT IT",
            "DONT KNOW MUCH ABOUT IT",
        ]
        # No-answer tokens
        na_tokens = [
            "[NA(N)]",
            "NA(N)",
            "NO ANSWER",
            "NA",
            "N/A",
            "NOT ASCERTAINED",
        ]

        def contains_any(tokens):
            mask = pd.Series(False, index=s.index)
            for t in tokens:
                mask = mask | s.str.contains(repr(t)[1:-1], regex=False, na=False)
            return mask

        dk = contains_any(dk_tokens)
        na = contains_any(na_tokens)
        return dk, na

    def _tabulate_one(raw):
        # numeric values for 1..5
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit DK/NA markers (if any)
        dk_exp, na_exp = _explicit_missing_masks(raw)

        # non-1..5 entries (includes NaN and other codes)
        nonvalid = ~x.isin([1, 2, 3, 4, 5])

        # If we have explicit classification, use it; otherwise fall back.
        if int(dk_exp.sum() + na_exp.sum()) > 0:
            # Anything nonvalid not explicitly NA is treated as DK (conservative)
            other_missing = nonvalid & ~(dk_exp | na_exp)
            dk = dk_exp | other_missing
            na = na_exp
            note = ""
        else:
            # No explicit DK/NA preservation; cannot separate from raw export.
            # Put all missing/nonvalid into DK and NA=0 (both computed from raw data).
            dk = nonvalid.copy()
            na = pd.Series(False, index=raw.index)
            note = "NOTE: DK vs No answer not distinguishable in this CSV export (missing collapsed). All non-1..5 counted as DK; NA set to 0."

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        dk_n = int(dk.sum())
        na_n = int(na.sum())
        return counts_1_5, dk_n, na_n, mean_val, note

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")
    notes = []

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val, note = _tabulate_one(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

        if note:
            notes.append(f"{var}: {note}")

    # -----------------------
    # Save human-readable output in 3 panels of 6 genres
    # -----------------------
    def _format_block(block_df):
        fmt = block_df.copy()
        for idx in fmt.index:
            if idx == "Mean":
                fmt.loc[idx] = fmt.loc[idx].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
            else:
                fmt.loc[idx] = fmt.loc[idx].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))
        fmt.insert(0, "Attitude", fmt.index)
        fmt = fmt.reset_index(drop=True)
        return fmt

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing/non-1..5.\n\n")

        if notes:
            f.write("Data note(s):\n")
            # write unique notes
            for n in sorted(set(notes)):
                f.write(f"- {n}\n")
            f.write("\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Panel {bi}:\n")
            block = table.loc[:, cols]
            f.write(_format_block(block).to_string(index=False))
            f.write("\n\n")

    return table