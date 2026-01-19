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

    # 1993 only
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
    # Missing-category parsing
    # -----------------------
    # Goal:
    #   - Always compute 1..5 counts and Mean from actual numeric values 1..5.
    #   - Compute DK vs No answer ONLY if the dataset preserves separable missing codes.
    #   - If the CSV collapses all missings into blank/NA (common), we cannot infer DK vs NA.
    #     In that case we fall back to: DK=total_missing, NoAnswer=0, and we clearly note this.
    #
    # This avoids runtime errors and ensures the table is always produced from raw data.

    dk_tokens = {
        "[NA(D)]", "NA(D)", "NA(DON'T KNOW)", "NA(DONT KNOW)", "DON'T KNOW", "DONT KNOW",
        "DK", "D/K", "DONT KNOW MUCH", "DON'T KNOW MUCH"
    }
    na_tokens = {
        "[NA(N)]", "NA(N)", "NO ANSWER", "N/A", "NA", "NOT ASCERTAINED"
    }

    def _to_str_series(s):
        # keep NA as <NA> so str ops are safe
        return s.astype("string").str.strip()

    def _explicit_missing_masks(raw):
        s = _to_str_series(raw).str.upper()

        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)

        # direct token matches / containment
        for t in dk_tokens:
            dk_mask = dk_mask | s.eq(t) | s.str.contains(t.replace("[", r"\[").replace("]", r"\]"), regex=True, na=False)
        for t in na_tokens:
            na_mask = na_mask | s.eq(t) | s.str.contains(t.replace("[", r"\[").replace("]", r"\]"), regex=True, na=False)

        return dk_mask, na_mask

    def _tabulate_one(raw):
        # numeric valid 1..5
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # explicit coded missing (if preserved)
        dk_mask_exp, na_mask_exp = _explicit_missing_masks(raw)

        # total missing (anything not in 1..5)
        total_missing = int(valid.isna().sum())

        explicit_any = int(dk_mask_exp.sum()) + int(na_mask_exp.sum())
        explicit_cover = int((dk_mask_exp | na_mask_exp).sum())

        # If the file has explicit separable codes, use them; any remaining non-1..5 are folded into DK
        # (conservative: treat unclassified missing as DK rather than NA).
        if explicit_any > 0:
            other_missing = valid.isna() & ~(dk_mask_exp | na_mask_exp)
            dk_n = int((dk_mask_exp | other_missing).sum())
            na_n = int(na_mask_exp.sum())
            note = ""
            # sanity: should not exceed total missing
            if dk_n + na_n != total_missing:
                # enforce accounting identity by pushing any difference into DK
                dk_n = total_missing - na_n
            return counts_1_5, dk_n, na_n, mean_val, note

        # If there are no explicit codes, we cannot split DK vs NA from this CSV.
        # Return DK as total missing, NA as 0, and attach a note.
        dk_n = total_missing
        na_n = 0
        note = "NOTE: DK vs No answer not distinguishable in this CSV export; all missing/non-1..5 counted as DK and No answer set to 0."
        return counts_1_5, dk_n, na_n, mean_val, note

    # -----------------------
    # Build numeric table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")
    notes = []

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val, note = _tabulate_one(df[var])
        if note:
            notes.append(f"{var}: {note}")

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Save human-readable text (3 blocks of 6)
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(v)))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        if notes:
            f.write("\nDATA NOTE(S):\n")
            # de-duplicate identical notes
            for line in sorted(set(notes)):
                f.write(f"- {line}\n")
        f.write("\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table