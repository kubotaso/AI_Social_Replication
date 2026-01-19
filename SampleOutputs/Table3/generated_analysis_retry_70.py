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

    missing_tokens = {
        "DK": {
            "[NA(D)]", "NA(D)", "NA D", "NAD",
            "DON'T KNOW", "DONT KNOW", "DK",
            "DON'T KNOW MUCH", "DONT KNOW MUCH", "DON'T KNOW MUCH ABOUT IT", "DONT KNOW MUCH ABOUT IT"
        },
        "NA": {
            "[NA(N)]", "NA(N)", "NA N", "NAN",
            "NO ANSWER", "NA", "N/A", "NO ANS"
        },
    }

    def _to_str_series(s: pd.Series) -> pd.Series:
        # robust string conversion (preserve <NA> handling)
        return s.astype("string").str.strip()

    def _token_mask(s_str: pd.Series, tokens) -> pd.Series:
        s_up = s_str.str.upper()
        mask = pd.Series(False, index=s_str.index)
        for t in tokens:
            t_up = str(t).upper()
            # avoid regex pitfalls: literal contains
            mask = mask | s_up.str.contains(t_up, regex=False, na=False)
        return mask

    def _explicit_missing_masks(raw: pd.Series):
        s_str = _to_str_series(raw)
        dk = _token_mask(s_str, missing_tokens["DK"])
        na = _token_mask(s_str, missing_tokens["NA"])
        # if something matches both (rare), treat as NA
        dk = dk & ~na
        return dk, na

    def _tabulate_one(raw: pd.Series, varname: str):
        # parse numeric
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit missing codes if present as strings
        dk_exp, na_exp = _explicit_missing_masks(raw)
        exp_any = int(dk_exp.sum()) + int(na_exp.sum()) > 0

        # "missing pool" are records not in 1..5
        missing_pool = valid.isna()

        if exp_any:
            # classify all missing not explicitly NA(n) as DK (including blank NA)
            na_mask = na_exp
            dk_mask = missing_pool & ~na_mask
            return valid, dk_mask, na_mask

        # No explicit separation possible from this CSV.
        # Many GSS extracts collapse all missings to blank/NA, which makes DK vs NA unidentifiable.
        # To still produce Table 3 (which requires both rows), we attempt a deterministic split
        # that uses *cross-variable information* only when a variable provides explicit labels.
        # If none exist anywhere, we cannot infer the split without external information.
        raise ValueError(
            f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' counts for {varname}: "
            "this CSV export does not preserve distinguishable missing categories (e.g., '[NA(d)]'/'[NA(n)]' "
            "or labeled DK/No answer strings/codes). Re-export data preserving these codes."
        )

    # Validate presence of all genre variables
    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {v}")

    # -----------------------
    # Build table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        raw = df[var]
        valid, dk_mask, na_mask = _tabulate_one(raw, var)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # -----------------------
    # Save human-readable text
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

    # Split into 3 blocks of 6 genres (paper-style layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i: i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table