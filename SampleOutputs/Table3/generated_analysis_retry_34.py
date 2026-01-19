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

    # Keep 1993 only; drop missing YEAR if any
    df = df.loc[df["YEAR"].notna() & (df["YEAR"].astype(int) == 1993)].copy()

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
    # Helper: robustly detect DK vs NA in common GSS encodings
    # -----------------------
    # We do NOT guess/split missing. If the extract doesn't preserve DK vs NA distinctly,
    # we error with a clear message (rather than outputting wrong counts).
    def _as_str_upper(s: pd.Series) -> pd.Series:
        # Keep NA as <NA> in pandas string dtype, not the literal "NAN"
        return s.astype("string").str.strip().str.upper()

    def _detect_dk_na_masks(raw: pd.Series):
        s = _as_str_upper(raw)

        # Common encodings seen across extracts/codebooks:
        #   "[NA(D)]", "NA(D)", "NA(D) ..."
        #   "[NA(N)]", "NA(N)"
        #   also sometimes "DK", "DON'T KNOW", "DONT KNOW", "NO ANSWER", "NA"
        # We keep patterns conservative to avoid false positives.
        dk_pat = r"(\[NA\(D\)\]|(^|\b)NA\(D\)(\b|$)|(^|\b)DK(\b|$)|DON'?T\s+KNOW)"
        na_pat = r"(\[NA\(N\)\]|(^|\b)NA\(N\)(\b|$)|NO\s+ANSWER)"

        dk_mask = s.str.contains(dk_pat, regex=True, na=False)
        na_mask = s.str.contains(na_pat, regex=True, na=False)
        return dk_mask, na_mask

    # -----------------------
    # Build numeric representation + validate split feasibility
    # -----------------------
    # Strategy:
    # - Parse numeric values; valid attitudes are 1..5.
    # - Missing pool = values not in 1..5 (including NaN).
    # - DK/NA must be separable using explicit encodings in the raw column.
    #   If not, refuse to proceed (prevents the prior wrong outputs / runtime guessing).
    def _compute_counts_and_mean(raw: pd.Series, varname: str):
        # Numeric parse for 1..5 and mean
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Missing pool (anything not valid 1..5, including NaN)
        miss_pool = valid.isna()

        # Detect explicit DK/NA
        dk_mask, na_mask = _detect_dk_na_masks(raw)

        # If there is any missing, we must be able to attribute it to DK/NA
        # (Table 3 requires both). If the extract only has blank NaN, we cannot.
        if int(miss_pool.sum()) > 0:
            classified = (dk_mask | na_mask)
            unclassified_missing = miss_pool & ~classified

            # Some extracts may encode DK/NA as numeric values outside 1..5.
            # Try to detect a stable two-code split among non-1..5 numerics.
            # If exactly two distinct non-1..5 numeric codes exist, map smaller->DK larger->NA
            # only if there is still no string evidence.
            if int(classified.sum()) == 0:
                nonvalid_numeric = x[miss_pool & x.notna()]
                uniq = sorted(pd.unique(nonvalid_numeric))
                if len(uniq) == 2:
                    dk_code, na_code = uniq[0], uniq[1]
                    dk_mask = (x == dk_code)
                    na_mask = (x == na_code)
                    classified = dk_mask | na_mask
                    unclassified_missing = miss_pool & ~classified

            if int(unclassified_missing.sum()) > 0:
                raise ValueError(
                    f"Cannot compute separate '(M) Don\\'t know much about it' vs '(M) No answer' "
                    f"counts for {varname}: dataset extract does not preserve distinct DK/NA codes "
                    f"for all missing/non-1..5 values."
                )

        # Counts
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
    # Construct table (counts + mean)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _compute_counts_and_mean(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # Optional: total row could be useful, but Table 3 doesn't show it; keep out of returned structure.
    # Users can compute totals by summing the 7 count rows.

    # -----------------------
    # Save human-readable text file in 3 blocks of 6
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
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: "
            "General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table