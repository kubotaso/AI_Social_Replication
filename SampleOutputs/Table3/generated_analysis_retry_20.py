def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Required column 'year' not found.")

    # Filter to GSS 1993
    df = df.loc[df["year"].eq(1993)].copy()

    # ---- Variables (Table 3) ----
    genre_map = [
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

    # ---- Helpers ----
    # GSS missing codes may appear as:
    #   - real NaN (blank)
    #   - strings like "[NA(d)]" or "NA(d)"
    # We must count DK vs NA separately when identifiable; otherwise, report all missing as "No answer"
    # (rather than inventing a split).
    def split_missing(series):
        s = series

        # Identify explicit string-coded missingness
        s_str = s.astype("string")
        s_up = s_str.str.strip().str.upper()

        dk_mask = s_up.str.contains(r"\[?NA\(D\)\]?", regex=True, na=False)
        na_mask = s_up.str.contains(r"\[?NA\(N\)\]?", regex=True, na=False)

        # Numeric parsing
        x = pd.to_numeric(s_str, errors="coerce")

        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Missing pool includes:
        # - NaN after numeric parse
        # - numeric values not in 1..5 (e.g., 0, 8, 9, 98, 99 etc.)
        invalid_numeric = x.notna() & (~x.isin([1, 2, 3, 4, 5]))
        missing_pool = x.isna() | invalid_numeric

        # If DK/NA were explicitly coded, use them (and ensure they are within missing_pool)
        if dk_mask.any() or na_mask.any():
            dk_mask = dk_mask & missing_pool
            na_mask = na_mask & missing_pool & (~dk_mask)
            # Any remaining missing that is not DK/NA is treated as "No answer" to match table's two M-rows
            other_missing = missing_pool & (~dk_mask) & (~na_mask)
            na_mask = na_mask | other_missing
            return valid, dk_mask, na_mask

        # If not explicitly coded, we cannot distinguish DK vs NA from raw file reliably.
        # To avoid fabricating counts, put all missing into "No answer" and DK=0.
        dk_mask = pd.Series(False, index=s.index)
        na_mask = missing_pool
        return valid, dk_mask, na_mask

    # ---- Build table (numeric) ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        valid, dk_mask, na_mask = split_missing(df[var])

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_count = int(dk_mask.sum())
        na_count = int(na_mask.sum())

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don't know much about it", genre_label] = dk_count
        table.loc["(M) No answer", genre_label] = na_count
        table.loc["Mean", genre_label] = mean_val

    # ---- Format for display (counts as int; mean to 2 decimals with trailing zeros) ----
    formatted = table.copy()

    for idx in formatted.index:
        if idx == "Mean":
            formatted.loc[idx] = formatted.loc[idx].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        else:
            formatted.loc[idx] = formatted.loc[idx].map(
                lambda v: "" if pd.isna(v) else str(int(v))
            )

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # ---- Save as three 6-column blocks (layout like the printed table) ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g[0] for g in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n"
        )
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            block_df = display.loc[:, cols_with_stub]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table