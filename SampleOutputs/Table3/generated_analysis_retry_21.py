def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(data_source, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found in data.")

    # Filter to 1993
    df = df.loc[df["YEAR"].eq(1993)].copy()

    # ---- Variables (Table 3) ----
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

    # ---- Helpers ----
    def _norm_missing_token(x):
        """Map various missing encodings to canonical tokens: 'D' (dk), 'N' (no answer), or None."""
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        s = str(x).strip().upper()
        if s == "":
            return None

        # Common GSS encodings
        # Bracketed: [NA(d)], [NA(n)] and variants
        if "[NA(D)]" in s or "NA(D)" in s:
            return "D"
        if "[NA(N)]" in s or "NA(N)" in s:
            return "N"

        # Textual fallbacks sometimes present in extracts
        # (Do not overreach: keep conservative)
        if "DON'T KNOW" in s or "DONT KNOW" in s:
            return "D"
        if "NO ANSWER" in s:
            return "N"

        return None

    def _series_to_tokens_and_numeric(raw):
        """
        Returns:
          valid_num: numeric series with values 1..5 else NaN
          miss_token: series with 'D' for dk, 'N' for no answer, or None
        """
        miss_token = raw.map(_norm_missing_token)

        # Parse numeric; keep only 1..5 as valid for attitude distribution and mean
        x = pd.to_numeric(raw, errors="coerce")
        valid_num = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # If value is non-1..5 numeric, treat as missing (but not DK/NA unless explicitly labeled)
        # miss_token remains None for those cases.
        return valid_num, miss_token

    def _allocate_missing_when_unspecified(raw, valid_num, miss_token):
        """
        If dataset doesn't explicitly distinguish DK vs NA, we cannot infer reliably from microdata.
        In that case, we return DK=0 and NA=total_missing (including blanks and non-1..5 numerics).
        This keeps the table internally consistent without fabricating a split.
        """
        # Identify missing pool not already labeled as D/N
        raw_numeric = pd.to_numeric(raw, errors="coerce")
        pool = valid_num.isna()  # includes NaN and non-1..5 codes coerced to NaN in valid_num
        # But exclude explicitly-labeled DK/NA from pool counting
        labeled = miss_token.isin(["D", "N"])
        pool = pool & (~labeled)

        # If we have any explicit D/N labels, no need for allocation fallback.
        if (miss_token.isin(["D", "N"]).sum()) > 0:
            return None

        # No explicit labels: we cannot split DK vs NA from raw. Allocate all to "No answer".
        # (Better than inventing counts; still computed from raw missingness.)
        na_count = int(pool.sum())
        dk_count = 0
        return dk_count, na_count

    # ---- Build numeric table ----
    genre_labels = [g[0] for g in genre_map]
    table_num = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")

        raw = df[var]

        valid_num, miss_token = _series_to_tokens_and_numeric(raw)

        counts_1_5 = (
            valid_num.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        # Missing counts (preferred: explicit NA(d) and NA(n))
        dk_count = int((miss_token == "D").sum())
        na_count = int((miss_token == "N").sum())

        # If no explicit split exists, compute missingness from raw and allocate conservatively
        fallback = _allocate_missing_when_unspecified(raw, valid_num, miss_token)
        if fallback is not None:
            dk_count, na_count = fallback

        mean_val = float(valid_num.mean(skipna=True)) if valid_num.notna().any() else np.nan

        table_num.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table_num.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table_num.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table_num.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table_num.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table_num.loc["(M) Don't know much about it", genre_label] = dk_count
        table_num.loc["(M) No answer", genre_label] = na_count
        table_num.loc["Mean", genre_label] = mean_val

    # ---- Format for output (counts as int, mean to 2 decimals) ----
    formatted = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="object")
    for r in row_labels:
        if r == "Mean":
            formatted.loc[r] = table_num.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = table_num.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # ---- Save as three 6-column blocks (like printed layout) ----
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("Missing rows require explicit coding in the extract (e.g., [NA(d)] and [NA(n)]). If absent, all unspecified missing are shown under 'No answer'.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            cols_with_stub = ["Attitude"] + cols
            f.write(display.loc[:, cols_with_stub].to_string(index=False))
            f.write("\n\n")

    return table_num