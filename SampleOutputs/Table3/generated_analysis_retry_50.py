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
    # Table 3 variable mapping
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

    # Verify required columns exist
    missing_vars = [v for _, v in genre_map if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required genre variables: {missing_vars}")

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
    # Missing-code handling
    # -----------------------
    # We must compute DK vs NA from the raw data, not hard-code paper numbers.
    # The CSV may encode these as:
    #   - strings like "[NA(d)]", "NA(d)", "[NA(n)]", "NA(n)", etc., OR
    #   - numeric distinct codes for DK/NA (rare), OR
    #   - plain NaN (unsplittable).
    #
    # Strategy:
    #   1) detect explicit DK/NA string tokens (covers many GSS exports);
    #   2) if not found, detect distinct numeric codes outside 1..5 consistently across vars;
    #   3) if still unsplittable, we do NOT error; we set DK/NA counts to NaN and
    #      write a warning in the output file. (Means and 1..5 counts remain correct.)
    #
    # This avoids the previous runtime errors while still "always computing from raw data".

    DK_TOKENS = {
        "[NA(D)]", "NA(D)", "NAD", "[NA(DK)]", "NA(DK)", "DK", "DON'T KNOW", "DONT KNOW",
        "DON'T KNOW MUCH ABOUT IT", "DONT KNOW MUCH ABOUT IT",
    }
    NA_TOKENS = {
        "[NA(N)]", "NA(N)", "NAN", "[NA(NA)]", "NA(NA)",
        "NO ANSWER", "NA", "NOT ANSWERED",
    }

    def _as_clean_string(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    def _extract_explicit_masks(raw: pd.Series):
        s = _as_clean_string(raw)
        # If raw is numeric, string conversion yields e.g. "1.0"; tokens won't match.
        dk_mask = s.isin(DK_TOKENS) | s.str.fullmatch(r"\[NA\(\s*D\s*\)\]", na=False) | s.str.fullmatch(r"NA\(\s*D\s*\)", na=False)
        na_mask = s.isin(NA_TOKENS) | s.str.fullmatch(r"\[NA\(\s*N\s*\)\]", na=False) | s.str.fullmatch(r"NA\(\s*N\s*\)", na=False)
        return dk_mask.fillna(False), na_mask.fillna(False)

    def _valid_numeric(raw: pd.Series) -> pd.Series:
        x = pd.to_numeric(raw, errors="coerce")
        return x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

    # Try to learn numeric missing codes across the dataset (outside 1..5), if present
    # We will scan all genre vars for repeated non-1..5 numeric codes.
    numeric_code_counts = {}
    for _, var in genre_map:
        x = pd.to_numeric(df[var], errors="coerce")
        non_valid = x[~x.isin([1, 2, 3, 4, 5]) & x.notna()]
        if not non_valid.empty:
            vc = non_valid.value_counts()
            for code, cnt in vc.items():
                numeric_code_counts[code] = numeric_code_counts.get(code, 0) + int(cnt)

    # Heuristic mapping of numeric codes to DK vs NA if we see at least 2 distinct codes
    # and they appear broadly. We keep it conservative: only map if there are 2+ codes
    # and each appears at least 30 times overall (to avoid misclassifying stray values).
    numeric_code_candidates = sorted(
        [(code, cnt) for code, cnt in numeric_code_counts.items() if cnt >= 30],
        key=lambda t: t[1],
        reverse=True,
    )
    numeric_dk_code = None
    numeric_na_code = None
    if len(numeric_code_candidates) >= 2:
        # Common convention in some GSS numeric coding: DK > NA or vice versa; unknown here.
        # We cannot assume which is which reliably, but we can still separate them deterministically.
        # Assign the most frequent to DK and second to NA.
        numeric_dk_code = numeric_code_candidates[0][0]
        numeric_na_code = numeric_code_candidates[1][0]

    def _split_missing(raw: pd.Series):
        valid = _valid_numeric(raw)

        dk_mask_s, na_mask_s = _extract_explicit_masks(raw)
        explicit_found = bool(dk_mask_s.any() or na_mask_s.any())

        if explicit_found:
            # Anything else non-valid and non-explicit is treated as generic missing; we fold into NA.
            other_missing = valid.isna() & ~(dk_mask_s | na_mask_s)
            na_mask = na_mask_s | other_missing
            dk_mask = dk_mask_s
            splittable = True
            method = "explicit_strings"
            return valid, dk_mask, na_mask, splittable, method

        # Try numeric-code separation
        x = pd.to_numeric(raw, errors="coerce")
        if numeric_dk_code is not None and numeric_na_code is not None:
            dk_mask_n = x.eq(numeric_dk_code)
            na_mask_n = x.eq(numeric_na_code)
            if dk_mask_n.any() or na_mask_n.any():
                other_missing = valid.isna() & ~(dk_mask_n | na_mask_n)
                # Fold any remaining missing into NA (conservative)
                na_mask = na_mask_n | other_missing
                dk_mask = dk_mask_n
                splittable = True
                method = f"numeric_codes(dk={numeric_dk_code},na={numeric_na_code})"
                return valid, dk_mask, na_mask, splittable, method

        # Unsplittable: we can still compute 1..5 counts and mean; DK/NA become NaN
        splittable = False
        method = "unsplittable_missing"
        dk_mask = pd.Series(False, index=raw.index)
        na_mask = pd.Series(False, index=raw.index)
        return valid, dk_mask, na_mask, splittable, method

    # -----------------------
    # Build Table 3
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")
    meta = []  # per-variable method notes

    for genre_label, var in genre_map:
        valid, dk_mask, na_mask, splittable, method = _split_missing(df[var])

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

        if splittable:
            table.loc["(M) Don't know much about it", genre_label] = int(dk_mask.sum())
            table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        else:
            table.loc["(M) Don't know much about it", genre_label] = np.nan
            table.loc["(M) No answer", genre_label] = np.nan

        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        meta.append((genre_label, var, method, int(valid.notna().sum()), int(valid.isna().sum())))

    # -----------------------
    # Human-readable output: 3 panels of 6 genres
    # -----------------------
    def _format_table_block(block_cols):
        formatted = table.loc[:, block_cols].copy()

        # integer rows for counts, 2 decimals for mean
        for r in formatted.index:
            if r == "Mean":
                formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
            else:
                def fmt_int(v):
                    if pd.isna(v):
                        return ""
                    return str(int(round(float(v))))
                formatted.loc[r] = formatted.loc[r].map(fmt_int)

        formatted.insert(0, "Attitude", formatted.index)
        formatted = formatted.reset_index(drop=True)
        return formatted

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        f.write("\nIMPORTANT:\n")
        f.write("The CSV extract may not preserve distinct missing categories for '(M) Don't know much about it' vs '(M) No answer'.\n")
        f.write("If those categories are not explicitly encoded (e.g., as '[NA(d)]' and '[NA(n)]' strings or distinct numeric codes),\n")
        f.write("their counts cannot be separated from the raw file and are left blank.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Panel {bi} (6 genres):\n")
            block_df = _format_table_block(cols)
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

        # Diagnostics: how missing was handled per variable
        f.write("Diagnostics (per genre variable):\n")
        diag = pd.DataFrame(
            meta,
            columns=["Genre", "Variable", "Missing split method", "Valid N (1-5)", "Non-1..5 or missing N"],
        )
        f.write(diag.to_string(index=False))
        f.write("\n")

    return table