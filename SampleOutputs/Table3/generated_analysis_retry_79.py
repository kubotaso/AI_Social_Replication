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

    missing_labels = [
        "(M) Don't know much about it",
        "(M) No answer",
    ]

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        missing_labels[0],
        missing_labels[1],
        "Mean",
    ]

    # -----------------------
    # Helpers
    # -----------------------
    def _is_null_like(s: pd.Series) -> pd.Series:
        # treat empty/whitespace as missing too
        if s.dtype == "O" or pd.api.types.is_string_dtype(s):
            ss = s.astype("string")
            return ss.isna() | (ss.str.strip() == "")
        return s.isna()

    def _explicit_missing_masks(raw: pd.Series):
        """
        Attempt to identify DK vs No-answer if the CSV preserved either:
          - literal tokens like '[NA(d)]', '[NA(n)]'
          - plain 'NA(d)' / 'NA(n)'
          - strings containing those tokens
          - common text labels (dk/don't know; no answer/na)
        Returns (dk_mask, na_mask, has_any_explicit)
        """
        s = raw.astype("string")

        # base null-like (true NaN / empty) -- unclassifiable unless explicit split exists
        null_like = _is_null_like(raw)

        up = s.str.upper().str.strip()

        # Tokens from GSS-style labeled missings
        dk_token = up.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na_token = up.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Common textual exports
        dk_text = up.str.contains(r"\bDON['’]T KNOW\b|\bDONT KNOW\b|\bDK\b", regex=True, na=False)
        na_text = up.str.contains(r"\bNO ANSWER\b|\bNA\b", regex=True, na=False)

        dk_mask = dk_token | dk_text
        na_mask = na_token | na_text

        has_any_explicit = bool((dk_mask | na_mask).any())

        # Explicit masks only apply where raw isn't a valid numeric 1..5;
        # we will enforce that later in _tabulate_one.
        return dk_mask, na_mask, has_any_explicit, null_like

    def _tabulate_one(raw: pd.Series, varname: str):
        """
        Returns:
          counts_1_5: pd.Series indexed 1..5
          dk_n: int
          na_n: int
          mean_val: float
        """
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        dk_mask, na_mask, has_any_explicit, null_like = _explicit_missing_masks(raw)

        # Any non-1..5 response that is not null-like might be a special numeric code;
        # but in this export it's likely already NaN. We'll check.
        nonvalid_mask = valid.isna()
        # Identify "observed but nonvalid" numeric values (e.g., 8/9) if present
        observed_nonvalid_numeric = x.notna() & nonvalid_mask & ~x.isin([1, 2, 3, 4, 5])

        # Constrain explicit DK/NA masks to nonvalid cells only
        dk_mask = dk_mask & nonvalid_mask
        na_mask = na_mask & nonvalid_mask

        # If both explicit exist, we can classify remaining nonvalid/null_like as "No answer"
        # ONLY if we have a rule; but Table 3 requires exact split and the export may not support it.
        unclassified = nonvalid_mask & ~(dk_mask | na_mask)

        if has_any_explicit:
            # If we have explicit codes for at least some, assume any remaining missing are "No answer"
            # ONLY when those remaining are null-like (blank/NaN), not weird numeric codes.
            # Weird numeric codes should not happen; if they do, treat as No answer (still missing).
            na_mask = na_mask | unclassified
            dk_mask = dk_mask  # unchanged
        else:
            # No explicit split in the export; cannot recover DK vs No-answer from raw data.
            # This export collapses them into generic missing (NaN) -> unclassifiable.
            total_nonvalid = int(nonvalid_mask.sum())
            if total_nonvalid > 0:
                examples = []
                # collect a few example tallies
                examples.append(f"{varname}({total_nonvalid})")
                raise ValueError(
                    "Cannot compute separate '(M) Don't know much about it' vs '(M) No answer' counts: "
                    "this CSV export does not preserve distinguishable missing categories (e.g., '[NA(d)]' "
                    "and '[NA(n)]' or distinct numeric codes). "
                    f"Found {total_nonvalid} missing/non-1..5 values for {varname}. "
                    "Re-export data preserving DK vs No-answer codes."
                )

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
    genre_labels = [g for g, _ in genre_map]
    table = pd.DataFrame(index=row_labels, columns=genre_labels, dtype="float64")

    for genre_label, var in genre_map:
        if var not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {var}")
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var], var)

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc[missing_labels[0], genre_label] = dk_n
        table.loc[missing_labels[1], genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Write human-readable text (3 blocks of 6)
    # -----------------------
    formatted = table.copy()
    for r in formatted.index:
        if r == "Mean":
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]
    out_path = "./output/table3_frequency_distributions_gss1993.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table