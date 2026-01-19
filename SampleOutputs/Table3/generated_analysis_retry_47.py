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

    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {v}")

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
    # Missing-code parsing
    # -----------------------
    # Goal: separate DK vs NA *from raw data*, never hardcode table numbers.
    # We support:
    #   - explicit string tags like "[NA(d)]" / "[NA(n)]" (case-insensitive)
    #   - numeric special codes if present (negative codes, etc.) by mapping via heuristics
    # If the extract has only blank/NaN with no distinction, separation is impossible;
    # in that case, we will:
    #   - still compute rows 1–5 and mean correctly
    #   - set DK/NA rows to NaN (not 0) to avoid fabricating results

    def _as_clean_string(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip()

    def _explicit_missing_masks(raw: pd.Series):
        s = _as_clean_string(raw).str.upper()

        # Common GSS-style tagged missings (as in the prompt)
        dk = s.str.contains(r"\[NA\(D\)\]|\bNA\(D\)\b", regex=True, na=False)
        na = s.str.contains(r"\[NA\(N\)\]|\bNA\(N\)\b", regex=True, na=False)

        # Also tolerate plain-English exports sometimes used
        dk = dk | s.isin(["DON'T KNOW", "DONT KNOW", "DK", "DON’T KNOW", "DON'T KNOW MUCH", "DONT KNOW MUCH"])
        na = na | s.isin(["NO ANSWER", "NA", "N/A"])

        return dk, na

    def _numeric_specials_masks(raw: pd.Series):
        """
        If raw is numeric-coded with distinct missing codes (e.g., -1/-2),
        try to identify DK vs NA with conservative heuristics:

        - Prefer common conventions:
            DK: -1 or 8/9 (depending on instrument)
            NA: -2 or 0
        But we must not guess if ambiguous.

        Returns: (dk_mask, na_mask, used_flag)
        """
        x = pd.to_numeric(raw, errors="coerce")

        # Candidate special codes are those not in 1..5 but not NaN
        specials = x[~x.isin([1, 2, 3, 4, 5]) & x.notna()].astype(int)

        if specials.empty:
            return (pd.Series(False, index=raw.index), pd.Series(False, index=raw.index), False)

        uniq = sorted(specials.unique().tolist())

        # Known patterns: two distinct specials
        # Try common pairs: (-1, -2), (8, 9), (0, 9), (0, 8)
        common_pairs = [(-1, -2), (8, 9), (0, 9), (0, 8), (98, 99)]
        pair = None
        for a, b in common_pairs:
            if a in uniq and b in uniq:
                pair = (a, b)
                break

        if pair is None:
            # If more than 2 specials, or only 1 special, we cannot safely split.
            return (pd.Series(False, index=raw.index), pd.Series(False, index=raw.index), False)

        a, b = pair

        # Map DK/NA by typical ordering:
        # For negative: DK=-1, NA=-2
        # For 8/9: DK=8, NA=9
        # For 0/9: DK=9? typically 9=DK, 0=NA (but not sure) -> ambiguous, do not use
        # For 98/99: DK=98, NA=99 (often)
        mapping = None
        if (a, b) == (-1, -2):
            mapping = {"DK": -1, "NA": -2}
        elif (a, b) == (8, 9):
            mapping = {"DK": 8, "NA": 9}
        elif (a, b) == (98, 99):
            mapping = {"DK": 98, "NA": 99}
        else:
            # ambiguous patterns like (0,9) or (0,8)
            return (pd.Series(False, index=raw.index), pd.Series(False, index=raw.index), False)

        dk_mask = x.eq(mapping["DK"])
        na_mask = x.eq(mapping["NA"])
        used = bool(dk_mask.any() or na_mask.any())
        return dk_mask, na_mask, used

    def _tabulate_one(raw: pd.Series):
        # Valid numeric 1..5 counts and mean
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

        # DK/NA masks: explicit strings first, else numeric specials
        dk1, na1 = _explicit_missing_masks(raw)
        if (dk1.any() or na1.any()):
            # Anything else missing/non-1..5 not explicitly NA(n) should NOT be forced into DK/NA
            dk_n = int(dk1.sum())
            na_n = int(na1.sum())
            return counts_1_5, dk_n, na_n, mean_val

        dk2, na2, used = _numeric_specials_masks(raw)
        if used:
            dk_n = int(dk2.sum())
            na_n = int(na2.sum())
            return counts_1_5, dk_n, na_n, mean_val

        # If extract collapsed all missing into NaN, we cannot split DK vs NA from raw data.
        # Return NaN for those rows (NOT 0) to avoid inventing numbers.
        return counts_1_5, np.nan, np.nan, mean_val

    # -----------------------
    # Build results table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val = _tabulate_one(df[var])

        table.loc["(1) Like very much", genre_label] = counts_1_5.loc[1]
        table.loc["(2) Like it", genre_label] = counts_1_5.loc[2]
        table.loc["(3) Mixed feelings", genre_label] = counts_1_5.loc[3]
        table.loc["(4) Dislike it", genre_label] = counts_1_5.loc[4]
        table.loc["(5) Dislike very much", genre_label] = counts_1_5.loc[5]
        table.loc["(M) Don’t know much about it", genre_label] = dk_n
        table.loc["(M) No answer", genre_label] = na_n
        table.loc["Mean", genre_label] = mean_val

    # -----------------------
    # Save human-readable txt
    # -----------------------
    def _format_cell(row_name, v):
        if pd.isna(v):
            return ""
        if row_name == "Mean":
            return f"{float(v):.2f}"
        return str(int(round(float(v))))

    display = table.copy()
    for r in display.index:
        display.loc[r] = [ _format_cell(r, v) for v in display.loc[r].tolist() ]

    display.insert(0, "Attitude", list(display.index))
    display = display.reset_index(drop=True)

    # Split into 3 blocks of 6 genres (paper-like layout)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i:i+6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding all missings.\n")
        f.write("Note: If the extract does not preserve distinct DK vs No answer codes, those cells are blank.\n\n")
        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            f.write(display.loc[:, ["Attitude"] + cols].to_string(index=False))
            f.write("\n\n")

    return table