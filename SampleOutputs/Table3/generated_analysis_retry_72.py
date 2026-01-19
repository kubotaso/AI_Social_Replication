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

    missing_code_map = {
        "DK": {"[NA(D)]", "NA(D)", "NAD", "DK", "DON'T KNOW", "DONT KNOW", "DON’T KNOW"},
        "NA": {"[NA(N)]", "NA(N)", "NAN", "NO ANSWER", "NOANSWER", "NA", "N/A"},
    }

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        "(M) Don\u2019t know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Helpers
    # -----------------------
    def _norm_str_series(s):
        # Normalize to comparable tokens: uppercase, strip, normalize apostrophes
        out = s.astype("string")
        out = out.str.replace("\u2019", "'", regex=False)
        out = out.str.strip().str.upper()
        return out

    def _detect_explicit_missing(raw):
        """
        Returns (dk_mask, na_mask, explicit_found)
        Detects explicit DK/NA encodings if present as strings.
        """
        s = _norm_str_series(raw)

        dk_tokens = set(missing_code_map["DK"])
        na_tokens = set(missing_code_map["NA"])

        # token-based matches
        dk_mask = s.isin(dk_tokens) | s.str.contains(r"\[NA\(D\)\]", regex=True, na=False) | s.str.contains(r"\bNA\(D\)\b", regex=True, na=False)
        na_mask = s.isin(na_tokens) | s.str.contains(r"\[NA\(N\)\]", regex=True, na=False) | s.str.contains(r"\bNA\(N\)\b", regex=True, na=False)

        explicit_found = bool(int(dk_mask.sum()) + int(na_mask.sum()) > 0)
        return dk_mask, na_mask, explicit_found

    def _tabulate_one(raw, varname):
        """
        Tabulate counts for 1..5 plus separate DK/NA.
        If DK/NA are not distinguishable in the CSV export, infer them by:
          NA_count = number of rows where the corresponding respondent has any non-missing
                     value on at least one of the other music items but this item is missing.
          DK_count = remaining missing for this item.
        This identification is possible here because the GSS battery tends to show item-nonresponse
        as sparse "holes" across the battery, whereas "don't know much" is item-specific.
        """
        # numeric for 1..5
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # explicit missing codes if present
        dk_exp, na_exp, explicit_found = _detect_explicit_missing(raw)

        # missing pool = not valid (includes NaN and any non-1..5 codes)
        miss_pool = valid.isna()

        if explicit_found:
            # If explicit is present, use it; any remaining missing goes to DK by default.
            other_missing = miss_pool & ~(dk_exp | na_exp)
            dk = dk_exp | other_missing
            na = na_exp
            return valid, dk, na

        # If not explicit: infer NA via "item-missing while other battery items observed"
        # Build a "has any other genre response observed" indicator once outside? We do it here
        # by using df-level cached mask built in outer scope.
        raise RuntimeError("INTERNAL: _tabulate_one requires inferred-na mask from outer scope.")

    # -----------------------
    # Prepare inferred "No answer" masks (only used if explicit codes absent)
    # -----------------------
    # For each item, define NA as missing on that item AND has at least one other genre answered (1..5) OR coded missing explicitly (if any).
    # We compute "answered_any_other" based on valid 1..5 only to avoid circularity.
    vars_present = []
    for _, v in genre_map:
        if v not in df.columns:
            raise ValueError(f"Required genre variable not found in data: {v}")
        vars_present.append(v)

    # Matrix of valid responses (1..5) for each item
    valid_mat = {}
    explicit_any = False
    for v in vars_present:
        raw = df[v]
        x = pd.to_numeric(raw, errors="coerce")
        valid_mat[v] = x.isin([1, 2, 3, 4, 5])
        dk_exp, na_exp, explicit_found = _detect_explicit_missing(raw)
        if explicit_found:
            explicit_any = True

    # Precompute "any answered in battery" to support inference
    # (True if respondent answered at least one item in the battery with 1..5)
    any_answered_battery = None
    if len(vars_present) > 0:
        any_answered_battery = pd.concat([valid_mat[v] for v in vars_present], axis=1).any(axis=1)
    else:
        any_answered_battery = pd.Series(False, index=df.index)

    # For each item, compute answered_any_other
    answered_any_other = {}
    for v in vars_present:
        others = [vv for vv in vars_present if vv != v]
        if others:
            answered_any_other[v] = pd.concat([valid_mat[vv] for vv in others], axis=1).any(axis=1)
        else:
            answered_any_other[v] = pd.Series(False, index=df.index)

    # -----------------------
    # Build table
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")

    for genre_label, var in genre_map:
        raw = df[var]

        # numeric for 1..5
        x = pd.to_numeric(raw, errors="coerce")
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        miss_pool = valid.isna()

        # explicit missing codes if present
        dk_exp, na_exp, explicit_found = _detect_explicit_missing(raw)

        if explicit_found:
            other_missing = miss_pool & ~(dk_exp | na_exp)
            dk_mask = dk_exp | other_missing
            na_mask = na_exp
        else:
            # Inferred split:
            # - NA (No answer) := missing on this item but answered at least one OTHER item in the battery
            # - DK (Don't know much) := remaining missing on this item (including those who did not answer other items)
            na_mask = miss_pool & answered_any_other[var]
            dk_mask = miss_pool & ~na_mask

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
        table.loc["(M) Don\u2019t know much about it", genre_label] = int(dk_mask.sum())
        table.loc["(M) No answer", genre_label] = int(na_mask.sum())
        table.loc["Mean", genre_label] = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan

    # -----------------------
    # Format + write text output (3 blocks of 6)
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
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n")
        if not explicit_any:
            f.write(
                "Note: CSV export does not preserve explicit DK vs No-answer codes; the two (M) rows are inferred.\n"
                "      No answer is defined as item-missing while at least one other music item is answered (1–5).\n\n"
            )
        else:
            f.write("Note: DK/No-answer counts use explicit missing codes when present; otherwise remaining missing are inferred.\n\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table