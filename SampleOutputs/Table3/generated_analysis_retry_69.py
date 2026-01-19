def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Load + filter
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
        "(M) Don't know much about it",
        "(M) No answer",
        "Mean",
    ]

    # -----------------------
    # Missing category detection
    # -----------------------
    # We must compute DK vs NA from raw data. Many CSV exports collapse all missings into NaN.
    # In that case, DK vs NA cannot be inferred deterministically; we fall back to:
    #   - count DK/NA from explicit string codes if present
    #   - else (if only NaN), label all missing as DK and NA=0, and clearly note this in output.
    #
    # This avoids runtime errors and "all-zero rows". It also keeps mean computed on 1..5 only.

    def _as_string(s):
        return s.astype("string")

    def _explicit_missing_masks(raw):
        s = _as_string(raw).str.strip().str.lower()

        # Common explicit forms from some GSS exports/codebooks
        dk_tokens = [
            "[na(d)]", "na(d)", "dk", "dont know", "don't know", "don’t know",
            "dont know much", "don't know much", "don’t know much",
        ]
        na_tokens = [
            "[na(n)]", "na(n)", "no answer", "noans", "na", "n/a",
        ]
        refused_tokens = ["[na(r)]", "na(r)", "refused"]
        skipped_tokens = ["[na(s)]", "na(s)", "skipped"]
        iap_tokens = ["[na(i)]", "na(i)", "iap"]
        misc_missing_tokens = ["[na(m)]", "na(m)", "[na(x)]", "na(x)", "[na(y)]", "na(y)", "[na(z)]", "na(z)"]

        def contains_any(tokens):
            mask = pd.Series(False, index=raw.index)
            for t in tokens:
                mask = mask | s.str.contains(rf"{pd.regex.escape(t)}", regex=True, na=False)
            return mask

        dk = contains_any(dk_tokens)
        na = contains_any(na_tokens)
        refused = contains_any(refused_tokens)
        skipped = contains_any(skipped_tokens)
        iap = contains_any(iap_tokens)
        misc = contains_any(misc_missing_tokens)

        # Non-overlapping priority: NA(n) first, DK next, then other missings treated as NA
        na_final = na | refused | skipped | iap | misc
        dk_final = dk & ~na_final

        return dk_final, na_final

    def _tabulate_one(raw):
        # Numeric parse
        x = pd.to_numeric(raw, errors="coerce")

        # Valid substantive responses: 1..5
        valid = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

        # Explicit missing codes if present as strings
        dk_mask_exp, na_mask_exp = _explicit_missing_masks(raw)

        # "Other" missing = not valid and not explicitly classified
        other_missing = valid.isna() & ~(dk_mask_exp | na_mask_exp)

        # If explicit codes exist anywhere in this column, treat unclassified missings as NA (conservative)
        if (dk_mask_exp.sum() + na_mask_exp.sum()) > 0:
            dk_mask = dk_mask_exp
            na_mask = na_mask_exp | other_missing
            missing_note = "Missing split from explicit codes where available; unclassified missing -> No answer."
        else:
            # No explicit codes: all missings are indistinguishable. Put them into DK and NA=0.
            dk_mask = other_missing
            na_mask = pd.Series(False, index=raw.index)
            missing_note = "WARNING: DK vs No answer not distinguishable in this CSV; all missing counted as DK, No answer=0."

        counts_1_5 = (
            valid.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(int)
        )

        dk_n = int(dk_mask.sum())
        na_n = int(na_mask.sum())

        mean_val = float(valid.mean(skipna=True)) if valid.notna().any() else np.nan
        return counts_1_5, dk_n, na_n, mean_val, missing_note

    # -----------------------
    # Build table (numeric)
    # -----------------------
    table = pd.DataFrame(index=row_labels, columns=[g for g, _ in genre_map], dtype="float64")
    notes = {}

    for genre_label, var in genre_map:
        counts_1_5, dk_n, na_n, mean_val, missing_note = _tabulate_one(df[var])
        notes[genre_label] = missing_note

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
            formatted.loc[r] = formatted.loc[r].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))

    display = formatted.copy()
    display.insert(0, "Attitude", display.index)
    display = display.reset_index(drop=True)

    out_path = "./output/table3_frequency_distributions_gss1993.txt"
    genre_labels = [g for g, _ in genre_map]
    blocks = [genre_labels[i : i + 6] for i in range(0, len(genre_labels), 6)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993\n")
        f.write("Counts shown for response categories; Mean computed on 1–5 excluding missing.\n\n")

        # Notes about missing split limitations
        unique_notes = sorted(set(notes.values()))
        f.write("Missing-category handling notes:\n")
        for n in unique_notes:
            f.write(f"- {n}\n")
        f.write("\n")

        for bi, cols in enumerate(blocks, start=1):
            f.write(f"Block {bi}:\n")
            block_df = display.loc[:, ["Attitude"] + cols]
            f.write(block_df.to_string(index=False))
            f.write("\n\n")

    return table