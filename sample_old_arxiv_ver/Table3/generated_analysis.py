def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd

    # ---- I/O ----
    df = pd.read_csv(data_source)

    # ---- Restrict to GSS 1993 ----
    year_col = "year" if "year" in df.columns else "YEAR" if "YEAR" in df.columns else None
    if year_col is None:
        raise KeyError("YEAR/year column not found.")
    df = df.loc[df[year_col] == 1993].copy()

    # ---- Variables (dataset uses lowercase) ----
    genres = [
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

    missing_val_labels = {
        "dk": "(M) Don’t know much about it",
        "na": "(M) No answer",
    }

    row_labels = [
        "(1) Like very much",
        "(2) Like it",
        "(3) Mixed feelings",
        "(4) Dislike it",
        "(5) Dislike very much",
        missing_val_labels["dk"],
        missing_val_labels["na"],
        "Mean",
    ]

    # ---- Helper to interpret GSS-style missing codes if present ----
    def _count_missing_types(s: pd.Series):
        """
        Returns (dk_count, na_count) based on:
          - explicit strings: 'd', 'n', 'dk', 'na', 'dont know', 'no answer'
          - common numeric conventions in some extracts (defensive): 8/9, 98/99, 0
        If none match, counts will be 0 and missing remain as NaN for mean.
        """
        dk = 0
        na = 0

        # If object: try string patterns
        if s.dtype == "object":
            ss = s.astype("string")
            low = ss.str.lower().str.strip()

            dk_mask = low.isin(["d", "dk", "don't know", "dont know", "dontknow", "don’t know", "dont know much", "don't know much"])
            na_mask = low.isin(["n", "na", "no answer", "noanswer"])

            dk = int(dk_mask.sum(skipna=True))
            na = int(na_mask.sum(skipna=True))
            return dk, na

        # If numeric: attempt common non-substantive codes (best-effort)
        sn = pd.to_numeric(s, errors="coerce")
        # Substantive expected: 1..5
        # Common missing patterns in some GSS exports: 8/9 or 98/99 (dk/na), sometimes 0
        dk_mask = sn.isin([8, 98])
        na_mask = sn.isin([9, 99])
        dk = int(dk_mask.sum(skipna=True))
        na = int(na_mask.sum(skipna=True))
        return dk, na

    # ---- Build table ----
    table = pd.DataFrame(index=row_labels, columns=[g[0] for g in genres], dtype="float")

    for genre_label, var in genres:
        if var not in df.columns:
            raise KeyError(f"Expected variable '{var}' not found in dataset.")

        s_raw = df[var]

        # Frequencies for 1..5
        s_num = pd.to_numeric(s_raw, errors="coerce")
        for k in [1, 2, 3, 4, 5]:
            table.loc[f"({k}) " + ["Like very much", "Like it", "Mixed feelings", "Dislike it", "Dislike very much"][k - 1], genre_label] = int((s_num == k).sum())

        # Missing categories: try to detect explicit dk/na codes; otherwise 0
        dk_count, na_count = _count_missing_types(s_raw)
        table.loc[missing_val_labels["dk"], genre_label] = dk_count
        table.loc[missing_val_labels["na"], genre_label] = na_count

        # Mean on 1..5 only
        mean_val = s_num.where(s_num.isin([1, 2, 3, 4, 5])).mean()
        table.loc["Mean", genre_label] = float(mean_val) if pd.notna(mean_val) else np.nan

    # Convert count rows to integers (nullable) and mean row to 3 decimals
    count_rows = [r for r in table.index if r != "Mean"]
    table.loc[count_rows, :] = table.loc[count_rows, :].round(0).astype("Int64")
    table.loc["Mean", :] = table.loc["Mean", :].astype(float).round(3)

    # ---- Save human-readable text ----
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/table3_frequency_distributions_gss1993.txt"

    # Format: align columns, keep integers without .0, mean with 3 decimals
    def _format_cell(rname, x):
        if pd.isna(x):
            return ""
        if rname == "Mean":
            return f"{float(x):.3f}"
        return str(int(x))

    formatted = table.copy()
    for r in formatted.index:
        formatted.loc[r, :] = [ _format_cell(r, v) for v in formatted.loc[r, :].tolist() ]

    title = "Table 3. Frequency Distributions for Attitude toward 18 Music Genres: General Social Survey, 1993"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(formatted.to_string())
        f.write("\n")

    return table