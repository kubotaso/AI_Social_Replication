def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats

    os.makedirs("./output", exist_ok=True)

    def _read_csv(path):
        df0 = pd.read_csv(path)
        # normalize column names to lowercase for robustness
        df0.columns = [c.strip().lower() for c in df0.columns]
        return df0

    def _to_numeric(s):
        return pd.to_numeric(s, errors="coerce")

    def _na_if_outside(series, valid_values):
        # keep only values in valid_values; else NaN
        s = _to_numeric(series)
        return s.where(s.isin(valid_values), np.nan)

    def _dislike_indicator(item_series):
        # 1 if {4,5}; 0 if {1,2,3}; missing otherwise
        s = _na_if_outside(item_series, [1, 2, 3, 4, 5])
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin([4, 5])] = 1.0
        out.loc[s.isin([1, 2, 3])] = 0.0
        return out

    def _binary_from_twolevel(series, one_value, zero_value):
        s = _to_numeric(series)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s == one_value] = 1.0
        out.loc[s == zero_value] = 0.0
        return out

    def _safe_percap_income(realinc, hompop):
        r = _to_numeric(realinc)
        h = _to_numeric(hompop)
        # avoid division by 0; also treat nonpositive household size as missing
        h = h.where(h > 0, np.nan)
        return r / h

    def _zscore_series(s):
        s = _to_numeric(s)
        m = np.nanmean(s.values)
        sd = np.nanstd(s.values, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - m) / sd

    def _standardized_ols(df, dv, xcols, model_name):
        # listwise deletion on dv and xcols
        use_cols = [dv] + xcols
        d = df[use_cols].copy()
        # ensure numeric
        for c in use_cols:
            d[c] = _to_numeric(d[c])

        # drop inf/-inf
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(
                f"{model_name}: not enough complete cases after cleaning "
                f"(n={d.shape[0]}, k={len(xcols)})."
            )

        y = _zscore_series(d[dv])
        X = pd.DataFrame(index=d.index)

        # Standardize all predictors to match "standardized coefficients" mechanically.
        for c in xcols:
            X[c] = _zscore_series(d[c])

        # After z-scoring, drop any rows that became NaN (e.g., constant predictors)
        ZX = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if ZX.shape[0] < (len(xcols) + 2):
            raise ValueError(
                f"{model_name}: not enough cases after standardization cleaning "
                f"(n={ZX.shape[0]}, k={len(xcols)})."
            )

        y2 = ZX["y"].astype(float)
        X2 = ZX.drop(columns=["y"]).astype(float)
        X2 = sm.add_constant(X2, has_constant="add")

        model = sm.OLS(y2, X2).fit()

        # Build table with standardized betas (these are the coefficients in this regression)
        res = pd.DataFrame(
            {
                "beta": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        res.index.name = "term"

        fit = pd.DataFrame(
            {
                "n": [int(model.nobs)],
                "r2": [float(model.rsquared)],
                "adj_r2": [float(model.rsquared_adj)],
                "df_model": [float(model.df_model)],
                "df_resid": [float(model.df_resid)],
            }
        )

        # Pretty text output
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

        # Also write a compact coefficient table
        res_rounded = res.copy()
        for col in ["beta", "std_err", "t", "p_value"]:
            res_rounded[col] = res_rounded[col].astype(float)
        res_rounded.to_string_buf = None
        with open(f"./output/{model_name}_coefficients.txt", "w", encoding="utf-8") as f:
            f.write(res_rounded.to_string(float_format=lambda x: f"{x: .4f}"))
            f.write("\n\n")
            f.write(fit.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

        return model, res, fit, ZX.index

    # -----------------------
    # Load data and construct variables
    # -----------------------
    df = _read_csv(data_source)

    # Filter to 1993
    if "year" in df.columns:
        df = df.loc[_to_numeric(df["year"]) == 1993].copy()
    else:
        raise ValueError("Column 'year' not found; cannot filter to 1993.")

    # Music items lists (lowercase columns)
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal",
    ]

    for col in minority_items + other12_items:
        if col not in df.columns:
            raise ValueError(f"Required music field '{col}' not found in dataset.")

    # Construct dislike counts with "any NA-coded -> missing for that item" already handled;
    # DV is sum across available indicators, then require complete set (listwise across items)
    # to match "missing cases excluded" practice.
    for col in minority_items + other12_items:
        df[f"dislike_{col}"] = _dislike_indicator(df[col])

    # Require all items present for each DV (strict listwise within DV)
    df["dislike_minority_genres"] = (
        df[[f"dislike_{c}" for c in minority_items]]
        .sum(axis=1, min_count=len(minority_items))
    )
    df["dislike_other12_genres"] = (
        df[[f"dislike_{c}" for c in other12_items]]
        .sum(axis=1, min_count=len(other12_items))
    )

    # Racism components
    needed_race_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for col in needed_race_items:
        if col not in df.columns:
            raise ValueError(f"Required racism item '{col}' not found in dataset.")

    df["rac1"] = _binary_from_twolevel(df["rachaf"], one_value=1, zero_value=2)
    df["rac2"] = _binary_from_twolevel(df["busing"], one_value=2, zero_value=1)
    df["rac3"] = _binary_from_twolevel(df["racdif1"], one_value=2, zero_value=1)
    df["rac4"] = _binary_from_twolevel(df["racdif3"], one_value=2, zero_value=1)
    df["rac5"] = _binary_from_twolevel(df["racdif4"], one_value=1, zero_value=2)

    df["racism_score"] = (
        df[["rac1", "rac2", "rac3", "rac4", "rac5"]]
        .sum(axis=1, min_count=5)
    )

    # Controls
    for col in ["educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region"]:
        if col not in df.columns:
            # denom might be missing in some extracts; but here it's in available variables
            raise ValueError(f"Required control field '{col}' not found in dataset.")

    df["education_years"] = _to_numeric(df["educ"])
    df["hh_income_per_capita"] = _safe_percap_income(df["realinc"], df["hompop"])
    df["occ_prestige"] = _to_numeric(df["prestg80"])

    # Female: SEX 1 male, 2 female
    df["female"] = _binary_from_twolevel(df["sex"], one_value=2, zero_value=1)

    df["age_years"] = _to_numeric(df["age"])

    # Race dummies from RACE: 1 white, 2 black, 3 other
    r = _to_numeric(df["race"])
    df["black"] = pd.Series(np.nan, index=df.index, dtype="float64")
    df["other_race"] = pd.Series(np.nan, index=df.index, dtype="float64")
    df.loc[r.isin([1, 2, 3]), "black"] = (r.loc[r.isin([1, 2, 3])] == 2).astype(float)
    df.loc[r.isin([1, 2, 3]), "other_race"] = (r.loc[r.isin([1, 2, 3])] == 3).astype(float)

    # Hispanic not available: include as missing entirely would drop all cases; instead omit.
    # To keep models runnable and faithful to available data, we exclude 'hispanic' from RHS.
    # (Saved in notes output.)
    # Religion dummies
    rel = _to_numeric(df["relig"])
    den = _to_numeric(df["denom"])

    # Conservative Protestant proxy:
    # RELIG==1 Protestant AND DENOM in {1 Baptist, 6 Other, 7 None/non-denom}
    df["cons_protestant"] = pd.Series(np.nan, index=df.index, dtype="float64")
    valid_rel = rel.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 98, 99])  # broad; will be set below
    # apply only where rel and denom are finite
    mask_rd = rel.notna() & den.notna()
    df.loc[mask_rd, "cons_protestant"] = (
        (rel.loc[mask_rd] == 1) & (den.loc[mask_rd].isin([1, 6, 7]))
    ).astype(float)

    # No religion: RELIG==4
    df["no_religion"] = pd.Series(np.nan, index=df.index, dtype="float64")
    mask_r = rel.notna()
    df.loc[mask_r, "no_religion"] = (rel.loc[mask_r] == 4).astype(float)

    # South: REGION==3
    reg = _to_numeric(df["region"])
    df["south"] = pd.Series(np.nan, index=df.index, dtype="float64")
    mask_reg = reg.notna()
    df.loc[mask_reg, "south"] = (reg.loc[mask_reg] == 3).astype(float)

    # -----------------------
    # Fit models (standardized OLS)
    # -----------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Diagnostics notes
    notes = []
    notes.append("Table 2 replication attempt using 1993 GSS extract provided.")
    notes.append("Note: 'Hispanic' indicator is not available in provided data; model omits it.")
    notes.append("Standardized coefficients computed by z-scoring DV and all predictors (including dummies).")
    notes.append("Listwise deletion applied per model (DV + all predictors).")
    notes_text = "\n".join(notes)
    with open("./output/Table2_notes.txt", "w", encoding="utf-8") as f:
        f.write(notes_text + "\n")

    results = {}

    # Model A
    mA, tabA, fitA, idxA = _standardized_ols(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    results["ModelA_coefficients"] = tabA
    results["ModelA_fit"] = fitA

    # Model B
    mB, tabB, fitB, idxB = _standardized_ols(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )
    results["ModelB_coefficients"] = tabB
    results["ModelB_fit"] = fitB

    # Combined text summary
    def _format_sig(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def _compact_table(tab, fit, title):
        # drop constant for display of standardized betas (constant is not meaningful after z-scoring)
        t = tab.copy()
        if "const" in t.index:
            t = t.drop(index="const")
        lines = [title]
        lines.append(f"N={int(fit.loc[0,'n'])}  R2={fit.loc[0,'r2']:.3f}  Adj.R2={fit.loc[0,'adj_r2']:.3f}")
        lines.append("term\tbeta\tp")
        for term, row in t.iterrows():
            lines.append(f"{term}\t{row['beta']:.4f}{_format_sig(row['p_value'])}\t{row['p_value']:.4g}")
        return "\n".join(lines)

    combined = []
    combined.append(notes_text)
    combined.append("")
    combined.append(_compact_table(tabA, fitA, "Model A: DV=dislike_minority_genres (6 items)"))
    combined.append("")
    combined.append(_compact_table(tabB, fitB, "Model B: DV=dislike_other12_genres (12 items)"))
    combined_text = "\n".join(combined)

    with open("./output/Table2_combined_results.txt", "w", encoding="utf-8") as f:
        f.write(combined_text + "\n")

    # Return dict of DataFrames for programmatic access
    return results