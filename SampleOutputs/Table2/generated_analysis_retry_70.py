def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing cleaning for this extract:
        - Coerce to numeric
        - Treat common GSS missing codes as NaN (8/9, 98/99, 998/999, 9998/9999)
        """
        x = to_num(x).copy()
        miss = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(miss))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 scale, 4/5 indicate dislike.
        - 1,2,3 => 0
        - 4,5 => 1
        - else => missing
        """
        x = clean_gss_missing(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_allow_partial(df, items, min_answered):
        """
        Build dislike count, allowing partial completion:
        - each item -> dislike indicator (0/1/NaN)
        - count = sum across answered items
        - require at least min_answered non-missing items
        """
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        answered = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, min_count=1)
        count = count.where(answered >= min_answered)
        return count

    def standardize_series(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_unstd_and_betas(df, dv, xcols, model_name):
        """
        Fit unstandardized OLS (with intercept) on listwise-complete data for dv+xcols.
        Compute standardized betas for predictors: beta_j = b_j * sd(x_j)/sd(y)
        (Intercept is reported unstandardized only.)
        """
        cols = [dv] + xcols
        d = df[cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: too few complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)
        X = sm.add_constant(d[xcols].astype(float), has_constant="add")

        m = sm.OLS(y, X).fit()

        # standardized betas (for non-constant terms only)
        y_sd = y.std(ddof=0)
        betas = {}
        for c in xcols:
            x_sd = d[c].astype(float).std(ddof=0)
            if not np.isfinite(x_sd) or x_sd <= 0 or not np.isfinite(y_sd) or y_sd <= 0:
                betas[c] = np.nan
            else:
                betas[c] = float(m.params[c] * (x_sd / y_sd))

        # Stars based on p-values from the unstandardized model
        def stars(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        table = pd.DataFrame(
            {
                "std_beta": pd.Series(betas),
                "p_value": pd.Series({c: float(m.pvalues[c]) for c in xcols}),
            }
        )
        table["std_beta_star"] = table["std_beta"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + table["p_value"].map(stars)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m.nobs),
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                    "constant_unstd": float(m.params["const"]),
                    "constant_p": float(m.pvalues["const"]),
                }
            ]
        )

        # Also keep a full unstandardized coefficient table (replication output; not in Bryson table)
        full = pd.DataFrame(
            {
                "b": m.params,
                "std_err": m.bse,
                "t": m.tvalues,
                "p_value": m.pvalues,
            }
        )

        return m, table, full, fit, d.index

    def write_text(path, s):
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)

    # -------------------------
    # Load and filter
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables (counts)
    # Use partial completion to avoid collapsing N.
    # Require at least 5/6 for minority set, and at least 10/12 for other set.
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_allow_partial(df, minority_items, min_answered=5)
    df["dislike_other12_genres"] = build_count_allow_partial(df, other12_items, min_answered=10)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing realinc and/or hompop column.")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: black, other (white is reference)
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC as a pragmatic proxy (needed to avoid dropping the variable entirely in this extract).
    # Coding: 1="Hispanic" in many GSS extracts; otherwise set missing if ambiguous.
    # If your extract uses a different coding, this will show up in diagnostics in output.
    if "ethnic" not in df.columns:
        raise ValueError("Missing ethnic column (needed here for Hispanic proxy).")
    ethnic = clean_gss_missing(df["ethnic"])
    df["hispanic"] = np.where(ethnic.isna(), np.nan, (ethnic == 1).astype(float))

    # Conservative Protestant proxy using RELIG and DENOM
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing relig and/or denom column.")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # South
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Diagnostics (to catch zero-variance / all-missing)
    # -------------------------
    x_order = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # -------------------------
    # Fit the two models
    # -------------------------
    results = {}

    mA, tabA, fullA, fitA, idxA = fit_unstd_and_betas(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, tabB, fullB, fitB, idxB = fit_unstd_and_betas(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # Pretty "Table 2 style" output: standardized betas + stars, and constant separately (unstandardized)
    def table2_style(tab, fitrow, label_map):
        out = tab.copy()
        out.index = [label_map.get(i, i) for i in out.index]
        out = out[["std_beta", "std_beta_star"]].copy()
        const_val = float(fitrow["constant_unstd"])
        const_p = float(fitrow["constant_p"])

        def stars(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        const_row = pd.DataFrame(
            {"std_beta": [np.nan], "std_beta_star": [f"{const_val:.3f}{stars(const_p)}"]},
            index=["Constant"],
        )
        return pd.concat([out, const_row], axis=0)

    labels = {
        "racism_score": "Racism score",
        "education_years": "Education",
        "hh_income_per_capita": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
    }

    paperA = table2_style(tabA, fitA.iloc[0].to_dict(), labels)
    paperB = table2_style(tabB, fitB.iloc[0].to_dict(), labels)

    # Save outputs
    write_text("./output/Table2_ModelA_summary.txt", mA.summary().as_text())
    write_text("./output/Table2_ModelB_summary.txt", mB.summary().as_text())

    write_text("./output/Table2_ModelA_table2style.txt", paperA.to_string(float_format=lambda x: f"{x:.6f}"))
    write_text("./output/Table2_ModelB_table2style.txt", paperB.to_string(float_format=lambda x: f"{x:.6f}"))

    write_text("./output/Table2_ModelA_full_unstandardized.txt", fullA.to_string(float_format=lambda x: f"{x:.6f}"))
    write_text("./output/Table2_ModelB_full_unstandardized.txt", fullB.to_string(float_format=lambda x: f"{x:.6f}"))

    write_text("./output/Table2_fit.txt", pd.concat([fitA, fitB], axis=0).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Additional diagnostics (variation and missingness)
    diag = []
    for name, idx in [("ModelA", idxA), ("ModelB", idxB)]:
        dsub = df.loc[idx, x_order].copy()
        row = {"model": name, "n": int(len(idx))}
        for c in x_order:
            s = dsub[c]
            row[f"{c}_mean"] = float(np.nanmean(s.values)) if s.notna().any() else np.nan
            row[f"{c}_sd"] = float(np.nanstd(s.values, ddof=0)) if s.notna().any() else np.nan
            row[f"{c}_missing"] = int(s.isna().sum())
        diag.append(row)
    diag_df = pd.DataFrame(diag)
    write_text("./output/Table2_diagnostics.txt", diag_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    results["ModelA_table2style"] = paperA
    results["ModelB_table2style"] = paperB
    results["ModelA_betas"] = tabA
    results["ModelB_betas"] = tabB
    results["ModelA_full"] = fullA
    results["ModelB_full"] = fullB
    results["fit"] = pd.concat([fitA, fitB], axis=0)
    results["diagnostics"] = diag_df

    return results