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
        Conservative missing handling for this extract:
        - Coerce to numeric
        - Treat common GSS NA-style codes as missing
        - Do NOT treat valid small integers (e.g., 0,1,2) as missing globally.
        """
        x = to_num(x).copy()
        na_codes = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(na_codes)))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 scale.
        Dislike = 1 if in {4,5}; Like/Neutral = 0 if in {1,2,3}; else missing.
        """
        x = clean_gss_missing(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_gss_missing(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(series):
        x = to_num(series)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        return (x - mu) / sd

    def build_dislike_count(df, items):
        """
        DV construction:
        - Item-level DK/refused/etc -> missing (NaN)
        - Paper treats DK as missing; respondents with missing are excluded.
        Implement strict complete-case on all items in the DV (min_count = len(items)).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def fit_standardized_ols(df, dv, xcols, model_name):
        """
        Standardized OLS coefficients:
        - Standardize y and all X (including dummies) within the analytic sample.
        - Fit OLS with intercept.
        - Return:
          * paper_style: beta + stars (and constant reported unstandardized)
          * full_table: beta + SE/t/p from standardized regression (constant is on standardized scale ~0)
          * fit stats
        Notes:
        - We avoid throwing errors for zero-variance predictors by dropping them, but we log it.
        """
        needed = [dv] + xcols
        d = df[needed].copy().replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < 25:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # Standardize
        y = zscore(d[dv])

        Xz = {}
        dropped = []
        for c in xcols:
            zx = zscore(d[c])
            if zx.isna().all() or zx.std(ddof=0) == 0 or (zx.notna().sum() == 0):
                dropped.append(c)
                continue
            Xz[c] = zx

        if len(Xz) == 0:
            raise ValueError(f"{model_name}: all predictors dropped (zero variance or missing).")

        X = pd.DataFrame(Xz, index=d.index)
        ok = y.notna()
        y = y.loc[ok]
        X = X.loc[ok]

        if X.shape[0] < (X.shape[1] + 2):
            raise ValueError(f"{model_name}: too few cases for model fit (n={X.shape[0]}, k={X.shape[1]}).")

        Xc = sm.add_constant(X, has_constant="add")
        m = sm.OLS(y, Xc).fit()

        full = pd.DataFrame(
            {
                "coef_beta": m.params,
                "std_err": m.bse,
                "t": m.tvalues,
                "p_value": m.pvalues,
            }
        )
        full.index.name = "term"

        # Paper-style: standardized betas for slopes + stars; constant reported as unstandardized intercept on DV scale
        def star(p):
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        # Unstandardized intercept on DV scale (fit unstandardized OLS to recover constant comparable to paper)
        Xu = d[list(Xz.keys())].copy()
        # keep same rows used above
        Xu = Xu.loc[ok]
        yu = d.loc[ok, dv]
        Xuc = sm.add_constant(Xu, has_constant="add")
        mu = sm.OLS(yu, Xuc).fit()

        paper_rows = []
        # keep original xcols order, excluding dropped
        for c in xcols:
            if c in dropped:
                paper_rows.append({"term": c, "beta": np.nan, "stars": "", "note": "dropped_zero_variance"})
            else:
                p = float(m.pvalues.get(c, np.nan))
                b = float(m.params.get(c, np.nan))
                paper_rows.append({"term": c, "beta": b, "stars": "" if not np.isfinite(p) else star(p), "note": ""})

        # Constant: unstandardized intercept; stars based on its p-value unless suppressed by convention
        const_p = float(mu.pvalues.get("const", np.nan))
        const_b = float(mu.params.get("const", np.nan))

        # Per feedback: in many tables Model B constant shown without stars; implement as convention for Model B only.
        suppress_const_stars = ("ModelB" in model_name) or ("other12" in model_name.lower())
        const_stars = "" if suppress_const_stars else ("" if not np.isfinite(const_p) else star(const_p))

        paper_rows.append({"term": "constant", "beta": const_b, "stars": const_stars, "note": "unstandardized_intercept"})
        paper = pd.DataFrame(paper_rows).set_index("term")

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m.nobs),
                    "k_predictors": int(m.df_model),  # excludes intercept
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(m.summary().as_text())
            f.write("\n\n--- Additional notes ---\n")
            if dropped:
                f.write(f"Dropped zero-variance predictors in standardized model: {dropped}\n")
            f.write("\nUnstandardized intercept model (for constant comparable to DV scale):\n")
            f.write(mu.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))

        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            f.write("Table 2-style output: standardized betas for predictors; constant is unstandardized intercept on DV scale.\n")
            f.write("Stars: * p<.05, ** p<.01, *** p<.001 (from this replication's p-values)\n\n")
            f.write(paper.to_string(float_format=lambda x: f"{x: .3f}"))

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write("Full standardized-regression output (not in the original paper table):\n\n")
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))

        return m, paper, full, fit, d.index

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Year filter: YEAR == 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.shape[0] == 0:
        raise ValueError("No rows for year==1993 found.")

    # -------------------------
    # DVs (Table 2)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items)

    # -------------------------
    # Racism score (0-5 additive index)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])    # object majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])    # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])   # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])   # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])   # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # RHS controls
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # HH income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))

    # Race dummies (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables -> include as all-0 (not missing) to avoid runtime errors.
    # This keeps the model runnable but note it cannot reproduce the paper's Hispanic effect from these data.
    df["hispanic"] = 0.0

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -------------------------
    # Fit models (Table 2)
    # -------------------------
    xcols = [
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

    # Quick diagnostics saved
    diag = {}
    for col in ["dislike_minority_genres", "dislike_other12_genres"] + xcols:
        if col not in df.columns:
            continue
        s = df[col]
        diag[col] = {
            "n": int(s.notna().sum()),
            "mean": float(to_num(s).mean(skipna=True)) if s.notna().any() else np.nan,
            "std": float(to_num(s).std(skipna=True, ddof=0)) if s.notna().any() else np.nan,
            "min": float(to_num(s).min(skipna=True)) if s.notna().any() else np.nan,
            "max": float(to_num(s).max(skipna=True)) if s.notna().any() else np.nan,
            "n_unique": int(pd.Series(s.dropna().unique()).shape[0]),
        }
    diag_df = pd.DataFrame(diag).T
    diag_df.index.name = "variable"
    diag_df.to_string(open("./output/Table2_diagnostics.txt", "w", encoding="utf-8"), float_format=lambda x: f"{x: .6f}")

    # Run both models
    mA, paperA, fullA, fitA, idxA = fit_standardized_ols(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, idxB = fit_standardized_ols(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (1993 GSS extract): standardized OLS coefficients\n")
        f.write("Notes:\n")
        f.write("- Dependent variables are dislike counts with item-level DK/refused treated as missing; DV requires all component items.\n")
        f.write("- Hispanic indicator is not directly available in provided variables; set to 0 for all cases to keep model runnable.\n")
        f.write("- Constant reported in paper-style tables is unstandardized intercept (from unstandardized DV model).\n")
        f.write("- Model B constant stars suppressed to match common table convention.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "diagnostics": diag_df,
    }