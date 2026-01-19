def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Helpers
    # -----------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing-value handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing: 8/9, 98/99, 998/999, 9998/9999
        Note: We do NOT blanket-drop large values (income can be large).
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5; dislike if 4 or 5.
        Missing if not in 1..5 after cleaning.
        """
        x = clean_gss_missing(x)
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

    def build_count_complete_case(df, items):
        """
        DV construction:
        - build 0/1 dislike indicator per item, missing if item missing
        - sum requiring ALL items observed (listwise within DV), matching
          "DK treated as missing and cases with missing excluded"
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardized_betas_from_unstd(model, y, X_no_const):
        """
        Standardized beta_j = b_j * sd(X_j) / sd(Y)
        Uses sample SD (ddof=1) over the estimation sample.
        """
        sd_y = np.std(y, ddof=1)
        betas = {}
        for c in X_no_const.columns:
            sd_x = np.std(X_no_const[c], ddof=1)
            b = model.params.get(c, np.nan)
            if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (sd_x / sd_y)
        return pd.Series(betas)

    def stars_from_p(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_model_unstd_y_report_std_beta(df, dv, xcols, model_name):
        """
        - Listwise delete on dv + xcols (ONLY)
        - Fit OLS with intercept on unstandardized DV
        - Compute standardized betas post-estimation for predictors (NOT const)
        """
        needed = [dv] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)
        X = d[xcols].astype(float)

        # Drop any zero-variance predictors (prevents NaN standardized betas and singular fits)
        variances = X.var(axis=0, ddof=1)
        keep_cols = [c for c in X.columns if np.isfinite(variances.get(c, np.nan)) and variances[c] > 0]
        dropped = [c for c in X.columns if c not in keep_cols]
        X = X[keep_cols]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (predictors only)
        beta_std = standardized_betas_from_unstd(model, y.values, X)

        # Build tables with names and stable ordering (paper order if present)
        rows = []

        # predictor rows
        for c in xcols:
            if c in X.columns:
                rows.append(
                    {
                        "term": c,
                        "beta_std": float(beta_std.get(c, np.nan)),
                        "b_unstd": float(model.params.get(c, np.nan)),
                        "std_err": float(model.bse.get(c, np.nan)),
                        "t": float(model.tvalues.get(c, np.nan)),
                        "p_value": float(model.pvalues.get(c, np.nan)),
                        "sig": stars_from_p(float(model.pvalues.get(c, np.nan))),
                    }
                )
            else:
                rows.append(
                    {
                        "term": c,
                        "beta_std": np.nan,
                        "b_unstd": np.nan,
                        "std_err": np.nan,
                        "t": np.nan,
                        "p_value": np.nan,
                        "sig": "",
                    }
                )

        # constant row (unstandardized only; beta_std not defined)
        rows.append(
            {
                "term": "const",
                "beta_std": np.nan,
                "b_unstd": float(model.params.get("const", np.nan)),
                "std_err": float(model.bse.get("const", np.nan)),
                "t": float(model.tvalues.get("const", np.nan)),
                "p_value": float(model.pvalues.get("const", np.nan)),
                "sig": stars_from_p(float(model.pvalues.get("const", np.nan))),
            }
        )

        tab_full = pd.DataFrame(rows).set_index("term")

        # "Paper-style" view: standardized betas + stars, plus constant b
        tab_paper = pd.DataFrame(index=tab_full.index)
        tab_paper["coef"] = tab_full["beta_std"]
        tab_paper.loc["const", "coef"] = tab_full.loc["const", "b_unstd"]
        tab_paper["sig"] = tab_full["sig"]
        tab_paper["coef_sig"] = tab_paper["coef"].map(lambda v: "" if not np.isfinite(v) else f"{v:.3f}") + tab_paper["sig"]
        tab_paper = tab_paper[["coef", "sig", "coef_sig"]]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv,
                    "n": int(model.nobs),
                    "k_predictors_requested": int(len(xcols)),
                    "k_predictors_used": int(X.shape[1]),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        return model, tab_paper, tab_full, fit

    # -----------------------
    # Load + normalize cols
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required base vars
    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Construct DVs
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # -----------------------
    # Racism score (0-5)
    # -----------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls
    # -----------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita = realinc / hompop
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
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: black, other_race, and "hispanic" proxy
    # NOTE: Provided extract lacks a direct Hispanic flag. To avoid omitting the variable entirely,
    # we construct a best-effort proxy from ETHNIC *only when it is clearly Hispanic-coded*.
    # If ETHNIC is not usable, hispanic will be all-missing and the model will fail earlier,
    # which is preferable to silently dropping it.
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic proxy:
    # Common in many GSS extracts: ETHNIC codes 1=Mexican, 2=Puerto Rican, 3=Other Spanish (or similar).
    # We use 1..3 as Hispanic; otherwise 0. Missing preserved.
    if "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[eth.notna()] = 0.0
        hisp.loc[eth.isin([1, 2, 3])] = 1.0
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan

    # Religion dummies
    if "relig" not in df.columns:
        raise ValueError("Missing column: relig")
    relig = clean_gss_missing(df["relig"])
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy using RELIG and DENOM where available
    # RELIG: 1 Protestant; DENOM broad groups: 1 Baptist; 6 Other; 7 No denom
    if "denom" in df.columns:
        denom = clean_gss_missing(df["denom"])
        consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
        consprot = pd.Series(consprot, index=df.index).astype(float)
        consprot.loc[relig.isna() | denom.isna()] = np.nan
        df["cons_protestant"] = consprot
    else:
        df["cons_protestant"] = np.nan

    # Southern
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Model specification (Table 2)
    # -----------------------
    x_cols = [
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

    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # -----------------------
    # Fit both models
    # -----------------------
    results = {}

    modelA, paperA, fullA, fitA = fit_model_unstd_y_report_std_beta(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    modelB, paperB, fullB, fitB = fit_model_unstd_y_report_std_beta(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )

    # -----------------------
    # Save outputs
    # -----------------------
    def write_table(df_table, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(df_table.to_string(float_format=lambda v: f"{v: .6f}" if np.isfinite(v) else " NaN"))
            f.write("\n")

    # Model summaries
    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(modelA.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n")
    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(modelB.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    # Tables
    write_table(paperA, "./output/Table2_ModelA_table_paper_style.txt")
    write_table(fullA, "./output/Table2_ModelA_table_full.txt")
    write_table(paperB, "./output/Table2_ModelB_table_paper_style.txt")
    write_table(fullB, "./output/Table2_ModelB_table_full.txt")

    # Fit overview
    fit_all = pd.concat([fitA, fitB], axis=0, ignore_index=True)
    with open("./output/Table2_fit_overview.txt", "w", encoding="utf-8") as f:
        f.write(fit_all.to_string(index=False))
        f.write("\n")

    # DV descriptives (helps diagnose constants / DV construction)
    dv_desc = df[["dislike_minority_genres", "dislike_other12_genres"]].describe()
    with open("./output/Table2_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write(dv_desc.to_string())
        f.write("\n")

    results["ModelA_table_paper_style"] = paperA
    results["ModelB_table_paper_style"] = paperB
    results["ModelA_table_full"] = fullA
    results["ModelB_table_full"] = fullB
    results["fit"] = fit_all

    return results