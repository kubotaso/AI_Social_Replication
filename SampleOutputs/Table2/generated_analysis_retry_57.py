def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing-code handling for this extract:
        - Treat explicit sentinel codes as missing.
        - Do NOT blanket-drop values like 0 for variables where 0 can be valid (e.g., education could be 0).
        """
        s = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        s = s.mask(s.isin(sentinels))
        return s

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5.
        Dislike = 1 if in {4,5}; 0 if in {1,2,3}; missing otherwise.
        """
        s = clean_gss_missing(x)
        s = s.where(s.between(1, 5))
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin([1, 2, 3])] = 0.0
        out.loc[s.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        s = clean_gss_missing(x)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin(false_codes)] = 0.0
        out.loc[s.isin(true_codes)] = 1.0
        return out

    def build_count_disliked(df_in, items, require_all_answered=True):
        mat = pd.concat([likert_dislike_indicator(df_in[c]).rename(c) for c in items], axis=1)
        if require_all_answered:
            # Paper: DK treated as missing; missing cases excluded -> complete-case on the DV components
            return mat.sum(axis=1, min_count=len(items))
        # If ever needed: allow partial; not used here.
        return mat.sum(axis=1, min_count=1)

    def stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def ols_with_posthoc_betas(dfin, dv, xcols, model_name):
        """
        Fit OLS on original scales (with intercept), listwise deletion on dv+xcols.
        Then compute standardized betas as: beta_j = b_j * SD(X_j)/SD(Y),
        using SDs from the analytic sample actually used in the regression.
        """
        needed = [dv] + xcols
        d = dfin[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}, k={len(xcols)}).")

        # Drop zero-variance predictors (do not error; keep model runnable)
        zero_var = []
        for c in xcols:
            v = d[c].to_numpy(dtype=float)
            if np.nanstd(v, ddof=0) == 0 or not np.isfinite(np.nanstd(v, ddof=0)):
                zero_var.append(c)
        x_use = [c for c in xcols if c not in zero_var]
        if len(x_use) == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance after listwise deletion.")

        y = d[dv].astype(float)
        X = sm.add_constant(d[x_use].astype(float), has_constant="add")
        model = sm.OLS(y, X).fit()

        # Post-hoc standardized betas (slopes only; constant is unstandardized)
        y_sd = float(np.nanstd(y.to_numpy(dtype=float), ddof=0))
        beta = {}
        for term in model.params.index:
            if term == "const":
                beta[term] = np.nan
                continue
            x_sd = float(np.nanstd(d[term].to_numpy(dtype=float), ddof=0))
            beta[term] = float(model.params[term] * (x_sd / y_sd)) if (y_sd > 0 and x_sd > 0) else np.nan

        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": pd.Series(beta),
            }
        )

        # Paper-style: standardized betas for predictors + unstandardized constant
        paper_rows = []
        for c in xcols:
            if c in x_use:
                b = full.loc[c, "beta_std"]
                p = full.loc[c, "p_value"]
                paper_rows.append((c, b, stars_from_p(p)))
            else:
                paper_rows.append((c, np.nan, ""))

        const_b = full.loc["const", "b_unstd"] if "const" in full.index else np.nan
        const_p = full.loc["const", "p_value"] if "const" in full.index else np.nan
        paper_rows.append(("const", const_b, stars_from_p(const_p)))

        paper = pd.DataFrame(paper_rows, columns=["term", "coef", "stars"]).set_index("term")
        paper["coef_star"] = paper["coef"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + paper["stars"]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
                }
            ]
        )
        return model, paper, full, fit, d.index

    # ----------------------------
    # Filter: 1993 only
    # ----------------------------
    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain 'year' and 'id' columns.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # Dependent variables
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_disliked(df, minority_items, require_all_answered=True)
    df["dislike_other12_genres"] = build_count_disliked(df, other12_items, require_all_answered=True)

    # ----------------------------
    # Racism score (0-5)
    # ----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])    # object to majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])    # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])   # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])   # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])   # endorse lack of motivation
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Controls
    # ----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' column.")
    df["education_years"] = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))

    # Income per capita: REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing 'realinc' and/or 'hompop' columns for income per capita.")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' column.")
    df["age_years"] = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))

    # Race dummies: RACE: 1 white, 2 black, 3 other
    if "race" not in df.columns:
        raise ValueError("Missing 'race' column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in this extract -> include as all-0 (not missing) so model can run
    # (This will not reproduce the paper without a true Hispanic identifier, but avoids runtime errors.)
    df["hispanic"] = 0.0

    # Conservative Protestant proxy: RELIG==1 (Protestant) and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing 'relig' and/or 'denom' columns.")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing 'region' column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Model spec (Table 2 RHS)
    # ----------------------------
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

    # ----------------------------
    # Fit models
    # ----------------------------
    mA, paperA, fullA, fitA, idxA = ols_with_posthoc_betas(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, idxB = ols_with_posthoc_betas(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # ----------------------------
    # Label terms for output (paper-like)
    # ----------------------------
    label_map = {
        "racism_score": "Racism score",
        "education_years": "Education (years)",
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
        "const": "Constant",
    }

    def relabel_index(d):
        out = d.copy()
        out.index = [label_map.get(i, i) for i in out.index]
        return out

    paperA_out = relabel_index(paperA)
    paperB_out = relabel_index(paperB)
    fullA_out = relabel_index(fullA)
    fullB_out = relabel_index(fullB)

    # ----------------------------
    # Save outputs
    # ----------------------------
    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(mA.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nStandardized coefficients (betas) + stars (computed from this model):\n")
        f.write(paperA_out[["coef_star"]].to_string())

    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(mB.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nStandardized coefficients (betas) + stars (computed from this model):\n")
        f.write(paperB_out[["coef_star"]].to_string())

    with open("./output/Table2_ModelA_tables.txt", "w", encoding="utf-8") as f:
        f.write("Paper-style (standardized betas for predictors; unstandardized constant):\n")
        f.write(paperA_out.to_string())
        f.write("\n\nFull re-estimation output (unstandardized b + SE + t + p + computed beta):\n")
        f.write(fullA_out.to_string())

    with open("./output/Table2_ModelB_tables.txt", "w", encoding="utf-8") as f:
        f.write("Paper-style (standardized betas for predictors; unstandardized constant):\n")
        f.write(paperB_out.to_string())
        f.write("\n\nFull re-estimation output (unstandardized b + SE + t + p + computed beta):\n")
        f.write(fullB_out.to_string())

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from provided extract; no numbers copied from paper).\n")
        f.write("Notes:\n")
        f.write("- Standardized betas computed post-hoc: beta = b * SD(X)/SD(Y) using the analytic sample.\n")
        f.write("- This extract has no Hispanic identifier; 'Hispanic' is set to 0 for all cases to keep the model runnable.\n")
        f.write("  This prevents exact replication of the paper's Table 2 where Hispanic varies.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA_out,
        "ModelB_paper_style": paperB_out,
        "ModelA_full": fullA_out,
        "ModelB_full": fullB_out,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }