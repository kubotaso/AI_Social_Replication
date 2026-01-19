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

    def clean_na_codes(x):
        """
        Conservative NA cleaning for this extract:
        - Convert to numeric
        - Drop common GSS sentinel codes
        - Do NOT treat legitimate codes (e.g., 8 on 1-10 ideology) as missing unless specified
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste: 1-5 (like very much ... dislike very much)
        dislike = 1 if 4/5; 0 if 1/2/3; missing otherwise.
        """
        x = clean_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_completecase(df, items):
        """
        DV is the sum of item-level dislike indicators.
        To match "DK treated as missing and missing cases excluded", require ALL items observed.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def star_from_p(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardized_ols_table(df, dv, xcols, model_name):
        """
        Run OLS with standardized coefficients (beta) by z-scoring y and each x
        on the estimation sample (listwise deletion for dv + xcols).
        Keep an intercept. Return:
          - paper_style: term, beta, stars
          - fit: n, r2, adj_r2
          - full: params/bse/t/p for transparency (not "from paper")
        Never drop predictors silently; if a predictor has zero variance in-sample, mark "omitted".
        """
        use = df[[dv] + xcols].replace([np.inf, -np.inf], np.nan).copy()
        use = use.dropna(axis=0, how="any")
        n0 = use.shape[0]

        # Prepare standardized y
        y = zscore(use[dv])
        if y.isna().any():
            # if dv has no variance in-sample, stop
            raise ValueError(f"{model_name}: DV has zero variance or could not be standardized.")

        # Prepare standardized X, but do not crash on zero variance; mark omitted
        Xz = {}
        omitted = []
        for c in xcols:
            z = zscore(use[c])
            if z.isna().all():
                omitted.append(c)
            else:
                # if any NaNs appear after zscore (shouldn't, because listwise deletion) treat as omitted
                if z.isna().any():
                    omitted.append(c)
                else:
                    Xz[c] = z

        X = pd.DataFrame(Xz, index=use.index)
        Xc = sm.add_constant(X, has_constant="add")

        # Fit model
        model = sm.OLS(y, Xc).fit()

        # Full table (computed from data; not from paper)
        full = pd.DataFrame(
            {
                "coef_beta": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        full.index.name = "term"

        # Paper-style table: include all expected predictors in the requested order (+ constant)
        rows = []
        # constant first
        rows.append({"term": "const", "beta": float(full.loc["const", "coef_beta"]), "stars": star_from_p(float(full.loc["const", "p_value"]))})
        for c in xcols:
            if c in omitted or c not in full.index:
                rows.append({"term": c, "beta": np.nan, "stars": "omitted (collinear/zero-variance)"})
            else:
                p = float(full.loc[c, "p_value"])
                rows.append({"term": c, "beta": float(full.loc[c, "coef_beta"]), "stars": star_from_p(p)})

        paper_style = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "k_including_const": int(model.df_model + 1),
                    "n_listwise_input": int(n0),
                    "omitted_predictors": ", ".join(omitted) if omitted else "",
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: SEs/p-values are computed from this replication data; Table 2 in the paper reports standardized coefficients with stars only.\n")
            if omitted:
                f.write(f"\nOmitted predictors (collinear/zero-variance after listwise deletion): {', '.join(omitted)}\n")

        with open(f"./output/{model_name}_table_paperstyle.txt", "w", encoding="utf-8") as f:
            f.write("Standardized OLS coefficients (beta). Stars derived from replication p-values: * p<.05, ** p<.01, *** p<.001.\n")
            f.write("If a predictor is omitted, it is marked accordingly.\n\n")
            f.write(paper_style.to_string(index=False, na_rep=""))

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda v: f"{v: .6f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))

        return paper_style, fit, full

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # DVs
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------
    # Racism score (0-5 additive index)
    # -----------------------
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

    # -----------------------
    # RHS variables
    # -----------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: Black, Other race (White omitted). Hispanic not available in this extract -> include but omitted (all-missing).
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.notna(), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.notna(), (race == 3).astype(float), np.nan)

    # Hispanic: not present in provided variables; keep as missing so it is explicitly marked omitted.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(relig.notna() & denom.notna())
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float).where(relig.notna())
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float).where(region.notna())

    # -----------------------
    # Models (Table 2 RHS order)
    # -----------------------
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

    # Run both models
    tabA, fitA, fullA = standardized_ols_table(df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6")
    tabB, fitB, fullB = standardized_ols_table(df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12")

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): Standardized OLS coefficients (beta) computed by z-scoring y and x.\n")
        f.write("Stars are derived from replication p-values (* p<.05, ** p<.01, *** p<.001). The published table does not report SEs.\n")
        f.write("Important: This extract does not include a Hispanic identifier; 'hispanic' is therefore omitted (all missing) and will not be estimated.\n\n")
        f.write("Model A DV: count dislikes among {rap, reggae, blues, jazz, gospel, latin}\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(tabA.to_string(index=False, na_rep=""))
        f.write("\n\nModel B DV: count dislikes among the other 12 genres in the battery\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(tabB.to_string(index=False, na_rep=""))
        f.write("\n")

    return {
        "ModelA_table_paperstyle": tabA,
        "ModelB_table_paperstyle": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "ModelA_table_full": fullA,
        "ModelB_table_full": fullB,
    }