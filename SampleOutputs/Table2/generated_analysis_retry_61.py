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
        Conservative NA handling for this extract:
        - Coerce to numeric
        - Set common GSS sentinel codes to NaN
        - Leave other values untouched
        """
        x = to_num(x).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(sentinel)))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5. Define dislike as 4 or 5.
        Return: 1 if dislike, 0 if not (1-3), NaN otherwise.
        """
        x = clean_na_codes(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_na_codes(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete(df, items):
        """
        Paper notes DK treated as missing and missing cases excluded.
        Implement DV as sum of item-level dislike indicators, requiring all items non-missing.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def beta_posthoc_from_unstd(b, X, y):
        """
        Compute standardized betas from unstandardized regression coefficients:
        beta_j = b_j * sd(x_j) / sd(y)
        Intercept excluded (NaN).
        """
        sd_y = y.std(ddof=0)
        betas = pd.Series(index=b.index, dtype="float64")
        betas.loc[:] = np.nan
        for term in b.index:
            if term == "const":
                continue
            if term not in X.columns:
                continue
            sd_x = X[term].std(ddof=0)
            if sd_y and np.isfinite(sd_y) and sd_y > 0 and sd_x and np.isfinite(sd_x) and sd_x > 0:
                betas.loc[term] = b.loc[term] * (sd_x / sd_y)
        return betas

    def star(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df, dv, xcols, model_name):
        """
        OLS with listwise deletion on dv + xcols.
        Report standardized coefficients (betas) via posthoc conversion from unstandardized OLS.
        If a predictor is constant in the analytic sample, drop it (do not error).
        """
        needed = [dv] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < 20:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        y = to_num(d[dv])
        X = d[xcols].apply(to_num)

        # Drop any constant/zero-variance predictors in this analytic sample
        keep = []
        dropped = []
        for c in X.columns:
            vc = X[c]
            if vc.nunique(dropna=True) <= 1:
                dropped.append(c)
                continue
            if vc.std(ddof=0) == 0 or not np.isfinite(vc.std(ddof=0)):
                dropped.append(c)
                continue
            keep.append(c)
        X = X[keep]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Posthoc standardized betas for slopes
        betas = beta_posthoc_from_unstd(model.params, X, y)

        full = pd.DataFrame(
            {
                "b": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta": betas,
            }
        )

        # Paper-style: betas + stars (intercept kept as unstandardized b)
        paper_rows = []
        # Preserve original Table 2 order as much as possible; include only those present
        order = xcols[:]  # desired order
        for c in order:
            if c in X.columns:
                paper_rows.append(c)
        paper = pd.DataFrame(index=paper_rows + ["const"], columns=["coef", "stars"], dtype="object")
        for term in paper.index:
            if term == "const":
                coef = model.params.get("const", np.nan)
                pval = model.pvalues.get("const", np.nan)
                paper.loc[term, "coef"] = coef
                paper.loc[term, "stars"] = star(pval) if np.isfinite(pval) else ""
            else:
                coef = betas.get(term, np.nan)
                pval = model.pvalues.get(term, np.nan)
                paper.loc[term, "coef"] = coef
                paper.loc[term, "stars"] = star(pval) if np.isfinite(pval) else ""

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_constant_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nDropped constant predictors (in analytic sample): ")
            f.write(", ".join(dropped) if dropped else "None")
            f.write("\n")

        # Human-readable tables
        def write_table(path, df_out, floatfmt="{: .6f}"):
            with open(path, "w", encoding="utf-8") as f:
                f.write(df_out.to_string(index=True, float_format=lambda x: floatfmt.format(x) if np.isfinite(x) else "nan"))
                f.write("\n")

        write_table(f"./output/{model_name}_full_table.txt", full)
        # Paper style: show coef rounded like typical tables
        paper_print = paper.copy()
        paper_print["coef"] = pd.to_numeric(paper_print["coef"], errors="coerce")
        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            f.write("Standardized OLS coefficients (beta) for slopes; intercept shown as unstandardized constant.\n")
            f.write("Significance: * p<.05, ** p<.01, *** p<.001\n\n")
            f.write(paper_print.to_string(index=True, float_format=lambda x: f"{x: .3f}" if np.isfinite(x) else "nan"))
            f.write("\n")

        return model, paper_print, full, fit, d.index

    # -----------------------
    # Load and preprocess
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must include YEAR and ID columns (case-insensitive).")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Dependent variables
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete(df, other12_items)

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
    # RHS controls
    # -----------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required income column: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race indicators (White reference; include Black, Other; Hispanic not available here)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator:
    # Not available in provided variables; keep as all-NaN so it does not force listwise deletion.
    # (Do NOT proxy using ETHNIC per mapping instruction.)
    df["hispanic"] = np.nan

    # Religion
    if "relig" not in df.columns:
        raise ValueError("Missing RELIG column (relig).")
    relig = clean_na_codes(df["relig"])

    # No religion dummy: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "denom" not in df.columns:
        raise ValueError("Missing DENOM column (denom).")
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = np.where(relig.isna() | denom.isna(), np.nan, consprot)
    df["cons_protestant"] = consprot.astype(float)

    # Southern
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Model specs (Table 2 RHS)
    # -----------------------
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

    # -----------------------
    # Fit both models
    # -----------------------
    mA, paperA, fullA, fitA, idxA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, idxB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS with standardized coefficients (beta) for slopes.\n")
        f.write("Note: In this provided extract, no direct Hispanic identifier is available; 'hispanic' will be dropped.\n")
        f.write("DVs are complete-case counts across their respective genre sets (DK treated as missing).\n\n")
        f.write("MODEL A: DV = count disliked among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nMODEL B: DV = count disliked among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }