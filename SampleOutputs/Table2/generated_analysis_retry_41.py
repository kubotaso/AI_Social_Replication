def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative NA cleaning for typical GSS-style codes.
        This extract is already mostly numeric; treat common sentinel codes as missing.
        """
        x = to_num(x).copy()
        na_codes = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(na_codes))
        return x

    def likert_dislike_indicator(s):
        """
        Music liking items: 1-5; dislike if 4 or 5; like/neutral if 1-3.
        Missing if not in 1..5 or NA-coded.
        """
        x = clean_gss_missing(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_gss_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items):
        inds = []
        for c in items:
            inds.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(inds, axis=1)
        # Paper: DK treated as missing and missing cases excluded -> require all items observed
        return mat.sum(axis=1, min_count=len(items))

    def zscore(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def standardized_betas_from_unstd(model, X, y):
        """
        Compute standardized betas from an unstandardized OLS fit:
        beta_j = b_j * sd(x_j) / sd(y), using the estimation sample.
        Intercept beta is left as NaN (not meaningful).
        """
        sd_y = y.std(ddof=0)
        betas = {}
        for term in model.params.index:
            if term == "const":
                betas[term] = np.nan
                continue
            sd_x = X[term].std(ddof=0)
            betas[term] = model.params[term] * (sd_x / sd_y) if (sd_x > 0 and sd_y > 0) else np.nan
        return pd.Series(betas)

    def star_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df, dv, x_terms_order, model_name):
        """
        Fit OLS with listwise deletion on DV + predictors.
        Compute standardized coefficients (betas) from unstandardized fit (SPSS-style).
        Output a paper-style table: beta + stars, plus a fit table.
        Do NOT drop predictors for zero variance; instead, keep them and let listwise deletion decide,
        but if a predictor is constant in the analytic sample, mark beta as NaN and still report.
        """
        needed = [dv] + x_terms_order
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        if d.shape[0] < len(x_terms_order) + 5:
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={d.shape[0]}).")

        y = d[dv].astype(float)
        X = d[x_terms_order].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(model, Xc.drop(columns=[], errors="ignore"), y)

        # Build output in the paper's order, plus constant at bottom
        rows = []
        for term in x_terms_order:
            p = model.pvalues.get(term, np.nan)
            b = betas.get(term, np.nan)
            rows.append(
                {
                    "term": term,
                    "beta": float(b) if pd.notna(b) else np.nan,
                    "star": star_from_p(p),
                    "p_value_replication": float(p) if pd.notna(p) else np.nan,
                }
            )

        # Constant (unstandardized only; table 2 prints constant but "standardized" label applies to slopes)
        rows.append(
            {
                "term": "const",
                "beta": np.nan,
                "star": star_from_p(model.pvalues.get("const", np.nan)),
                "p_value_replication": float(model.pvalues.get("const", np.nan)) if pd.notna(model.pvalues.get("const", np.nan)) else np.nan,
            }
        )

        paper = pd.DataFrame(rows).set_index("term")
        paper["beta_with_stars"] = paper["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + paper["star"]

        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": betas.reindex(model.params.index),
            }
        )

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv,
                    "n": int(model.nobs),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Standardized betas are computed from the unstandardized OLS slopes using SD(x)/SD(y) on the analytic sample.\n")
            f.write("- Stars are based on replication-model p-values (two-tailed): *<.05, **<.01, ***<.001.\n")
            f.write("- Table 2 in the paper reports betas and stars; SE/t/p are not printed there.\n")

        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            out = paper[["beta", "beta_with_stars", "p_value_replication"]].copy()
            f.write(out.to_string(float_format=lambda x: f"{x:.6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda x: f"{x:.6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))
            f.write("\n")

        return paper, full, fit

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
    # Dependent variables
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
    # Racism score (0-5 additive)
    # -------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
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
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    df["education_years"] = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))

    # Income per capita
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    df["age_years"] = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))

    # Race dummies from race (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: not available in the provided extract.
    # To keep the Table 2 RHS as close as possible while remaining computable, we create it as 0
    # (and document this). This avoids zero-variance errors caused by all-NaN.
    df["hispanic"] = 0.0

    # Conservative Protestant proxy
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column.")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion (RELIG==4)
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South (REGION==3)
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Fit both models
    # -------------------------
    # Paper's RHS order
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
    for c in x_order:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    paperA, fullA, fitA = fit_table2_model(df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6")
    paperB, fullB, fitB = fit_table2_model(df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12")

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS with standardized betas (computed from unstandardized slopes).\n")
        f.write("Important limitation: Hispanic indicator is not present in the provided extract; set to 0 for all cases.\n")
        f.write("If you supply a true Hispanic flag, replace the construction of df['hispanic'] accordingly.\n\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin) count\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres (12 remaining genres) count\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nModel A (paper-style):\n")
        f.write(paperA[["beta_with_stars"]].to_string())
        f.write("\n\nModel B (paper-style):\n")
        f.write(paperB[["beta_with_stars"]].to_string())
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }