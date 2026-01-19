def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # --------------------------
    # Helpers
    # --------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative NA cleaning for GSS-style extracts.
        - Always coerce to numeric.
        - Treat obvious sentinel codes as missing.
        - Do NOT blanket-drop values like 0 unless invalid for that variable (handled per-variable).
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items are 1-5.
        Dislike = 4 or 5; Like/neutral = 1-3.
        Anything outside 1-5 or NA-coded => missing.
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

    def build_count_allow_partial(df, items, min_answered):
        """
        Bryson notes DK treated as missing; cases excluded.
        In practice, published N suggests the DV is not requiring all items answered.
        We'll compute the count over available items and require at least min_answered observed.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        answered = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, min_count=1)
        count = count.where(answered >= min_answered)
        return count

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardized_betas_from_unstd(result, y, X_no_const):
        """
        beta_j = b_j * sd(x_j)/sd(y) using estimation sample.
        Do not compute for intercept.
        """
        y_sd = np.nanstd(y, ddof=0)
        betas = {}
        for c in X_no_const.columns:
            x_sd = np.nanstd(X_no_const[c].values, ddof=0)
            b = result.params.get(c, np.nan)
            if not np.isfinite(y_sd) or y_sd == 0 or not np.isfinite(x_sd) or x_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (x_sd / y_sd)
        return pd.Series(betas)

    def fit_table2_model(df, dv, x_order, model_name, min_dv_answered):
        """
        - Listwise deletion on RHS + DV after DV construction.
        - Fit OLS with intercept.
        - Compute standardized betas for slopes (not intercept).
        - Produce a "paper-style" table with betas + stars, and a "full" table with unstd + SE/t/p.
        - Do NOT error on zero-variance predictors: keep them out (but report).
        """
        # DV
        d = df.copy()
        d[dv] = d[dv]  # ensure exists

        # Subset to needed columns
        needed = [dv] + x_order
        d = d[needed].replace([np.inf, -np.inf], np.nan)

        # Drop rows missing DV or any RHS
        d = d.dropna(axis=0, how="any")

        # If no data, return empty outputs
        if d.shape[0] < 5:
            paper = pd.DataFrame({"term": [], "beta": [], "stars": []})
            full = pd.DataFrame({"term": [], "b_unstd": [], "std_err": [], "t": [], "p_value": [], "beta": []})
            fit = pd.DataFrame([{"model": model_name, "n": int(d.shape[0]), "r2": np.nan, "adj_r2": np.nan, "dropped_zero_variance": ""}])
            return paper, full, fit

        y = d[dv].astype(float)

        # Build X in paper order
        X = d[x_order].astype(float)

        # Drop zero-variance predictors (cannot be estimated)
        zero_var = []
        keep_cols = []
        for c in X.columns:
            v = np.nanvar(X[c].values, ddof=0)
            if not np.isfinite(v) or v == 0:
                zero_var.append(c)
            else:
                keep_cols.append(c)
        Xk = X[keep_cols].copy()

        Xc = sm.add_constant(Xk, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        beta = standardized_betas_from_unstd(res, y.values, Xk)

        # Full table
        full = pd.DataFrame(
            {
                "term": res.params.index,
                "b_unstd": res.params.values,
                "std_err": res.bse.values,
                "t": res.tvalues.values,
                "p_value": res.pvalues.values,
            }
        )
        # Add standardized beta for slopes; intercept -> NaN
        full["beta"] = np.nan
        for c in beta.index:
            full.loc[full["term"] == c, "beta"] = float(beta.loc[c])

        # Paper-style table in exact Table 2 order + Constant at bottom
        # If some predictors dropped (zero variance), they will appear as NaN (but labeled).
        paper_rows = []
        for c in x_order:
            if c in beta.index:
                p = float(res.pvalues.get(c, np.nan))
                paper_rows.append({"term": c, "beta": float(beta.loc[c]), "stars": stars(p)})
            else:
                paper_rows.append({"term": c, "beta": np.nan, "stars": ""})

        # Constant (unstandardized), with stars based on p
        p_const = float(res.pvalues.get("const", np.nan))
        paper_rows.append({"term": "const", "beta": float(res.params.get("const", np.nan)), "stars": stars(p_const)})

        paper = pd.DataFrame(paper_rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(res.nobs),
                    "k_including_const": int(res.df_model + 1),
                    "r2": float(res.rsquared),
                    "adj_r2": float(res.rsquared_adj),
                    "dropped_zero_variance": ", ".join(zero_var),
                }
            ]
        )

        # Save readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
            f.write("\n\nDropped zero-variance predictors (not estimable): ")
            f.write(", ".join(zero_var) if zero_var else "None")
            f.write("\n")

        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            f.write("Paper-style output: standardized betas for slopes; unstandardized constant; stars from model p-values\n")
            f.write(paper.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

        return paper, full, fit

    # --------------------------
    # Load data / normalize cols
    # --------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: YEAR (year).")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    if df.empty:
        raise ValueError("No rows with YEAR==1993 found.")

    # --------------------------
    # Dependent variables
    # --------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music taste item column: {c}")

    # Use partial-availability rule to avoid cutting N in half.
    # Require at least 5/6 answered for minority DV, and at least 10/12 for other12 DV.
    df["dislike_minority_genres"] = build_count_allow_partial(df, minority_items, min_answered=5)
    df["dislike_other12_genres"] = build_count_allow_partial(df, other12_items, min_answered=10)

    # --------------------------
    # Racism scale (0-5)
    # --------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    # Allow 1 missing item (min 4) to reduce unnecessary attrition; scale remains additive.
    df["racism_score"] = racism_mat.sum(axis=1, min_count=4)

    # --------------------------
    # Predictors/controls
    # --------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column: educ")
    educ = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required income component: {c}")
    realinc = clean_na_codes(df["realinc"]).where(clean_na_codes(df["realinc"]) >= 0)
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column: prestg80")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female: SEX (1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column: age")
    age = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(18, 89))
    df["age_years"] = age

    # Race dummies: RACE 1=white, 2=black, 3=other
    if "race" not in df.columns:
        raise ValueError("Missing RACE column: race")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided extract -> implement as all-missing but DO NOT include in model.
    # (Including all-missing would force n=0 under listwise deletion.)
    df["hispanic"] = np.nan

    # Conservative Protestant proxy (given available fields):
    # RELIG==1 (Protestant) and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index, dtype="float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4 (none)
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index, dtype="float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column: region")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index, dtype="float64")
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # --------------------------
    # Fit two models (Table 2)
    # --------------------------
    # Table 2 includes Hispanic, but this extract does not provide it.
    # To keep code runnable and faithful to available data, we omit Hispanic from estimation
    # and clearly report omission in outputs.
    x_order = [
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

    paperA, fullA, fitA = fit_table2_model(
        df,
        dv="dislike_minority_genres",
        x_order=x_order,
        model_name="Table2_ModelA_dislike_minority6",
        min_dv_answered=5,
    )
    paperB, fullB, fitB = fit_table2_model(
        df,
        dv="dislike_other12_genres",
        x_order=x_order,
        model_name="Table2_ModelB_dislike_other12",
        min_dv_answered=10,
    )

    # --------------------------
    # Overview file
    # --------------------------
    overview_path = "./output/Table2_overview.txt"
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS extract)\n")
        f.write("Models: OLS with standardized betas computed as b * SD(x)/SD(y) on the estimation sample.\n")
        f.write("Important: This dataset extract does NOT include a Hispanic identifier; the Hispanic dummy from the paper cannot be estimated here.\n")
        f.write("DV construction: dislike counts from 1-5 genre items where 4/5=dislike; DK/other codes treated as missing.\n")
        f.write("To reduce unnecessary attrition, DV counts require partial completion (>=5/6 for minority DV; >=10/12 for other12 DV).\n\n")

        f.write("Model A (minority-associated 6 genres) fit:\n")
        f.write(fitA.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        f.write("\n\nModel A paper-style table:\n")
        f.write(paperA.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

        f.write("\n\n\nModel B (other 12 genres) fit:\n")
        f.write(fitB.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        f.write("\n\nModel B paper-style table:\n")
        f.write(paperB.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }