def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative NA cleaning for this extract:
        - treat common GSS sentinels as missing
        - treat negatives as missing
        """
        x = to_num(x).copy()
        x = x.mask(x < 0)
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1..5 valid. Dislike if 4/5. Like/neutral if 1/2/3.
        Non-1..5 and NA-coded treated missing.
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

    def build_count_dv(df, items):
        """
        Sum of dislike indicators. Require complete responses on all component items (listwise)
        to match typical "DK treated as missing and cases excluded" DV construction.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardize_series(s, ddof=1):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def compute_standardized_betas(model, X, y, ddof=1):
        """
        Standardized beta for each slope term:
            beta_j = b_j * sd(x_j) / sd(y)
        Intercept beta is not meaningful; set to NaN.
        SDs computed on estimation sample (already listwise deleted by model fitting).
        """
        params = model.params.copy()
        betas = pd.Series(index=params.index, dtype="float64")
        y_sd = pd.Series(y).std(ddof=ddof)
        for term in params.index:
            if term == "const":
                betas.loc[term] = np.nan
            else:
                x_sd = pd.Series(X[term]).std(ddof=ddof)
                if (not np.isfinite(x_sd)) or x_sd == 0 or (not np.isfinite(y_sd)) or y_sd == 0:
                    betas.loc[term] = np.nan
                else:
                    betas.loc[term] = params.loc[term] * (x_sd / y_sd)
        return betas

    def add_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def safe_variation_report(d, cols):
        rep = {}
        for c in cols:
            s = d[c]
            rep[c] = {
                "n_nonmissing": int(s.notna().sum()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "sd": float(s.std(ddof=1)) if s.notna().sum() >= 2 else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
                "n_unique": int(s.dropna().nunique()),
            }
        return pd.DataFrame(rep).T

    def fit_table2_model(df, dv, x_cols, model_name, ddof_sd=1):
        """
        - listwise deletion on DV + RHS columns
        - OLS
        - standardized betas computed post-hoc using estimation-sample SDs
        """
        needed = [dv] + x_cols
        d = df[needed].copy()

        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Check that predictors vary (do not drop silently; but also do not crash the run)
        zero_var = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                zero_var.append(c)

        # If any are constant, we will drop them for estimation but report clearly.
        x_use = [c for c in x_cols if c not in zero_var]

        y = d[dv].astype(float)
        X = d[x_use].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        betas = compute_standardized_betas(model, Xc, y, ddof=ddof_sd)

        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": betas,
            }
        )

        # Paper-style table (standardized betas + stars; constant uses unstandardized b)
        paper = pd.DataFrame(index=full.index)
        paper["coef"] = full["beta_std"]
        paper.loc["const", "coef"] = full.loc["const", "b_unstd"]
        paper["stars"] = full["p_value"].apply(add_stars)
        paper["coef_with_stars"] = paper["coef"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + paper["stars"]

        fit = {
            "model": model_name,
            "n": int(model.nobs),
            "k_params": int(len(model.params)),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "dropped_constant_predictors": ",".join(zero_var) if zero_var else "",
        }

        # Write files
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nDropped (zero-variance) predictors (if any): ")
            f.write(fit["dropped_constant_predictors"] if fit["dropped_constant_predictors"] else "None")
            f.write("\n")

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write(paper[["coef", "stars", "coef_with_stars"]].to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")

        return model, full, paper, pd.DataFrame([fit]), d.index, zero_var

    # -----------------------------
    # Load and filter (1993)
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Input must include columns: year, id")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs: dislike counts
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_count_dv(df, minority_items)
    df["dislike_other12_genres"] = build_count_dv(df, other12_items)

    # -----------------------------
    # Racism score (0-5) additive index
    # -----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    rac_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = rac_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # RHS variables
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC (educ)")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing REALINC (realinc) and/or HOMPOP (hompop)")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 (prestg80)")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX (sex)")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE (age)")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies
    if "race" not in df.columns:
        raise ValueError("Missing RACE (race)")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: attempt to derive from ETHNIC if present.
    # Note: The mapping instruction warned ETHNIC is not a direct Hispanic flag, but the paper includes Hispanic.
    # For this dataset, we implement a simple, explicit rule and document it:
    # - hispanic = 1 if ethnic codes commonly used for Hispanic/Latino in many GSS extracts (e.g., 20..29, or 30..39)
    # If ETHNIC not present, set missing (cannot estimate).
    if "ethnic" in df.columns:
        eth = clean_na_codes(df["ethnic"])
        # Conservative banding: treat 20-29 and 30-39 as Hispanic-origin codes if present.
        df["hispanic"] = np.where(eth.isna(), np.nan, eth.between(20, 39).astype(float))
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing RELIG (relig) and/or DENOM (denom)")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion (RELIG==4)
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # South (REGION==3)
    if "region" not in df.columns:
        raise ValueError("Missing REGION (region)")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -----------------------------
    # Model specs (Table 2)
    # -----------------------------
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

    # -----------------------------
    # Fit models
    # -----------------------------
    mA, fullA, paperA, fitA, idxA, droppedA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, fullB, paperB, fitB, idxB, droppedB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # -----------------------------
    # Build labeled "paper-like" output tables in the paper's order
    # -----------------------------
    paper_order = [
        ("racism_score", "Racism score"),
        ("education_years", "Education (years)"),
        ("hh_income_per_capita", "Household income per capita"),
        ("occ_prestige", "Occupational prestige"),
        ("female", "Female"),
        ("age_years", "Age"),
        ("black", "Black"),
        ("hispanic", "Hispanic"),
        ("other_race", "Other race"),
        ("cons_protestant", "Conservative Protestant"),
        ("no_religion", "No religion"),
        ("south", "Southern"),
        ("const", "Constant"),
    ]

    def format_paper_table(paper_df, model_name):
        tmp = paper_df.copy()
        tmp.index = tmp.index.astype(str)
        rows = []
        for term, label in paper_order:
            if term in tmp.index:
                coef = tmp.loc[term, "coef"]
                stars = tmp.loc[term, "stars"]
                if pd.isna(coef):
                    coef_str = ""
                else:
                    coef_str = f"{coef:.3f}"
                rows.append({"term": term, "label": label, "coef": coef, "stars": stars, "coef_with_stars": coef_str + stars})
            else:
                rows.append({"term": term, "label": label, "coef": np.nan, "stars": "", "coef_with_stars": ""})
        out = pd.DataFrame(rows).set_index("label")
        out_path = f"./output/{model_name}_table_paper_order.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out[["coef_with_stars"]].to_string())
            f.write("\n")
        return out

    paperA_ordered = format_paper_table(paperA, "Table2_ModelA_dislike_minority6")
    paperB_ordered = format_paper_table(paperB, "Table2_ModelB_dislike_other12")

    # Diagnostics: variation in analytic samples (helps explain zero-variance issues)
    diagA = safe_variation_report(df.loc[idxA], ["dislike_minority_genres"] + x_order)
    diagB = safe_variation_report(df.loc[idxB], ["dislike_other12_genres"] + x_order)
    diagA.to_csv("./output/Table2_ModelA_analytic_sample_variation.csv")
    diagB.to_csv("./output/Table2_ModelB_analytic_sample_variation.csv")

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from provided 1993 GSS extract)\n")
        f.write("OLS fit on listwise-complete cases per model. Standardized betas computed as b*sd(x)/sd(y) on estimation sample.\n\n")
        f.write("IMPORTANT NOTE ON HISPANIC:\n")
        f.write("- The paper includes a Hispanic dummy.\n")
        f.write("- This extract does not provide an explicit Hispanic ethnicity flag; we approximate using ETHNIC if present (codes 20-39).\n")
        f.write("- If ETHNIC coding differs from this assumption, coefficients/N will differ from the paper.\n\n")
        f.write("Model A: DV = count disliked among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\nDropped zero-variance predictors (if any): " + (",".join(droppedA) if droppedA else "None") + "\n\n")
        f.write("Model B: DV = count disliked among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\nDropped zero-variance predictors (if any): " + (",".join(droppedB) if droppedB else "None") + "\n")

    return {
        "ModelA_table_full": fullA,
        "ModelB_table_full": fullB,
        "ModelA_table_paper_order": paperA_ordered,
        "ModelB_table_paper_order": paperB_ordered,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "ModelA_dropped_zero_variance": pd.DataFrame({"dropped": droppedA}),
        "ModelB_dropped_zero_variance": pd.DataFrame({"dropped": droppedB}),
    }