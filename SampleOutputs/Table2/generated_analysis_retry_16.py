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

    def clean_gss_missing(x):
        """
        Conservative GSS missing-code handling for this extract:
        - Treat common sentinel codes as missing: 8/9, 98/99, 998/999, 9998/9999
        - Leave other numeric values untouched.
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        return x.mask(x.isin(sentinels))

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 scale. Dislike if 4/5; Like/Neutral if 1/2/3.
        Non-1..5 or NA-coded -> missing.
        """
        x = clean_gss_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def recode_binary(x, true_codes, false_codes):
        x = clean_gss_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_completecase(df, items):
        """
        Build DV as count of dislikes across items.
        To match the paper's "DK treated as missing and missing cases excluded",
        require all included items non-missing (complete-case for DV components).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

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

    def standardized_ols_table(df, dv, xcols, model_name, keep_intercept_unstd=True):
        """
        Fit OLS and report standardized betas.
        Approach:
          - Fit OLS on unstandardized y and X (with intercept).
          - Compute standardized beta for each non-intercept coefficient as:
                beta_j = b_j * sd(x_j) / sd(y)
            using estimation-sample SDs (ddof=0), which matches standard beta definition.
          - Intercept reported unstandardized (paper reports constants).
        """
        needed = [dv] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < len(xcols) + 3:
            raise ValueError(f"{model_name}: not enough complete cases: n={d.shape[0]}, k={len(xcols)}")

        y = to_num(d[dv]).astype(float)
        X = d[xcols].apply(to_num).astype(float)

        # Drop zero-variance predictors (but log it; in a correct setup this should not happen for key dummies)
        dropped = []
        for c in list(X.columns):
            sd = X[c].std(skipna=True, ddof=0)
            if not np.isfinite(sd) or sd == 0:
                dropped.append(c)
                X = X.drop(columns=[c])

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (non-intercept)
        y_sd = y.std(ddof=0)
        betas = {}
        for term in model.params.index:
            if term == "const":
                betas[term] = np.nan
            else:
                x_sd = X[term].std(ddof=0)
                betas[term] = model.params[term] * (x_sd / y_sd) if (np.isfinite(x_sd) and x_sd != 0 and np.isfinite(y_sd) and y_sd != 0) else np.nan

        out = pd.DataFrame(
            {
                "beta_std": pd.Series(betas),
                "p_value": model.pvalues,
            }
        )

        # Add stars based on replication p-values (not from paper)
        out["stars"] = out["p_value"].apply(stars_from_p)

        # Add unstandardized constant for readability
        if keep_intercept_unstd and "const" in model.params.index:
            out.loc["const", "b_unstd"] = model.params["const"]
        else:
            out["b_unstd"] = np.nan

        # Order rows: predictors in paper-like order + constant last
        order = [c for c in xcols if c in out.index]
        if "const" in out.index:
            order = order + ["const"]
        out = out.loc[order]

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "n": int(model.nobs),
                "k_predictors": int(model.df_model),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
            }]
        )

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Table intended to match Bryson Table 2 reports standardized betas only.\n")
            if dropped:
                f.write(f"Dropped zero-variance predictors: {dropped}\n")

        # "Paper-style" table: only beta + stars (+ constant b_unstd)
        paper_style = out[["beta_std", "stars"]].copy()
        paper_style.loc["const", "beta_std"] = out.loc["const", "b_unstd"] if "const" in out.index else np.nan
        paper_style = paper_style.rename(columns={"beta_std": "coef (std beta; const is unstd)"})
        paper_style.to_string(
            open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x: .3f}",
        )

        # Full replication table (extra columns, clearly labeled)
        full = out.copy()
        full.to_string(
            open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x: .6f}",
        )

        return model, out, fit, paper_style

    def missingness_audit(df, cols, tag):
        d = df[cols].copy()
        miss = d.isna().mean().sort_values(ascending=False)
        cnt = d.isna().sum().sort_values(ascending=False)
        res = pd.DataFrame({"missing_count": cnt, "missing_share": miss})
        res.to_string(open(f"./output/{tag}_missingness.txt", "w", encoding="utf-8"), float_format=lambda x: f"{x: .4f}")
        return res

    # -----------------------------
    # Load + year filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Required column YEAR not found.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct DVs
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------------
    # Racism score (0-5 additive, complete-case)
    # -----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = recode_binary(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half Black school
    rac2 = recode_binary(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = recode_binary(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = recode_binary(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = recode_binary(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC (educ).")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} needed for income per capita.")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 (prestg80).")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing SEX (sex).")
    df["female"] = recode_binary(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing AGE (age).")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing RACE (race).")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables; do NOT proxy using ETHNIC.
    # To keep model faithful while runnable, include but it will force listwise deletion if left NaN.
    # Therefore we OMIT it explicitly from estimation and document omission.
    # (This is the only defensible choice given the provided extract.)
    hispanic_available = False

    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing RELIG and/or DENOM needed for religion dummies.")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])

    # Conservative Protestant proxy per mapping instruction
    df["cons_protestant"] = np.where(
        (relig.isna() | denom.isna()),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    if "region" not in df.columns:
        raise ValueError("Missing REGION (region).")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Modeling: Table 2 specs (except Hispanic not available)
    # -----------------------------
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

    # Audit missingness to help diagnose N gaps
    missingness_audit(df, ["dislike_minority_genres"] + x_cols, "ModelA_inputs")
    missingness_audit(df, ["dislike_other12_genres"] + x_cols, "ModelB_inputs")

    modelA, tableA, fitA, paperA = standardized_ols_table(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    modelB, tableB, fitB, paperB = standardized_ols_table(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Replication of Bryson Table 2 structure using provided GSS 1993 extract.\n")
        f.write("Key note: Hispanic dummy is not included because no direct Hispanic identifier is present in the provided variables.\n")
        f.write("Standardized betas computed as b * sd(x)/sd(y) from OLS on original scales; constant reported unstandardized.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table_full": tableA,
        "ModelB_table_full": tableB,
        "ModelA_table_paper_style": paperA,
        "ModelB_table_paper_style": paperB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }