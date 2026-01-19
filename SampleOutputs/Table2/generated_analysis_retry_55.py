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

    def clean_na_codes(series):
        """
        Conservative cleaning for common GSS-style NA codes in numeric extracts.
        We only blank out very common sentinel codes; do NOT blank out valid mid-range values.
        """
        x = to_num(series).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(sentinel)))
        return x

    def likert_dislike_indicator(item_series):
        """
        Music items: 1-5, where 4/5 mean dislike. DK/NA -> missing.
        Returns float {0,1} with NaN for missing.
        """
        x = clean_na_codes(item_series)
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

    def build_count_completecase(df, items):
        """
        Sum of binary dislike indicators; require ALL items observed for the DV (complete-case),
        consistent with 'DK treated as missing and cases excluded' for DV construction.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

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

    def ols_with_standardized_betas(df_in, dv, xcols, model_name):
        """
        Fit OLS on unstandardized variables with intercept.
        Compute standardized betas post-hoc:
            beta_j = b_j * SD(X_j) / SD(Y)
        using SDs from the FINAL analytic sample for this model.
        Return:
            model (statsmodels result),
            paper_table (betas + stars),
            full_table (b, se, t, p, beta),
            fit_df,
            analytic_df (for diagnostics)
        """
        needed = [dv] + xcols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # Ensure predictors vary (do not raise for missing hispanic in extract; just drop if constant)
        zero_var = [c for c in xcols if d[c].nunique(dropna=True) <= 1]
        xcols_use = [c for c in xcols if c not in zero_var]

        if len(xcols_use) == 0:
            raise ValueError(f"{model_name}: all predictors are constant in analytic sample.")

        y = d[dv].astype(float)
        X = d[xcols_use].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (slopes only)
        sd_y = y.std(ddof=0)
        betas = {}
        for c in xcols_use:
            sd_x = d[c].astype(float).std(ddof=0)
            if sd_y == 0 or sd_x == 0 or (not np.isfinite(sd_y)) or (not np.isfinite(sd_x)):
                betas[c] = np.nan
            else:
                betas[c] = model.params[c] * (sd_x / sd_y)

        # Tables
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        beta_series = pd.Series(betas, name="beta_std")
        full = full.join(beta_series, how="left")
        full.index.name = "term"

        # "Paper style": standardized betas for predictors + unstandardized constant (clearly labeled)
        paper_rows = []
        for c in xcols:
            if c in xcols_use:
                p = float(model.pvalues.get(c, np.nan))
                b = float(betas.get(c, np.nan))
                paper_rows.append((c, b, add_stars(p)))
            else:
                # predictor dropped due to zero variance in analytic sample
                paper_rows.append((c, np.nan, ""))

        paper = pd.DataFrame(paper_rows, columns=["term", "beta_std", "stars"]).set_index("term")
        paper["beta_with_stars"] = paper["beta_std"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + paper["stars"]

        # Add constant as unstandardized (not beta)
        const_p = float(model.pvalues.get("const", np.nan))
        const_b = float(model.params.get("const", np.nan))
        const_row = pd.DataFrame(
            {
                "beta_std": [np.nan],
                "stars": [add_stars(const_p)],
                "beta_with_stars": [("" if pd.isna(const_b) else f"{const_b:.3f}") + add_stars(const_p)],
            },
            index=["const_unstd"],
        )
        paper = pd.concat([paper, const_row], axis=0)

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

        return model, paper, full, fit, d

    # -----------------------------
    # Load & filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["year", "id"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Dependent variables
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
    # Racism score (0-5)
    # -----------------------------
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

    # -----------------------------
    # Controls
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing educ column (EDUC).")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column (PRESTG80).")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column (SEX).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column (AGE).")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (white ref)
    if "race" not in df.columns:
        raise ValueError("Missing race column (RACE).")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided extract -> create all-missing; do NOT error
    df["hispanic"] = np.nan

    # Religion and denomination
    if "relig" not in df.columns:
        raise ValueError("Missing relig column (RELIG).")
    relig = clean_na_codes(df["relig"])
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "denom" not in df.columns:
        raise ValueError("Missing denom column (DENOM).")
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # South
    if "region" not in df.columns:
        raise ValueError("Missing region column (REGION).")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Models (Table 2 RHS order)
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

    # Run both models
    mA, paperA, fullA, fitA, dA = ols_with_standardized_betas(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, dB = ols_with_standardized_betas(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # -----------------------------
    # Diagnostics for the specific reported issue: no_religion variation
    # -----------------------------
    def freq_table(series):
        vc = series.value_counts(dropna=False)
        vc.index = vc.index.map(lambda v: "NaN" if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v))
        return vc.to_frame("count")

    no_rel_A = freq_table(dA["no_religion"])
    no_rel_B = freq_table(dB["no_religion"])

    # -----------------------------
    # Save outputs
    # -----------------------------
    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    # Statsmodels summaries (full)
    write_text("./output/Table2_ModelA_summary.txt", mA.summary().as_text())
    write_text("./output/Table2_ModelB_summary.txt", mB.summary().as_text())

    # Full tables (unstandardized + beta)
    write_text("./output/Table2_ModelA_full_table.txt", fullA.to_string(float_format=lambda x: f"{x: .6f}"))
    write_text("./output/Table2_ModelB_full_table.txt", fullB.to_string(float_format=lambda x: f"{x: .6f}"))

    # Paper-style (standardized betas + stars + unstd constant)
    write_text("./output/Table2_ModelA_paper_style.txt", paperA[["beta_std", "stars", "beta_with_stars"]].to_string())
    write_text("./output/Table2_ModelB_paper_style.txt", paperB[["beta_std", "stars", "beta_with_stars"]].to_string())

    # Fit tables
    write_text("./output/Table2_ModelA_fit.txt", fitA.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    write_text("./output/Table2_ModelB_fit.txt", fitB.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # no_religion frequency diagnostics on analytic samples
    write_text("./output/Table2_no_religion_freq_ModelA.txt", no_rel_A.to_string())
    write_text("./output/Table2_no_religion_freq_ModelB.txt", no_rel_B.to_string())

    # Overview
    overview = []
    overview.append("Table 2 replication attempt (computed from provided data extract; 1993 only).")
    overview.append("Standardized betas computed post-hoc: beta = b * SD(X)/SD(Y) on the model's analytic sample.")
    overview.append("Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed), based on the fitted model p-values.")
    overview.append("")
    overview.append("NOTE: If 'hispanic' is not in the extract, it will be all-missing and will be dropped by listwise deletion.")
    overview.append("      This will reduce N relative to the paper and can change coefficients.")
    overview.append("")
    overview.append("Model A fit:")
    overview.append(fitA.to_string(index=False))
    overview.append("")
    overview.append("Model B fit:")
    overview.append(fitB.to_string(index=False))
    overview.append("")
    overview.append("No religion frequency (analytic samples):")
    overview.append("Model A:")
    overview.append(no_rel_A.to_string())
    overview.append("")
    overview.append("Model B:")
    overview.append(no_rel_B.to_string())
    write_text("./output/Table2_overview.txt", "\n".join(overview))

    # Return results as dict of DataFrames
    return {
        "ModelA_paper_style": paperA[["beta_std", "stars", "beta_with_stars"]],
        "ModelB_paper_style": paperB[["beta_std", "stars", "beta_with_stars"]],
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "no_religion_freq_ModelA": no_rel_A,
        "no_religion_freq_ModelB": no_rel_B,
    }