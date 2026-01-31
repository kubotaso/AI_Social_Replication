def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load
    # -----------------------------
    df = pd.read_csv(data_source)

    # normalize column names to lower snake-like
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def dislike_indicator(x):
        """
        1 if response is 4 or 5 (dislike/dislike very much),
        0 if response is 1/2/3,
        missing otherwise.
        """
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich_item(x, ones, zeros):
        """
        Dichotomize to {0,1}; anything else -> missing.
        """
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def strict_sum(dfin, cols):
        # missing if ANY component missing
        return dfin[cols].sum(axis=1, skipna=False)

    def star(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def safe_dummy(series, true_value):
        s = to_num(series)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        m = s.notna()
        out.loc[m] = (s.loc[m] == true_value).astype(float)
        return out

    def standardized_betas(fit, y, X_no_const):
        """
        beta_j = b_j * SD(X_j) / SD(Y)
        computed on the analytic sample.
        """
        y = to_num(y)
        y_sd = float(y.std(ddof=0))
        betas = {}
        for col in X_no_const.columns:
            b = float(fit.params.get(col, np.nan))
            x_sd = float(to_num(X_no_const[col]).std(ddof=0))
            if np.isnan(b) or np.isnan(y_sd) or y_sd == 0 or np.isnan(x_sd) or x_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = b * (x_sd / y_sd)
        return pd.Series(betas)

    def format_table(df_table, float_cols=None):
        if float_cols is None:
            float_cols = df_table.select_dtypes(include=[np.number]).columns.tolist()

        def _fmt(x):
            if pd.isna(x):
                return ""
            try:
                return f"{float(x):0.3f}"
            except Exception:
                return str(x)

        out = df_table.copy()
        for c in float_cols:
            out[c] = out[c].apply(_fmt)
        return out

    # -----------------------------
    # Required columns (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_required = [
        "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race",
        "relig", "denom", "region"
    ]
    required = base_required + minority_genres + remaining_genres + racism_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # numeric coercion (id may remain non-numeric; but keep as-is)
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables (strict count)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_dislike_minority_linked_6"] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_dislike_remaining_12"] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5, strict)
    # -----------------------------
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])    # 2=no -> 1
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])    # 2=no -> 1
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])    # 1=yes -> 1

    df["racism_score"] = strict_sum(df, ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"])

    # -----------------------------
    # Controls / indicators
    # -----------------------------
    # income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # female
    df["female"] = safe_dummy(df["sex"], 2)

    # age, educ, prestg80 already numeric in df

    # race dummies: keep missing if race not in {1,2,3}
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic indicator is not available in provided fields -> set all to 0 when race is known; else missing.
    # This preserves sample size vs forcing missing for everyone.
    df["hispanic"] = np.where(race_known, 0.0, np.nan)

    # Conservative Protestant proxy: RELIG==1 (Protestant) & DENOM==1 (Baptist)
    rel_denom_known = df["relig"].notna() & df["denom"].notna()
    df["cons_prot"] = np.nan
    df.loc[rel_denom_known, "cons_prot"] = (
        (df.loc[rel_denom_known, "relig"] == 1) & (df.loc[rel_denom_known, "denom"] == 1)
    ).astype(float)

    # No religion
    df["no_religion"] = safe_dummy(df["relig"], 4)

    # Southern
    df["southern"] = safe_dummy(df["region"], 3)

    # Predictor order aligned to Table 2
    predictors = [
        "racism_score",
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    labels = {
        "racism_score": "Racism score (0-5)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic (not available -> set to 0 when race known)",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy: Protestant & Baptist)",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    # -----------------------------
    # Model fitting & output
    # -----------------------------
    def fit_and_save(dv_col, model_name, dv_label):
        cols = [dv_col] + predictors
        d = df[cols].copy()

        # listwise deletion on DV + predictors
        d = d.dropna().copy()
        y = d[dv_col].astype(float)
        X = d[predictors].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        # fit
        fit = sm.OLS(y, Xc).fit()

        # standardized betas
        betas = standardized_betas(fit, y, X).reindex(predictors)

        # stars from p-values of unstandardized coefficients (same hypothesis tests)
        pvals = fit.pvalues.reindex(["const"] + predictors)
        stars = pvals.apply(star)

        out = pd.DataFrame(
            {
                "Std_Beta": betas.values,
                "Sig": [stars.get(p, "") for p in predictors],
            },
            index=[labels.get(p, p) for p in predictors],
        )

        fit_stats = pd.DataFrame(
            {
                "N": [int(fit.nobs)],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Sig": [stars.get("const", "")],
            },
            index=[model_name],
        )

        # Save human-readable text
        lines = []
        lines.append(f"{model_name}")
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {dv_label}")
        lines.append("Model: OLS with standardized coefficients (beta weights) reported.")
        lines.append("Standardization: beta_j = b_j * SD(X_j) / SD(Y), computed on analytic sample.")
        lines.append("Stars: two-tailed p-values on unstandardized coefficients (* p<.05, ** p<.01, *** p<.001).")
        lines.append("Missing data: listwise deletion on DV + all predictors.")
        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("-------------------------")
        lines.append(format_table(out, float_cols=["Std_Beta"]).to_string())
        lines.append("")
        lines.append("Fit statistics")
        lines.append("--------------")
        lines.append(format_table(fit_stats, float_cols=["N", "R2", "Adj_R2", "Constant"]).to_string())
        lines.append("")
        lines.append("Notes")
        lines.append("-----")
        lines.append("- DV dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV construction is strict sum: DV missing if any component genre is missing.")
        lines.append("- Hispanic indicator is not present in this extract; set to 0 when race is observed to avoid dropping all cases.")
        lines.append("")

        with open(f"./output/{model_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Save full regression diagnostics
        with open(f"./output/{model_name}_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Save CSVs
        out.to_csv(f"./output/{model_name}_table.csv", index=True)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        return out, fit_stats, fit

    m1_table, m1_fit, m1_obj = fit_and_save(
        "dv1_dislike_minority_linked_6",
        "Table2_ModelA_DV1_MinorityLinked6",
        "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
    )
    m2_table, m2_fit, m2_obj = fit_and_save(
        "dv2_dislike_remaining_12",
        "Table2_ModelB_DV2_Remaining12",
        "Dislike of the 12 Remaining Genres (count of 12)",
    )

    combined = pd.concat(
        [
            m1_table.rename(columns={"Std_Beta": "ModelA_Std_Beta", "Sig": "ModelA_Sig"}),
            m2_table.rename(columns={"Std_Beta": "ModelB_Std_Beta", "Sig": "ModelB_Sig"}),
        ],
        axis=1,
    )

    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # Basic diagnostics about missingness drivers (before listwise)
    # (kept simple but useful for runtime debugging)
    key_cols_1 = ["dv1_dislike_minority_linked_6"] + predictors
    key_cols_2 = ["dv2_dislike_remaining_12"] + predictors
    miss_1 = df[key_cols_1].isna().mean().sort_values(ascending=False).to_frame("share_missing")
    miss_2 = df[key_cols_2].isna().mean().sort_values(ascending=False).to_frame("share_missing")

    dv_desc = df[["dv1_dislike_minority_linked_6", "dv2_dislike_remaining_12"]].describe()

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt using provided 1993 GSS extract\n")
        f.write("=======================================================================\n\n")
        f.write("Combined standardized coefficients (betas)\n")
        f.write("-----------------------------------------\n")
        f.write(format_table(combined, float_cols=["ModelA_Std_Beta", "ModelB_Std_Beta"]).to_string())
        f.write("\n\nFit statistics\n")
        f.write("--------------\n")
        f.write(format_table(combined_fit, float_cols=["N", "R2", "Adj_R2", "Constant"]).to_string())
        f.write("\n\nDV descriptives (constructed; prior to listwise deletion)\n")
        f.write("--------------------------------------------------------\n")
        f.write(format_table(dv_desc).to_string())
        f.write("\n\nMissingness shares (Model A variables)\n")
        f.write("-------------------------------------\n")
        f.write(format_table(miss_1).to_string())
        f.write("\n\nMissingness shares (Model B variables)\n")
        f.write("-------------------------------------\n")
        f.write(format_table(miss_2).to_string())
        f.write("\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=True)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=True)
    miss_1.to_csv("./output/missingness_modelA.csv", index=True)
    miss_2.to_csv("./output/missingness_modelB.csv", index=True)

    return {
        "table2_betas": combined,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_modelA": miss_1,
        "missingness_modelB": miss_2,
    }