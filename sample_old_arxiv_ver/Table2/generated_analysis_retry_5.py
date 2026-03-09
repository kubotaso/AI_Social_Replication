def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + year filter
    # -----------------------------
    df = pd.read_csv(data_source)
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
        1 if 4/5, 0 if 1/2/3, else missing.
        """
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich_item(x, ones, zeros):
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def strict_sum(dfin, cols):
        # sum with missing if ANY component missing (strict/listwise construction)
        return dfin[cols].sum(axis=1, skipna=False)

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

    def standardized_betas(fit, y, X_no_const):
        """
        beta_std_j = b_j * sd(X_j) / sd(Y), computed on the analytic sample used to fit.
        """
        y = to_num(y)
        y_sd = float(y.std(ddof=0))
        betas = {}
        for term in fit.params.index:
            if term == "const":
                continue
            x = to_num(X_no_const[term])
            x_sd = float(x.std(ddof=0))
            b = float(fit.params[term])
            if y_sd == 0 or np.isnan(y_sd) or x_sd == 0 or np.isnan(x_sd):
                betas[term] = np.nan
            else:
                betas[term] = b * (x_sd / y_sd)
        return pd.Series(betas, name="beta_std")

    def drop_constant_or_empty_cols(X, desired_order):
        kept = []
        for c in desired_order:
            if c not in X.columns:
                continue
            s = X[c]
            n_nonmiss = int(s.notna().sum())
            if n_nonmiss == 0:
                continue
            if s.dropna().nunique() <= 1:
                continue
            kept.append(c)
        return X[kept].copy(), kept

    # -----------------------------
    # Columns / required vars
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_items
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Numeric coercion (id can stay as-is)
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables (strict counts)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_dislike_minority_6"] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_dislike_remaining_12"] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5, strict)
    # -----------------------------
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])
    df["racism_score"] = strict_sum(df, ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"])

    # -----------------------------
    # Controls / dummies
    # -----------------------------
    # income per capita
    df["income_pc"] = np.nan
    ok = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok, "income_pc"] = df.loc[ok, "realinc"] / df.loc[ok, "hompop"]

    # female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # race: white reference, include black and other (hispanic unavailable in this extract -> omitted)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # conservative protestant proxy: RELIG==1 and DENOM==1 (baptist)
    df["cons_prot"] = np.nan
    known = df["relig"].notna() & df["denom"].notna()
    df.loc[known, "cons_prot"] = ((df.loc[known, "relig"] == 1) & (df.loc[known, "denom"] == 1)).astype(float)

    # no religion (MUST include; previous version accidentally dropped it due to constant/empty issues)
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # southern
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictor order (Table 2 order, minus Hispanic which is not available here)
    predictors_order = [
        "racism_score",
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    pretty = {
        "racism_score": "Racism score",
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy: Protestant & Baptist)",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    def fit_table2(dv_col, model_name, dv_label):
        cols = [dv_col] + predictors_order
        d = df[cols].copy().dropna()  # listwise deletion per model

        y = d[dv_col].astype(float)
        X = d[predictors_order].astype(float)

        # Drop any constant/all-missing predictors after listwise deletion (prevents singular fits)
        X, kept = drop_constant_or_empty_cols(X, predictors_order)

        # If a key column got dropped unexpectedly, make that explicit
        dropped = [c for c in predictors_order if c not in kept]
        if dropped:
            with open(f"./output/{model_name}_dropped_predictors.txt", "w", encoding="utf-8") as f:
                f.write("Dropped predictors (all-missing or constant after listwise deletion):\n")
                f.write(", ".join(dropped) + "\n")

        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        beta = standardized_betas(fit, y, X)
        pvals = fit.pvalues.drop(labels=["const"], errors="ignore").reindex(beta.index)
        stars = pvals.apply(star_from_p)

        table = pd.DataFrame({"beta": beta, "star": stars}).reindex(kept)
        table.index = [pretty.get(i, i) for i in table.index]

        fit_stats = pd.DataFrame(
            {
                "N": [int(fit.nobs)],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant (unstd)": [float(fit.params.get("const", np.nan))],
            },
            index=[model_name],
        )

        # Save: paper-like output (betas + stars + fit stats + constant)
        out_txt = []
        out_txt.append(f"{model_name}: Standardized OLS coefficients (Table 2 style)")
        out_txt.append("=" * len(out_txt[-1]))
        out_txt.append("")
        out_txt.append(f"DV: {dv_label}")
        out_txt.append("Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing.")
        out_txt.append("DV construction: strict count (missing if any component genre item missing).")
        out_txt.append("Estimation: OLS; reported coefficients are standardized betas (post-estimation).")
        out_txt.append("Stars: two-tailed OLS p-values (*, **, *** for p<.05, .01, .001).")
        out_txt.append("Missing data: listwise deletion per model on DV + all included predictors.")
        out_txt.append("")
        out_txt.append(table.to_string(float_format=lambda v: f"{v:0.3f}"))
        out_txt.append("")
        out_txt.append("Fit statistics")
        out_txt.append("--------------")
        out_txt.append(fit_stats.to_string(float_format=lambda v: f"{v:0.3f}"))
        out_txt.append("")
        out_txt.append("Design columns used (in-order)")
        out_txt.append("-----------------------------")
        out_txt.append(", ".join(kept))

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(out_txt))

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=True)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Diagnostic summary (not the paper table)
        with open(f"./output/{model_name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n\nNOTE: The paper prints standardized betas (+stars) and fit stats; this file is diagnostics.\n")

        return table, fit_stats, fit, kept, dropped

    t1, fit1, m1, kept1, dropped1 = fit_table2(
        dv_col="dv1_dislike_minority_6",
        model_name="Model_1_Dislike_MinorityLinked_6",
        dv_label="Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
    )
    t2, fit2, m2, kept2, dropped2 = fit_table2(
        dv_col="dv2_dislike_remaining_12",
        model_name="Model_2_Dislike_Remaining_12",
        dv_label="Dislike of the 12 Remaining Genres (count of 12)",
    )

    combined_fit = pd.concat([fit1, fit2], axis=0)

    combined_betas = pd.concat(
        [
            t1.rename(columns={"beta": "beta_model_1", "star": "star_model_1"}),
            t2.rename(columns={"beta": "beta_model_2", "star": "star_model_2"}),
        ],
        axis=1,
    )

    # DV descriptives (constructed indices, before listwise deletion)
    dv_desc = df[["dv1_dislike_minority_6", "dv2_dislike_remaining_12"]].describe()

    # Missingness report (helps diagnose N shrinkage)
    miss_report = pd.DataFrame(
        {
            "missing_dv1": df["dv1_dislike_minority_6"].isna(),
            "missing_dv2": df["dv2_dislike_remaining_12"].isna(),
            "missing_any_predictor": df[predictors_order].isna().any(axis=1),
        }
    ).mean().rename("share_missing").to_frame()

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (1993 GSS; available fields)\n")
        f.write("======================================================================\n\n")
        f.write("IMPORTANT LIMITATION\n")
        f.write("--------------------\n")
        f.write("This extract does not include a dedicated Hispanic ethnicity indicator; Table 2 includes Hispanic.\n")
        f.write("Accordingly, models here omit Hispanic (cannot be faithfully replicated from provided fields).\n\n")

        f.write("Fit statistics\n")
        f.write("--------------\n")
        f.write(combined_fit.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nStandardized coefficients (Table-2-style)\n")
        f.write("----------------------------------------\n")
        f.write(combined_betas.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nDV descriptives (constructed counts; before listwise deletion)\n")
        f.write("-------------------------------------------------------------\n")
        f.write(dv_desc.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nMissingness shares (diagnostic)\n")
        f.write("------------------------------\n")
        f.write(miss_report.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nPredictors requested (in-order)\n")
        f.write("------------------------------\n")
        f.write(", ".join(predictors_order))
        f.write("\n\nPredictors kept (Model 1)\n")
        f.write("------------------------\n")
        f.write(", ".join(kept1) + "\n")
        if dropped1:
            f.write("Dropped (Model 1): " + ", ".join(dropped1) + "\n")
        f.write("\nPredictors kept (Model 2)\n")
        f.write("------------------------\n")
        f.write(", ".join(kept2) + "\n")
        if dropped2:
            f.write("Dropped (Model 2): " + ", ".join(dropped2) + "\n")

    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    combined_betas.to_csv("./output/combined_table2_style.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=True)
    miss_report.to_csv("./output/missingness_report.csv", index=True)

    return {
        "table2_style": combined_betas,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_report": miss_report,
        "model_1_predictors_kept": pd.DataFrame({"kept_predictors": kept1}),
        "model_2_predictors_kept": pd.DataFrame({"kept_predictors": kept2}),
    }