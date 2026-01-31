def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)

    # -----------------------------
    # Basic checks + year filter
    # -----------------------------
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
        1 if 4/5 (dislike/dislike very much), 0 if 1/2/3, else missing.
        """
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich_item(x, ones, zeros):
        """
        Dichotomize to 0/1 with missing otherwise.
        """
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def strict_row_sum(dfin, cols):
        """
        Row-wise sum, but returns missing if any component missing.
        """
        s = dfin[cols].sum(axis=1, skipna=False)
        return s

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

    def standardized_betas_from_fit(fit, y, X_no_const):
        """
        beta_std_j = b_j * sd(X_j) / sd(Y), computed on the analytic sample.
        """
        y = to_num(y)
        y_sd = float(y.std(ddof=0))
        out = {}
        for term in fit.params.index:
            if term == "const":
                continue
            x = to_num(X_no_const[term])
            x_sd = float(x.std(ddof=0))
            b = float(fit.params[term])
            if y_sd == 0 or np.isnan(y_sd) or x_sd == 0 or np.isnan(x_sd):
                out[term] = np.nan
            else:
                out[term] = b * (x_sd / y_sd)
        return pd.Series(out, name="beta")

    def clean_design_matrix(X, required_cols_in_order):
        """
        Drop columns that are all-missing or constant after listwise deletion.
        (Fixes the '0.000000 se=0 p=NaN' singularity artifact.)
        Keeps remaining columns in required order.
        """
        keep = []
        for c in required_cols_in_order:
            if c not in X.columns:
                continue
            s = X[c]
            if s.notna().sum() == 0:
                continue
            if s.dropna().nunique() <= 1:
                continue
            keep.append(c)
        return X[keep].copy(), keep

    # -----------------------------
    # Variables per mapping instruction
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_cols = [
        "id", "hompop", "educ", "realinc", "prestg80", "sex", "age",
        "race", "relig", "denom", "region", "ethnic"
    ]
    needed = [c for c in (base_cols + minority_genres + remaining_genres + racism_items) if c in df.columns]
    missing_required = [c for c in (["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"] + minority_genres + remaining_genres + racism_items) if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing expected columns: {missing_required}")

    # numeric coercion
    for c in needed:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables (strict counts)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1"] = strict_row_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2"] = strict_row_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5, strict)
    # -----------------------------
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])     # object to school > half black
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])     # oppose busing
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])   # not mainly due to discrimination
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])   # not mainly due to lack of education opp.
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])   # mainly due to lack of motivation

    df["racism_score"] = strict_row_sum(df, ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"])

    # -----------------------------
    # Controls
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: use ETHNIC if present (best-effort, avoids all-missing)
    # If ETHNIC codes differ, this will still produce 0/1 for some codes and won't force NA.
    if "ethnic" in df.columns:
        hisp_codes = {16, 20, 21, 22, 23, 24}  # defensive set, common in some GSS recodes
        df["hispanic"] = np.where(df["ethnic"].notna(), df["ethnic"].isin(list(hisp_codes)).astype(float), np.nan)
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 (protestant) and DENOM==1 (baptist)
    df["cons_prot"] = np.nan
    denom_known = df["relig"].notna() & df["denom"].notna()
    df.loc[denom_known, "cons_prot"] = ((df.loc[denom_known, "relig"] == 1) & (df.loc[denom_known, "denom"] == 1)).astype(float)

    # No religion
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictor order to match Table 2
    predictors_order = [
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

    pretty = {
        "racism_score": "Racism score",
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic (ETHNIC proxy)",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy)",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    def fit_table2(dv_col, model_name, dv_label):
        cols_needed = [dv_col] + predictors_order
        d = df[cols_needed].copy().dropna()  # listwise per model on DV+predictors

        y = d[dv_col].astype(float)
        X = d[predictors_order].astype(float)

        # Remove all-missing/constant predictors (prevents singular matrix artifacts)
        X, kept = clean_design_matrix(X, predictors_order)

        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        beta = standardized_betas_from_fit(fit, y, X)
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

        # Save: paper-like table (only betas + stars + fit stats + constant)
        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}: Standardized OLS coefficients (Table 2 style)\n")
            f.write("=" * (len(model_name) + 44) + "\n\n")
            f.write(f"DV: {dv_label}\n")
            f.write("Dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.\n")
            f.write("DV construction: strict count (missing if any component genre item missing).\n")
            f.write("Estimation: OLS; reported coefficients are standardized betas; stars from two-tailed OLS p-values.\n")
            f.write("Missing data: listwise deletion per model on DV + all predictors.\n\n")
            f.write(table.to_string(float_format=lambda v: f"{v:0.3f}"))
            f.write("\n\nFit statistics\n--------------\n")
            f.write(fit_stats.to_string(float_format=lambda v: f"{v:0.3f}"))
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=True)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Save full OLS summary for debugging (not "paper table")
        with open(f"./output/{model_name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n\nNOTE: Bryson (1996) Table 2 prints standardized betas + significance only; this is diagnostics.\n")

        # Also write the kept design columns to make row-to-variable mapping explicit
        with open(f"./output/{model_name}_design_columns.txt", "w", encoding="utf-8") as f:
            f.write("Predictors requested (Table 2 order):\n")
            f.write(", ".join(predictors_order) + "\n\n")
            f.write("Predictors kept after dropping all-missing/constant columns:\n")
            f.write(", ".join(kept) + "\n")

        return table, fit_stats, fit, kept

    # Model labels matching the paper (Model 1 / Model 2)
    t1, fit1, m1, kept1 = fit_table2(
        dv_col="dv1",
        model_name="Model_1_Dislike_MinorityLinked_6",
        dv_label="Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
    )
    t2, fit2, m2, kept2 = fit_table2(
        dv_col="dv2",
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

    dv_desc = df[["dv1", "dv2"]].describe()

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (1993 GSS; available fields)\n")
        f.write("======================================================================\n\n")
        f.write("Fit statistics\n--------------\n")
        f.write(combined_fit.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nStandardized coefficients (Table-2-style)\n----------------------------------------\n")
        f.write(combined_betas.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nDV descriptives (constructed counts; before listwise deletion)\n-------------------------------------------------------------\n")
        f.write(dv_desc.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nNotes / limitations\n-------------------\n")
        f.write("- Hispanic is proxied from ETHNIC using a defensive set of common Hispanic/Spanish codes.\n")
        f.write("- Conservative Protestant is proxied as RELIG==1 and DENOM==1 (Baptist).\n")
        f.write("- If any predictor becomes constant after listwise deletion, it is dropped to avoid singular fits.\n")

    combined_fit.to_csv("./output/combined_fit.csv")
    combined_betas.to_csv("./output/combined_table2_style.csv")
    dv_desc.to_csv("./output/dv_descriptives.csv")

    return {
        "table2_style": combined_betas,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "model_1_predictors_kept": pd.DataFrame({"kept_predictors": kept1}),
        "model_2_predictors_kept": pd.DataFrame({"kept_predictors": kept2}),
    }