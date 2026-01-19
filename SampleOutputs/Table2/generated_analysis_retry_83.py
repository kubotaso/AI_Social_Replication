def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats

    os.makedirs("./output", exist_ok=True)

    # --------------------------
    # Helpers
    # --------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing-value handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        NOTE: We only mask a small set of common sentinels to avoid removing valid values.
        """
        x = to_num(x).astype("float64")
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5.
        Dislike = 4 or 5; Like/Neutral = 1/2/3.
        Non-1..5 or sentinel -> missing.
        """
        x = clean_gss_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_recode(x, true_codes, false_codes):
        x = clean_gss_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_series(s):
        s = to_num(s).astype("float64")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_dislike_count(df, items, require_complete=True):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_complete:
            # Paper: DK treated as missing; implement complete-case for the DV construction
            return mat.sum(axis=1, min_count=len(items))
        # Alternative (not used): allow partial information
        return mat.sum(axis=1, min_count=1)

    def compute_standardized_betas(model, y, X_no_const):
        """
        Compute standardized betas for an OLS model fit on unstandardized data.
        beta_j = b_j * sd(x_j)/sd(y)
        """
        y_sd = y.std(ddof=0)
        betas = {}
        for c in X_no_const.columns:
            x_sd = X_no_const[c].std(ddof=0)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = model.params[c] * (x_sd / y_sd)
        return pd.Series(betas)

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

    def fit_one_model(df, dv_col, rhs_cols_ordered, pretty_names, model_name):
        # Listwise deletion on exactly DV + RHS
        d = df[[dv_col] + rhs_cols_ordered].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] <= (len(rhs_cols_ordered) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(rhs_cols_ordered)}).")

        y = d[dv_col].astype("float64")
        X = d[rhs_cols_ordered].astype("float64")

        # Drop zero-variance predictors (but keep order for reporting with NaN)
        variances = X.var(axis=0, ddof=0)
        keep_cols = [c for c in rhs_cols_ordered if np.isfinite(variances.get(c, np.nan)) and variances[c] > 0]
        dropped = [c for c in rhs_cols_ordered if c not in keep_cols]

        Xk = X[keep_cols]
        Xk_const = sm.add_constant(Xk, has_constant="add")
        model = sm.OLS(y, Xk_const).fit()

        # Standardized betas (computed, not taken from paper)
        betas = compute_standardized_betas(model, y, Xk)

        # Create full beta series including dropped predictors as NaN
        betas_full = pd.Series(index=rhs_cols_ordered, dtype="float64")
        betas_full.loc[keep_cols] = betas.loc[keep_cols].values
        betas_full.loc[dropped] = np.nan

        # p-values for standardized betas are the same as for unstandardized slopes (scaling does not change t)
        pvals_full = pd.Series(index=rhs_cols_ordered, dtype="float64")
        pvals_full.loc[keep_cols] = model.pvalues.loc[keep_cols].values
        pvals_full.loc[dropped] = np.nan

        # Build results table in the paper's row order + Constant
        rows = []
        for c in rhs_cols_ordered:
            beta = betas_full.loc[c]
            p = pvals_full.loc[c]
            rows.append(
                {
                    "Variable": pretty_names.get(c, c),
                    "Beta": beta,
                    "Sig": stars_from_p(p),
                }
            )
        # Constant (unstandardized)
        const_p = model.pvalues.get("const", np.nan)
        rows.append(
            {
                "Variable": "Constant",
                "Beta": model.params.get("const", np.nan),
                "Sig": stars_from_p(const_p),
            }
        )
        table = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join([pretty_names.get(c, c) for c in dropped]) if dropped else "",
                }
            ]
        )

        # Save text outputs
        # 1) statsmodels summary
        with open(f"./output/{model_name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Table reports standardized coefficients (betas) computed as b * sd(x)/sd(y).\n")
            if dropped:
                f.write(f"Dropped zero-variance predictors: {', '.join(dropped)}\n")

        # 2) human-readable "Table 2 style" (beta + stars; no SE column)
        out = table.copy()
        # Format Beta column: standardized betas to 3 decimals; constant to 3 decimals as well
        def fmt_beta(var, val):
            if pd.isna(val):
                return ""
            return f"{val:.3f}"

        out["Beta"] = [fmt_beta(v, b) for v, b in zip(out["Variable"], out["Beta"])]
        out["Reported"] = out["Beta"] + out["Sig"]
        out = out[["Variable", "Reported"]]

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv_col}\n")
            f.write(f"N: {int(model.nobs)}   R^2: {model.rsquared:.3f}   Adj R^2: {model.rsquared_adj:.3f}\n")
            if dropped:
                f.write(f"Dropped (zero variance): {', '.join([pretty_names.get(c, c) for c in dropped])}\n")
            f.write("\n")
            f.write(out.to_string(index=False))

        return model, table, fit, d.index

    # --------------------------
    # Load
    # --------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # --------------------------
    # Construct DVs
    # --------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority6"] = build_dislike_count(df, minority_items, require_complete=True)
    df["dislike_other12"] = build_dislike_count(df, other12_items, require_complete=True)

    # --------------------------
    # Construct racism score (0-5)
    # --------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_recode(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_recode(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_recode(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_recode(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_recode(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # --------------------------
    # Controls
    # --------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_recode(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: black, hispanic (not available), other race
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not present in provided variables. Do not proxy using ETHNIC.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing column(s): relig and/or denom")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna(), np.nan)
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # --------------------------
    # Modeling: Table 2 order
    # --------------------------
    rhs_order = [
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
    pretty = {
        "racism_score": "Racism score",
        "education_years": "Education",
        "hh_income_per_capita": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
    }

    # Save a quick missingness + distribution report to debug N issues
    report_cols = ["dislike_minority6", "dislike_other12"] + rhs_order
    miss = []
    for c in report_cols:
        s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        s = to_num(s)
        miss.append(
            {
                "variable": c,
                "missing_n": int(s.isna().sum()),
                "nonmissing_n": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "std": float(s.std(skipna=True, ddof=0)) if s.notna().any() else np.nan,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            }
        )
    miss_df = pd.DataFrame(miss)
    miss_df.to_csv("./output/missingness_and_descriptives_1993.csv", index=False)
    with open("./output/missingness_and_descriptives_1993.txt", "w", encoding="utf-8") as f:
        f.write(miss_df.to_string(index=False))

    # Fit models
    modelA, tableA, fitA, idxA = fit_one_model(
        df=df,
        dv_col="dislike_minority6",
        rhs_cols_ordered=rhs_order,
        pretty_names=pretty,
        model_name="Table2_ModelA_Dislike_Minority_Associated6",
    )
    modelB, tableB, fitB, idxB = fit_one_model(
        df=df,
        dv_col="dislike_other12",
        rhs_cols_ordered=rhs_order,
        pretty_names=pretty,
        model_name="Table2_ModelB_Dislike_Other12",
    )

    # Overview text
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)\n")
        f.write("OLS on raw DV counts; standardized coefficients computed as b*sd(x)/sd(y).\n")
        f.write("Significance stars computed from model p-values (* p<.05, ** p<.01, *** p<.001).\n")
        f.write("NOTE: Hispanic indicator is not available in the provided variable list; it is coded as missing.\n")
        f.write("\n")
        f.write("Model A DV: Count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: Count of dislikes among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "missingness": miss_df,
    }