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

    def clean_gss_missing(s):
        """
        Conservative missing-code handling for this extract:
        - Convert to numeric
        - Treat common GSS sentinel codes as missing
        """
        x = to_num(s).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items are 1-5. Dislike if 4 or 5.
        Missing if not in [1,5] after cleaning.
        """
        x = clean_gss_missing(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_recode(s, true_codes, false_codes):
        x = clean_gss_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, item_cols, require_all_items=True):
        """
        Count dislikes across listed item columns.
        Paper: DK treated as missing and cases excluded.
        Default here: require all items non-missing to compute the count.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in item_cols], axis=1)
        if require_all_items:
            return mat.sum(axis=1, min_count=len(item_cols))
        return mat.sum(axis=1, min_count=1)

    def standardized_betas_from_ols(y, X_with_const, ols_res):
        """
        Compute standardized coefficients (beta weights) from an OLS fit where y is unstandardized.
        Beta_j = b_j * sd(x_j) / sd(y)
        Intercept is left unstandardized (NaN for beta).
        """
        y_sd = y.std(ddof=0)
        betas = pd.Series(index=ols_res.params.index, dtype="float64")

        for name in ols_res.params.index:
            if name == "const":
                betas.loc[name] = np.nan
                continue
            x_sd = X_with_const[name].std(ddof=0)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas.loc[name] = np.nan
            else:
                betas.loc[name] = ols_res.params[name] * (x_sd / y_sd)
        return betas

    def star(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_model(df_in, dv_col, xcols, model_name):
        """
        Fit OLS on unstandardized DV with intercept.
        Return:
          - table with standardized betas + stars (and constant as unstandardized b)
          - fit stats dataframe
          - statsmodels result
        """
        needed = [dv_col] + xcols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = to_num(d[dv_col]).astype(float)
        X = d[xcols].apply(to_num).astype(float)

        # Drop any zero-variance predictors (should not happen if coding is correct, but prevents crashes)
        vari = X.apply(lambda c: np.nanvar(c.values, ddof=0))
        keep = vari[(vari > 0) & np.isfinite(vari)].index.tolist()
        dropped = [c for c in xcols if c not in keep]
        X = X[keep]

        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: design matrix is empty after dropping zero-variance predictors: {dropped}")

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_ols(y, Xc, res)

        # Build "paper-style" table: standardized betas for predictors, unstandardized constant
        rows = []
        index = []

        # constant row
        index.append("Constant")
        rows.append(
            {
                "coef": float(res.params.get("const", np.nan)),
                "std_beta": np.nan,
                "p_value": float(res.pvalues.get("const", np.nan)),
                "sig": star(float(res.pvalues.get("const", np.nan))),
            }
        )

        # predictor rows in the requested order, with any dropped predictors shown as NaN
        for c in xcols:
            index.append(c)
            if c in res.params.index:
                p = float(res.pvalues[c])
                rows.append(
                    {
                        "coef": float(res.params[c]),
                        "std_beta": float(betas[c]) if np.isfinite(betas.get(c, np.nan)) else np.nan,
                        "p_value": p,
                        "sig": star(p),
                    }
                )
            else:
                rows.append({"coef": np.nan, "std_beta": np.nan, "p_value": np.nan, "sig": ""})

        tab = pd.DataFrame(rows, index=index)
        tab.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(res.nobs),
                    "k_including_const": int(res.df_model + 1),
                    "r2": float(res.rsquared),
                    "adj_r2": float(res.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
            f.write("\n\nNOTES:\n")
            f.write("- coef = unstandardized OLS coefficient (intercept included)\n")
            f.write("- std_beta = standardized coefficient computed as b * sd(x)/sd(y) for predictors\n")
            f.write("- sig uses two-tailed thresholds: * p<.05, ** p<.01, *** p<.001 (computed from this data)\n")
            if dropped:
                f.write(f"- Dropped (zero variance): {dropped}\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(tab.to_string(float_format=lambda v: f"{v: .6f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

        return tab, fit, res

    # --------------------------
    # Load data
    # --------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year==1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # --------------------------
    # Construct DVs
    # --------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband",
        "blugrass",
        "country",
        "musicals",
        "classicl",
        "folk",
        "moodeasy",
        "newage",
        "opera",
        "conrock",
        "oldies",
        "hvymetal",
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_items=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_items=True)

    # --------------------------
    # Construct racism score (0-5)
    # --------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_recode(df["rachaf"], true_codes=[1], false_codes=[2])     # object to school >half black
    rac2 = binary_recode(df["busing"], true_codes=[2], false_codes=[1])     # oppose busing
    rac3 = binary_recode(df["racdif1"], true_codes=[2], false_codes=[1])    # deny discrimination
    rac4 = binary_recode(df["racdif3"], true_codes=[2], false_codes=[1])    # deny educational chance
    rac5 = binary_recode(df["racdif4"], true_codes=[1], false_codes=[2])    # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # --------------------------
    # SES controls
    # --------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # --------------------------
    # Demographic/identity controls
    # --------------------------
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_recode(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator is not available in provided variables: keep as missing and exclude from models.
    # To preserve Table 2 structure, we will include the column and then explicitly omit it from estimation
    # (otherwise it forces n=0 due to all-missing listwise deletion).
    df["hispanic"] = np.nan

    # Conservative Protestant proxy per mapping instruction
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # --------------------------
    # Fit Table 2 models
    # --------------------------
    # Note: Hispanic cannot be used (all missing in this dataset extract). Keep the column for reporting
    # but do not include it in estimation; otherwise listwise deletion yields n=0.
    xcols_for_estimation = [
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

    # For output tables, we will present the Table-2 intended order including a placeholder for Hispanic.
    table2_order = [
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

    tabA, fitA, resA = fit_model(df, "dislike_minority_genres", xcols_for_estimation, "Table2_ModelA_dislike_minority6")
    tabB, fitB, resB = fit_model(df, "dislike_other12_genres", xcols_for_estimation, "Table2_ModelB_dislike_other12")

    # Reindex tables to the requested Table-2 structure (Constant + 12 predictors),
    # inserting the missing Hispanic row.
    def reindex_to_table2(tab, model_name):
        rows = []
        idx = []

        # constant
        idx.append("Constant")
        rows.append(tab.loc["Constant"].to_dict())

        # predictors in Table 2 order
        for v in table2_order:
            idx.append(v)
            if v in tab.index:
                rows.append(tab.loc[v].to_dict())
            else:
                rows.append({"coef": np.nan, "std_beta": np.nan, "p_value": np.nan, "sig": ""})

        out = pd.DataFrame(rows, index=idx)
        out.index.name = "term"
        with open(f"./output/{model_name}_table_table2order.txt", "w", encoding="utf-8") as f:
            f.write(out.to_string(float_format=lambda v: f"{v: .6f}"))
        return out

    tabA_t2 = reindex_to_table2(tabA, "Table2_ModelA_dislike_minority6")
    tabB_t2 = reindex_to_table2(tabB, "Table2_ModelB_dislike_other12")

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (GSS 1993) - OLS with standardized coefficients (betas)\n")
        f.write("\nImplementation notes:\n")
        f.write("- OLS is fit on the unstandardized DV with an intercept.\n")
        f.write("- Standardized coefficients are computed post-estimation: beta = b * sd(x) / sd(y).\n")
        f.write("- Significance stars are computed from this dataset's two-tailed p-values.\n")
        f.write("- Hispanic dummy is not available in the provided extract; shown as NA in tables and not estimated.\n")
        f.write("\nModel A DV: dislike_minority_genres (count of dislikes among rap, reggae, blues, jazz, gospel, latin)\n")
        f.write(fitA.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        f.write("\n\nModel B DV: dislike_other12_genres (count of dislikes among the other 12 genres)\n")
        f.write(fitB.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        f.write("\n")

    return {
        "ModelA_table": tabA_t2,
        "ModelB_table": tabB_t2,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }