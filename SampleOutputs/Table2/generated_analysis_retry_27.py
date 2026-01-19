def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_missing_like_gss(x):
        """
        Conservative missing handling for this extract:
        - coerce to numeric
        - treat common GSS sentinel codes as missing
        - keep other values as-is
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(series):
        """
        Music taste items: 1-5, where 4/5 = dislike.
        Non-1..5 and sentinel codes => missing.
        """
        x = clean_missing_like_gss(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_missing_like_gss(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_dislikes(df, cols):
        """
        Count dislikes across a specified set of items.
        Match the paper-style 'DK treated as missing' by requiring complete responses
        on the items in the count (listwise for those items only).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in cols], axis=1)
        # require all items observed
        return mat.sum(axis=1, min_count=len(cols))

    def standardize_betas_from_unstd(y, X, params_unstd):
        """
        Standardized beta for each non-constant regressor:
            beta_j = b_j * sd(X_j) / sd(Y)
        Uses sample SDs (ddof=0) computed on the estimation sample.
        """
        y_sd = y.std(ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = X[col].std(ddof=0)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = params_unstd[col] * (x_sd / y_sd)
        return betas

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

    def fit_table2_model(df, dv_col, x_cols_ordered, model_name):
        """
        Fit OLS on unstandardized DV with intercept, then compute standardized betas
        for predictors. Keep constant as unstandardized coefficient.
        """
        needed = [dv_col] + x_cols_ordered
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        # Drop any predictors with zero variance in the estimation sample (prevents NaN rows)
        zero_var = []
        for c in x_cols_ordered:
            s = d[c]
            if s.nunique(dropna=True) <= 1:
                zero_var.append(c)
        x_cols_used = [c for c in x_cols_ordered if c not in zero_var]

        if d.shape[0] < (len(x_cols_used) + 2):
            raise ValueError(
                f"{model_name}: not enough complete cases after cleaning "
                f"(n={d.shape[0]}, k={len(x_cols_used)}). Dropped zero-var: {zero_var}"
            )

        y = d[dv_col].astype(float)
        X = d[x_cols_used].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        betas = standardize_betas_from_unstd(y, Xc, model.params)

        # Output table in Table-2-like style:
        # - standardized betas for predictors
        # - unstandardized constant
        rows = []
        for c in x_cols_ordered:
            if c in zero_var:
                rows.append(
                    {
                        "term": c,
                        "beta_std": np.nan,
                        "p_value_reest": np.nan,
                        "sig": "",
                        "note": "dropped_zero_variance",
                    }
                )
            else:
                p = float(model.pvalues.get(c, np.nan))
                rows.append(
                    {
                        "term": c,
                        "beta_std": float(betas.get(c, np.nan)),
                        "p_value_reest": p,
                        "sig": star_from_p(p),
                        "note": "",
                    }
                )

        # Constant (unstandardized)
        p_const = float(model.pvalues.get("const", np.nan))
        rows.append(
            {
                "term": "Constant",
                "beta_std": np.nan,
                "p_value_reest": p_const,
                "sig": star_from_p(p_const),
                "note": f"unstandardized_const={float(model.params.get('const', np.nan)):.6f}",
            }
        )

        out = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
                }
            ]
        )

        return model, out, fit, d.index

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Year filter
    if "year" not in df.columns:
        raise ValueError("Missing YEAR column.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables (counts)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_dislikes(df, minority_items)
    df["dislike_other12_genres"] = build_count_dislikes(df, other12_items)

    # -------------------------
    # Racism score (0-5 additive)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny edu chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_missing_like_gss(df["educ"]).where(clean_missing_like_gss(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # HH income per capita: realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_missing_like_gss(df["realinc"])
    hompop = clean_missing_like_gss(df["hompop"]).where(clean_missing_like_gss(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_missing_like_gss(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_missing_like_gss(df["age"]).where(clean_missing_like_gss(df["age"]).between(18, 89))
    df["age_years"] = age

    # Race dummies: black, other_race (white reference)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_missing_like_gss(df["race"]).where(clean_missing_like_gss(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables; create as all-missing (kept for spec fidelity)
    # If later a proper Hispanic identifier is added to the extract, this will be used automatically.
    if "hispanic" not in df.columns:
        df["hispanic"] = np.nan

    # Religion: conservative Protestant proxy & no religion
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")

    relig = clean_missing_like_gss(df["relig"])
    denom = clean_missing_like_gss(df["denom"])

    # Conservative Protestant proxy (implementable with available fields)
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index, dtype="float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index, dtype="float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # South (REGION == 3 in this extract)
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_missing_like_gss(df["region"]).where(clean_missing_like_gss(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Model specification (Table 2 RHS order)
    # -------------------------
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
            raise ValueError(f"Missing constructed predictor column: {c}")

    results = {}

    def write_outputs(model, table_df, fit_df, model_name):
        # Human-readable text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTES:\n")
            f.write("- Standardized coefficients (beta_std) are computed as b * sd(X)/sd(Y) on the estimation sample.\n")
            f.write("- Constant reported in the model summary is unstandardized.\n")
            f.write("- p-values (and stars) are from this re-estimation; the original paper table does not report SEs.\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(table_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

    # Fit Model A
    mA, tabA, fitA, idxA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    write_outputs(mA, tabA, fitA, "Table2_ModelA_dislike_minority6")

    # Fit Model B
    mB, tabB, fitB, idxB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )
    write_outputs(mB, tabB, fitB, "Table2_ModelB_dislike_other12")

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS with standardized coefficients (betas)\n")
        f.write("DVs:\n")
        f.write(" - Model A: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(" - Model B: count of dislikes among the remaining 12 genres in the extract\n\n")
        f.write("Predictors (ordered):\n")
        f.write(", ".join(x_order) + "\n\n")
        f.write("Fit statistics:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write("Important limitation:\n")
        f.write("- If the dataset does not contain a true Hispanic identifier, 'hispanic' will be all-missing and dropped via listwise deletion,\n")
        f.write("  which can reduce N. Provide a valid Hispanic variable to fully match the paper specification.\n")

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    return results