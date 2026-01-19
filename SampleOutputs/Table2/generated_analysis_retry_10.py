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
        Conservative NA-code cleaning for this extract:
        - Convert to numeric
        - Treat common GSS NA sentinels as missing
        Note: We do NOT blanket-drop values like 0/negative unless the variable logically cannot take them.
        """
        x = to_num(x).copy()
        na_vals = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(na_vals))
        return x

    def likert_dislike_indicator(x):
        """
        Music liking items: 1-5 with 4/5 = dislike.
        Missing if not in 1..5 after NA cleaning.
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

    def build_count_completecase(df, items):
        """
        Sum dislike indicators across items; require ALL items observed (complete-case for DV),
        consistent with "DK treated as missing and missing cases excluded".
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def is_valid_design_matrix(X):
        if X is None or X.shape[0] == 0 or X.shape[1] == 0:
            return False
        if not np.isfinite(X.to_numpy()).all():
            return False
        return True

    def standardized_betas_from_unstd(y, X_noconst, b_unstd_noconst):
        """
        beta_j = b_j * sd(x_j) / sd(y)
        Computed on the estimation sample.
        """
        y_sd = y.std(ddof=0)
        if not np.isfinite(y_sd) or y_sd == 0:
            return pd.Series(np.nan, index=X_noconst.columns)
        x_sds = X_noconst.std(axis=0, ddof=0)
        betas = b_unstd_noconst * (x_sds / y_sd)
        return betas

    def sig_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df_in, dv_col, x_cols, model_name):
        """
        Fit OLS on unstandardized DV (so intercept is meaningful).
        Compute standardized coefficients (betas) post-hoc for slopes.
        Return:
          - table with: b_unstd, beta_std, p_value, sig
          - fit stats dataframe
        """
        needed = [dv_col] + x_cols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Ensure we have enough data
        if d.shape[0] < (len(x_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = to_num(d[dv_col])
        X = d[x_cols].apply(to_num)

        # Drop zero-variance predictors (prevents singularities and NaN betas)
        nunique = X.nunique(dropna=True)
        keep_cols = [c for c in X.columns if nunique.get(c, 0) > 1]
        dropped = [c for c in X.columns if c not in keep_cols]
        X = X[keep_cols]

        Xc = sm.add_constant(X, has_constant="add")

        # Final sanity: no inf/nan
        mask_ok = np.isfinite(y.to_numpy()) & np.isfinite(Xc.to_numpy()).all(axis=1)
        y = y.loc[mask_ok]
        Xc = Xc.loc[mask_ok]
        X = X.loc[mask_ok]

        if y.shape[0] < (X.shape[1] + 2):
            raise ValueError(f"{model_name}: not enough cases after finite filtering (n={y.shape[0]}, k={X.shape[1]}).")

        if not is_valid_design_matrix(Xc):
            raise ValueError(f"{model_name}: invalid design matrix (nan/inf or empty).")

        model = sm.OLS(y, Xc).fit()

        params = model.params.copy()
        pvals = model.pvalues.copy()

        # standardized betas for slopes only (constant has no standardized beta)
        betas = standardized_betas_from_unstd(y, X, params.drop(labels=["const"], errors="ignore"))
        beta_std = pd.Series(np.nan, index=params.index, dtype="float64")
        for c in betas.index:
            if c in beta_std.index:
                beta_std.loc[c] = betas.loc[c]

        out = pd.DataFrame(
            {
                "b_unstd": params,
                "beta_std": beta_std,
                "p_value": pvals,
            }
        )
        out["sig"] = out["p_value"].apply(sig_stars)

        # Put rows in requested Table 2 order (where available), then others, then const last
        desired_order = ["racism_score", "education_years", "hh_income_per_capita", "occ_prestige",
                         "female", "age_years", "black", "hispanic", "other_race",
                         "cons_protestant", "no_religion", "south"]
        rows = []
        for c in desired_order:
            if c in out.index:
                rows.append(c)
        # Any remaining non-const terms not in desired list
        remaining = [c for c in out.index if c not in rows and c != "const"]
        rows += remaining
        if "const" in out.index:
            rows += ["const"]
        out = out.loc[rows]

        fit = {
            "model": model_name,
            "dv": dv_col,
            "n": int(model.nobs),
            "k_predictors": int(model.df_model),  # excludes intercept
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
        }
        fit_df = pd.DataFrame([fit])

        # Write human-readable files
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nComputed standardized coefficients (betas) are provided in the table file.\n")
            if dropped:
                f.write(f"\nDropped zero-variance predictors: {', '.join(dropped)}\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv_col}\n\n")
            f.write(out.to_string(float_format=lambda v: f"{v: .6f}"))
            f.write("\n")

        return out, fit_df

    # -----------------------------
    # Load / filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

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
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided extract -> keep as missing; model will listwise-drop it if included.
    # To keep the Table 2 structure without killing the sample, we include it ONLY if it exists and has variation.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index, dtype="float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index, dtype="float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index, dtype="float64")
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Fit models
    # -----------------------------
    # Keep Table 2 RHS terms. Because hispanic is unavailable here, we include it only if it has any non-missing.
    x_cols = [
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

    # If Hispanic is entirely missing, drop it to avoid collapsing sample to n=0.
    if df["hispanic"].notna().sum() == 0:
        x_cols = [c for c in x_cols if c != "hispanic"]

    results = {}

    tabA, fitA = fit_table2_model(df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6")
    tabB, fitB = fit_table2_model(df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12")

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (GSS 1993): OLS on unstandardized DV; standardized betas computed post-estimation\n")
        f.write("Notes:\n")
        f.write("- Standardized coefficients (beta) computed as b * sd(x)/sd(y) on each model's estimation sample.\n")
        f.write("- Intercept is from the unstandardized OLS model.\n")
        if "hispanic" not in x_cols:
            f.write("- Hispanic dummy not estimated because no Hispanic identifier is present in the provided extract.\n")
        f.write("\nModel A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results