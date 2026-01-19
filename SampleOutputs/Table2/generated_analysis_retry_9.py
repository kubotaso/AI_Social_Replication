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

    def clean_gss_na(x):
        """
        Conservative missing-code cleaning for this extract:
        - Treat common GSS sentinels as missing.
        - Keep legitimate values (incl. 0 where meaningful).
        """
        x = to_num(x).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 where 4/5 indicate dislike.
        Return float 0/1 with NaN for invalid/missing.
        """
        x = clean_gss_na(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_gss_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all=True):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_all:
            # listwise on the items used to build the DV (DK treated as missing -> exclude)
            return mat.sum(axis=1, min_count=len(items))
        # (not used here)
        return mat.sum(axis=1, min_count=1)

    def standardize_series(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_ols_with_betas(df, y_col, x_cols, model_name):
        """
        Fit OLS on unstandardized y and x (with intercept).
        Compute standardized betas post-hoc: beta_j = b_j * sd(x_j) / sd(y).
        Intercept is reported unstandardized.
        """
        needed = [y_col] + x_cols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Require enough observations
        if d.shape[0] < (len(x_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = to_num(d[y_col]).astype(float)
        X = d[x_cols].apply(to_num).astype(float)
        X = sm.add_constant(X, has_constant="add")

        # Final safety: no NaN/inf in exog/endog
        ok = np.isfinite(y.values) & np.isfinite(X.values).all(axis=1)
        y = y.loc[ok]
        X = X.loc[ok]
        if y.shape[0] < (X.shape[1] + 1):
            raise ValueError(f"{model_name}: not enough finite cases after cleaning (n={y.shape[0]}, p={X.shape[1]}).")

        model = sm.OLS(y, X).fit()

        # Compute standardized betas for non-constant terms
        sd_y = y.std(ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                betas[col] = np.nan
            else:
                sd_x = X[col].std(ddof=0)
                if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                    betas[col] = np.nan
                else:
                    betas[col] = model.params[col] * (sd_x / sd_y)

        tab = pd.DataFrame(
            {
                "b_unstd": model.params,
                "beta_std": pd.Series(betas),
                "p_value": model.pvalues,
            }
        )

        def stars(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        tab["sig"] = tab["p_value"].apply(stars)

        fit = {
            "model": model_name,
            "dv": y_col,
            "n": int(model.nobs),
            "k_predictors_plus_const": int(model.df_model + 1),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        }

        return model, tab, pd.DataFrame([fit]), d.index

    # -------------------------
    # Load and filter
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Variables per mapping
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Dependent variables (require all component items present)
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all=True)

    # Racism score (0-5), require all five items present
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

    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' column.")
    edu = clean_gss_na(df["educ"]).where(clean_gss_na(df["educ"]).between(0, 20))
    df["education_years"] = edu

    # Household income per capita
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' needed for income per capita.")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(clean_gss_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column.")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' column.")
    df["age_years"] = clean_gss_na(df["age"]).where(clean_gss_na(df["age"]).between(18, 89))

    # Race dummies: white is reference
    if "race" not in df.columns:
        raise ValueError("Missing 'race' column.")
    race = clean_gss_na(df["race"]).where(clean_gss_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables.
    # To keep model faithful and avoid dropping all cases, include a 0 dummy (no Hispanic ID in extract).
    # This preserves the column and avoids runtime errors; it will have 0 variance and be dropped below.
    df["hispanic"] = 0.0

    # Religion dummies (proxy per mapping)
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' column.")
    relig = clean_gss_na(df["relig"])
    denom = clean_gss_na(df["denom"])

    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.mask(relig.isna() | denom.isna())
    df["cons_protestant"] = consprot

    norelig = (relig == 4).astype(float)
    norelig = norelig.mask(relig.isna())
    df["no_religion"] = norelig

    # South
    if "region" not in df.columns:
        raise ValueError("Missing 'region' column.")
    region = clean_gss_na(df["region"]).where(clean_gss_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Model spec (Table 2 RHS)
    # NOTE: Hispanic is included but will be dropped if it is constant in estimation sample.
    # -------------------------
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

    # Drop predictors that are constant (or all-missing) in the full 1993 data to prevent empty/NaN exog errors.
    # Keep order stable.
    def drop_constant_predictors(dfin, cols):
        kept = []
        dropped = []
        for c in cols:
            s = dfin[c]
            # consider only non-missing
            s_non = s.dropna()
            if s_non.shape[0] == 0:
                dropped.append((c, "all_missing"))
                continue
            if s_non.nunique() <= 1:
                dropped.append((c, "constant"))
                continue
            kept.append(c)
        return kept, dropped

    x_cols_kept, dropped_info = drop_constant_predictors(df, x_cols)

    # -------------------------
    # Run two models
    # -------------------------
    results = {}

    modelA, tabA, fitA, usedA = fit_ols_with_betas(
        df, "dislike_minority_genres", x_cols_kept, "Table2_ModelA_dislike_minority6"
    )
    modelB, tabB, fitB, usedB = fit_ols_with_betas(
        df, "dislike_other12_genres", x_cols_kept, "Table2_ModelB_dislike_other12"
    )

    # Order rows to match requested presentation: predictors then constant last (paper shows constant last)
    def reorder_table(tab, xcols_kept):
        cols_order = xcols_kept + ["const"]
        cols_order = [c for c in cols_order if c in tab.index]
        return tab.loc[cols_order, ["beta_std", "sig", "b_unstd", "p_value"]]

    tabA_out = reorder_table(tabA, x_cols_kept)
    tabB_out = reorder_table(tabB, x_cols_kept)

    # -------------------------
    # Save outputs
    # -------------------------
    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(modelA.summary().as_text())
    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(modelB.summary().as_text())

    with open("./output/Table2_ModelA_table.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication – Model A (minority-associated 6 genres)\n")
        f.write("Columns: beta_std (standardized coefficient), sig (stars from computed p), b_unstd, p_value\n\n")
        f.write(tabA_out.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n")
    with open("./output/Table2_ModelB_table.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication – Model B (other 12 genres)\n")
        f.write("Columns: beta_std (standardized coefficient), sig (stars from computed p), b_unstd, p_value\n\n")
        f.write(tabB_out.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    overview = []
    overview.append("Table 2 replication (GSS 1993) – OLS with post-hoc standardized coefficients (betas)\n")
    overview.append("Notes:\n")
    overview.append("- Models are estimated on unstandardized DV, with an intercept.\n")
    overview.append("- Standardized coefficients computed as beta = b * sd(x)/sd(y) over the estimation sample.\n")
    overview.append("- 'sig' stars are computed from model p-values (two-tailed), since the paper prints stars not SEs.\n")
    if dropped_info:
        overview.append("\nPredictors dropped for zero variance / all-missing in this extract:\n")
        for name, why in dropped_info:
            overview.append(f"  - {name}: {why}\n")

    overview.append("\nModel A fit:\n")
    overview.append(fitA.to_string(index=False))
    overview.append("\n\nModel B fit:\n")
    overview.append(fitB.to_string(index=False))
    overview.append("\n")

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("".join(overview))

    results["ModelA_table"] = tabA_out
    results["ModelB_table"] = tabB_out
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    return results