def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_numeric(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def recode_binary_series(s, mapping):
        # mapping: {raw_value: recoded_value}, all else -> NaN
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        for k, v in mapping.items():
            out = out.mask(s == k, v)
        return out

    def genre_dislike_indicator(s):
        # 1 if in {4,5}; 0 if in {1,2,3}; else NaN
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out = out.mask(s.isin([1, 2, 3]), 0.0)
        out = out.mask(s.isin([4, 5]), 1.0)
        return out

    def standardize_series(s):
        s = pd.to_numeric(s, errors="coerce")
        m = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd is None or np.isnan(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - m) / sd

    def fit_standardized_ols(df_model, y_col, x_cols):
        # Standardize y and each x (including dummies) to produce standardized betas.
        y = standardize_series(df_model[y_col])
        Xz = pd.DataFrame(index=df_model.index)
        for c in x_cols:
            Xz[c] = standardize_series(df_model[c])
        # Drop rows with any missing after standardization
        d = pd.concat([y.rename("y"), Xz], axis=1).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        y_clean = d["y"]
        X_clean = sm.add_constant(d[x_cols], has_constant="add")

        model = sm.OLS(y_clean, X_clean).fit()
        # Return only standardized betas (exclude constant) plus p-values
        res = pd.DataFrame(
            {
                "beta_std": model.params.drop("const"),
                "p_value": model.pvalues.drop("const"),
            }
        )
        fit = {
            "n": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        }
        return model, res, fit, d

    # -----------------------------
    # Load and basic cleaning
    # -----------------------------
    df = pd.read_csv(data_source)

    # normalize column names to lowercase to match provided sample
    df.columns = [c.strip().lower() for c in df.columns]

    # required columns list (only those we use)
    needed = [
        "year", "id",
        "hompop", "educ", "realinc", "prestg80",
        "sex", "age", "race", "relig", "denom", "region",
        "rachaf", "busing", "racdif1", "racdif3", "racdif4",
        "bigband", "blugrass", "country", "blues", "musicals", "classicl", "folk",
        "gospel", "jazz", "latin", "moodeasy", "newage", "opera", "rap", "reggae",
        "conrock", "oldies", "hvymetal",
    ]
    # Convert numeric-like columns
    df = to_numeric(df, [c for c in needed if c in df.columns])

    # Filter 1993
    if "year" not in df.columns:
        raise ValueError("Expected column 'year' not found.")
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct variables
    # -----------------------------
    # Dependent variables: dislike counts
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c in df.columns:
            df[f"dislike_{c}"] = genre_dislike_indicator(df[c])
        else:
            df[f"dislike_{c}"] = np.nan

    # "DK treated as missing and missing cases excluded": require all component items observed
    df["dislike_minority_genres"] = df[[f"dislike_{c}" for c in minority_items]].sum(axis=1, min_count=len(minority_items))
    df["dislike_other12_genres"] = df[[f"dislike_{c}" for c in other12_items]].sum(axis=1, min_count=len(other12_items))

    # Racism score (0-5), require all five components observed
    if "rachaf" in df.columns:
        df["rac1"] = recode_binary_series(df["rachaf"], {1: 1.0, 2: 0.0})
    else:
        df["rac1"] = np.nan
    if "busing" in df.columns:
        df["rac2"] = recode_binary_series(df["busing"], {2: 1.0, 1: 0.0})
    else:
        df["rac2"] = np.nan
    if "racdif1" in df.columns:
        df["rac3"] = recode_binary_series(df["racdif1"], {2: 1.0, 1: 0.0})
    else:
        df["rac3"] = np.nan
    if "racdif3" in df.columns:
        df["rac4"] = recode_binary_series(df["racdif3"], {2: 1.0, 1: 0.0})
    else:
        df["rac4"] = np.nan
    if "racdif4" in df.columns:
        df["rac5"] = recode_binary_series(df["racdif4"], {1: 1.0, 2: 0.0})
    else:
        df["rac5"] = np.nan

    df["racism_score"] = df[["rac1", "rac2", "rac3", "rac4", "rac5"]].sum(axis=1, min_count=5)

    # SES controls
    df["education_years"] = pd.to_numeric(df.get("educ", np.nan), errors="coerce")

    # Income per capita: guard against hompop<=0 and inf
    hompop = pd.to_numeric(df.get("hompop", np.nan), errors="coerce")
    realinc = pd.to_numeric(df.get("realinc", np.nan), errors="coerce")
    df["hh_income_per_capita"] = np.where((hompop > 0) & (~hompop.isna()) & (~realinc.isna()), realinc / hompop, np.nan)
    df["hh_income_per_capita"] = pd.to_numeric(df["hh_income_per_capita"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    df["occ_prestige"] = pd.to_numeric(df.get("prestg80", np.nan), errors="coerce")

    # Demographics
    sex = pd.to_numeric(df.get("sex", np.nan), errors="coerce")
    df["female"] = recode_binary_series(sex, {2: 1.0, 1: 0.0})

    df["age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")

    race = pd.to_numeric(df.get("race", np.nan), errors="coerce")
    df["black"] = np.where(race.isin([1, 2, 3]), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.isin([1, 2, 3]), (race == 3).astype(float), np.nan)

    # Hispanic not available in provided variables -> omit to avoid inventing a proxy.
    # Keep placeholder as NaN so it won't be used.
    df["hispanic"] = np.nan

    relig = pd.to_numeric(df.get("relig", np.nan), errors="coerce")
    denom = pd.to_numeric(df.get("denom", np.nan), errors="coerce")

    # Conservative Protestant approximation:
    # RELIG==1 Protestant AND DENOM in {1,6,7}
    df["cons_protestant"] = np.where(
        (~relig.isna()) & (~denom.isna()) & (relig == 1) & (denom.isin([1, 6, 7])),
        1.0,
        np.where((~relig.isna()) & (~denom.isna()) & (relig == 1) & (~denom.isin([1, 6, 7])), 0.0, np.nan)
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(
        relig.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        (relig == 4).astype(float),
        np.nan
    )

    # South: REGION==3
    region = pd.to_numeric(df.get("region", np.nan), errors="coerce")
    df["south"] = np.where(region.isin([1, 2, 3, 4]), (region == 3).astype(float), np.nan)

    # -----------------------------
    # Model specs (omit hispanic due to not present)
    # -----------------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age",
        "black",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Ensure all model columns exist
    for c in x_cols + ["dislike_minority_genres", "dislike_other12_genres"]:
        if c not in df.columns:
            df[c] = np.nan

    # -----------------------------
    # Fit both models
    # -----------------------------
    results = {}

    def run_one(dv_name, label):
        cols_needed = [dv_name] + x_cols
        d0 = df[cols_needed].copy()
        # Drop rows with any missing in raw variables before standardization to stabilize
        d0 = d0.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # If too few rows, return empty
        if d0.shape[0] < 30:
            return None, None, {"n": int(d0.shape[0]), "r2": np.nan, "adj_r2": np.nan}, d0

        model, tab, fit, d_used = fit_standardized_ols(d0, dv_name, x_cols)
        tab = tab.sort_index()
        tab.index.name = "predictor"
        tab.insert(0, "dv", label)
        return model, tab, fit, d_used

    mA, tabA, fitA, usedA = run_one("dislike_minority_genres", "A_dislike_minority6")
    mB, tabB, fitB, usedB = run_one("dislike_other12_genres", "B_dislike_other12")

    if tabA is not None:
        results["Table2_ModelA"] = tabA
    else:
        results["Table2_ModelA"] = pd.DataFrame({"note": ["Model A could not be estimated (insufficient complete cases)."]})

    if tabB is not None:
        results["Table2_ModelB"] = tabB
    else:
        results["Table2_ModelB"] = pd.DataFrame({"note": ["Model B could not be estimated (insufficient complete cases)."]})

    fit_df = pd.DataFrame(
        [
            {"model": "A_dislike_minority6", **fitA},
            {"model": "B_dislike_other12", **fitB},
        ]
    )
    results["Fit"] = fit_df

    # -----------------------------
    # Save human-readable outputs
    # -----------------------------
    def df_to_text(dframe, path):
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(dframe, pd.DataFrame):
                f.write(dframe.to_string(index=True))
            else:
                f.write(str(dframe))
            f.write("\n")

    df_to_text(results["Table2_ModelA"], "./output/table2_modelA_standardized_betas.txt")
    df_to_text(results["Table2_ModelB"], "./output/table2_modelB_standardized_betas.txt")
    df_to_text(results["Fit"], "./output/table2_fit_stats.txt")

    # Also save statsmodels summaries if available
    if mA is not None:
        with open("./output/table2_modelA_full_summary.txt", "w", encoding="utf-8") as f:
            f.write(mA.summary().as_text())
            f.write("\n")
    if mB is not None:
        with open("./output/table2_modelB_full_summary.txt", "w", encoding="utf-8") as f:
            f.write(mB.summary().as_text())
            f.write("\n")

    # Return results as dict of DataFrames
    return results