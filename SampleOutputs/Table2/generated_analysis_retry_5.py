def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def _to_numeric(df):
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _set_gss_missing(s: pd.Series) -> pd.Series:
        """
        Best-effort missing handling: treat common GSS NA codes as missing.
        Works for both integer-coded and float-coded columns.
        """
        s = pd.to_numeric(s, errors="coerce")
        na_codes = {0, 8, 9, 98, 99, 998, 999, 9998, 9999}
        return s.where(~s.isin(list(na_codes)))

    def _binary_from_1_2(s: pd.Series, one_is_one=True) -> pd.Series:
        """
        For variables coded 1/2: returns 1/0, missing otherwise.
        If one_is_one=True: 1->1, 2->0
        else: 1->0, 2->1
        """
        s = _set_gss_missing(s)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        if one_is_one:
            out = out.mask(s == 1, 1.0)
            out = out.mask(s == 2, 0.0)
        else:
            out = out.mask(s == 1, 0.0)
            out = out.mask(s == 2, 1.0)
        return out

    def _dislike_indicator(s: pd.Series) -> pd.Series:
        """
        5-point like/dislike:
          1,2,3 -> 0
          4,5 -> 1
          other/NA-coded -> missing
        """
        s = _set_gss_missing(s)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out = out.mask(s.isin([1, 2, 3]), 0.0)
        out = out.mask(s.isin([4, 5]), 1.0)
        return out

    def _sum_with_complete_case(df_sub: pd.DataFrame) -> pd.Series:
        """
        Paper summary implies DK treated as missing and missing cases excluded.
        So: require all items observed to form the count.
        """
        return df_sub.sum(axis=1, min_count=df_sub.shape[1])

    def _zscore(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        m = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd is None or not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - m) / sd

    def _fit_standardized_ols(d: pd.DataFrame, dv: str, xcols: list, model_name: str):
        cols_needed = [dv] + xcols
        d0 = d[cols_needed].copy()

        # Replace inf with nan before dropping
        d0 = d0.replace([np.inf, -np.inf], np.nan)

        # Drop incomplete cases (Table 2 uses listwise deletion per model)
        d0 = d0.dropna(axis=0, how="any")

        if d0.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases after cleaning (n={d0.shape[0]}, k={len(xcols)}).")

        # Standardize DV and all predictors to get standardized coefficients
        z = pd.DataFrame(index=d0.index)
        z[dv] = _zscore(d0[dv])
        for c in xcols:
            z[c] = _zscore(d0[c])

        # Drop any rows that became nan due to zero variance standardization
        z = z.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if z.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough cases after standardization (n={z.shape[0]}, k={len(xcols)}).")

        y = z[dv].astype(float)
        X = sm.add_constant(z[xcols].astype(float), has_constant="add")

        model = sm.OLS(y, X).fit()

        # Coef table
        tab = pd.DataFrame(
            {
                "beta_std": model.params,
                "se": model.bse,
                "t": model.tvalues,
                "p": model.pvalues,
            }
        )
        # Keep const too, though not typically interpreted as standardized; include for completeness
        fit = pd.DataFrame(
            {
                "n": [int(model.nobs)],
                "r2": [float(model.rsquared)],
                "adj_r2": [float(model.rsquared_adj)],
                "df_model": [float(model.df_model)],
                "df_resid": [float(model.df_resid)],
            },
            index=[model_name],
        )
        return model, tab, fit, z.index

    # ------------------------
    # Load and prepare data
    # ------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]
    df = _to_numeric(df)

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Required column 'year' not found.")
    df = df.loc[df["year"] == 1993].copy()

    # Ensure id exists (not required for modeling but for reference)
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)

    # ------------------------
    # Construct DVs
    # ------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    missing_music = [c for c in (minority_items + other12_items) if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music taste columns: {missing_music}")

    for c in minority_items + other12_items:
        df[c] = _set_gss_missing(df[c])

    dislike_min_df = pd.DataFrame({c: _dislike_indicator(df[c]) for c in minority_items})
    dislike_oth_df = pd.DataFrame({c: _dislike_indicator(df[c]) for c in other12_items})

    df["dislike_minority_genres"] = _sum_with_complete_case(dislike_min_df)
    df["dislike_other12_genres"] = _sum_with_complete_case(dislike_oth_df)

    # ------------------------
    # Construct racism scale (0-5)
    # ------------------------
    needed_rac = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    missing_rac = [c for c in needed_rac if c not in df.columns]
    if missing_rac:
        raise ValueError(f"Missing required racism columns: {missing_rac}")

    df["rac1"] = _binary_from_1_2(df["rachaf"], one_is_one=True)   # object if 1
    df["rac2"] = _binary_from_1_2(df["busing"], one_is_one=False)  # oppose if 2
    df["rac3"] = _binary_from_1_2(df["racdif1"], one_is_one=False) # deny discrim if 2
    df["rac4"] = _binary_from_1_2(df["racdif3"], one_is_one=False) # deny edu chance if 2
    df["rac5"] = _binary_from_1_2(df["racdif4"], one_is_one=True)  # endorse willpower if 1

    df["racism_score"] = _sum_with_complete_case(df[["rac1", "rac2", "rac3", "rac4", "rac5"]])

    # ------------------------
    # Controls
    # ------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing required column 'educ'.")
    df["education_years"] = _set_gss_missing(df["educ"])

    # HH income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing required columns for income per capita: 'realinc' and/or 'hompop'.")
    df["realinc"] = _set_gss_missing(df["realinc"])
    df["hompop"] = _set_gss_missing(df["hompop"])
    df.loc[df["hompop"] == 0, "hompop"] = np.nan
    df["hh_income_per_capita"] = df["realinc"] / df["hompop"]
    df["hh_income_per_capita"] = df["hh_income_per_capita"].replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing required column 'prestg80'.")
    df["occ_prestige"] = _set_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing required column 'sex'.")
    # GSS: 1=male, 2=female
    df["female"] = _binary_from_1_2(df["sex"], one_is_one=False)

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing required column 'age'.")
    df["age_years"] = _set_gss_missing(df["age"])

    # Race indicators from RACE (1=white, 2=black, 3=other)
    if "race" not in df.columns:
        raise ValueError("Missing required column 'race'.")
    r = _set_gss_missing(df["race"])
    df["black"] = pd.Series(np.nan, index=df.index, dtype="float64")
    df["other_race"] = pd.Series(np.nan, index=df.index, dtype="float64")
    # define when race observed
    obs = r.notna()
    df.loc[obs, "black"] = (r.loc[obs] == 2).astype(float)
    df.loc[obs, "other_race"] = (r.loc[obs] == 3).astype(float)

    # Hispanic is not available in provided data: create as missing, then exclude from models
    df["hispanic"] = np.nan

    # Conservative Protestant proxy
    # RELIG: 1=protestant, 4=none. DENOM broad: 1=baptist, 6=other, 7=nondenom
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing required columns for religion controls: 'relig' and/or 'denom'.")
    relig = _set_gss_missing(df["relig"])
    denom = _set_gss_missing(df["denom"])
    df["cons_protestant"] = np.nan
    obs_rel = relig.notna() & denom.notna()
    df.loc[obs_rel, "cons_protestant"] = (
        (relig.loc[obs_rel] == 1) & (denom.loc[obs_rel].isin([1, 6, 7]))
    ).astype(float)

    # No religion
    df["no_religion"] = np.nan
    obs_rel2 = relig.notna()
    df.loc[obs_rel2, "no_religion"] = (relig.loc[obs_rel2] == 4).astype(float)

    # South
    if "region" not in df.columns:
        raise ValueError("Missing required column 'region'.")
    region = _set_gss_missing(df["region"])
    df["south"] = np.nan
    obs_reg = region.notna()
    df.loc[obs_reg, "south"] = (region.loc[obs_reg] == 3).astype(float)

    # ------------------------
    # Fit two models
    # Note: Hispanic dummy unavailable -> excluded to avoid empty design matrix after listwise deletion.
    # ------------------------
    x_cols = [
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

    results = {}

    def run_one(dv_col: str, model_name: str):
        model, tab, fit, idx = _fit_standardized_ols(df, dv_col, x_cols, model_name)

        # Save human-readable output
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Coefficients shown above are from standardized regression (DV and all X z-scored).\n")
            f.write("      'const' is included after standardization; interpret betas for predictors.\n")

        tab_out = tab.copy()
        tab_out.index.name = "term"
        tab_out.to_csv(f"./output/{model_name}_coef_table.csv")
        with open(f"./output/{model_name}_coef_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_out.to_string(float_format=lambda x: f"{x: .6f}"))

        fit.to_csv(f"./output/{model_name}_fit.csv")
        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(float_format=lambda x: f"{x: .6f}"))

        return tab_out, fit

    tabA, fitA = run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    tabB, fitB = run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    results["Table2_ModelA_coef"] = tabA
    results["Table2_ModelA_fit"] = fitA
    results["Table2_ModelB_coef"] = tabB
    results["Table2_ModelB_fit"] = fitB

    # Combined quick view
    combined = pd.concat(
        [
            tabA[["beta_std", "p"]].rename(columns={"beta_std": "ModelA_beta", "p": "ModelA_p"}),
            tabB[["beta_std", "p"]].rename(columns={"beta_std": "ModelB_beta", "p": "ModelB_p"}),
        ],
        axis=1,
    )
    results["Table2_combined_betas"] = combined
    with open("./output/Table2_combined_betas.txt", "w", encoding="utf-8") as f:
        f.write(combined.to_string(float_format=lambda x: f"{x: .6f}"))

    # Minimal provenance / diagnostics
    diag = pd.DataFrame(
        {
            "n_total_1993": [int(df.shape[0])],
            "n_nonmissing_DV_A": [int(df["dislike_minority_genres"].notna().sum())],
            "n_nonmissing_DV_B": [int(df["dislike_other12_genres"].notna().sum())],
            "n_nonmissing_racism": [int(df["racism_score"].notna().sum())],
        }
    )
    results["diagnostics"] = diag
    diag.to_csv("./output/Table2_diagnostics.csv", index=False)
    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write(diag.to_string(index=False))

    return results