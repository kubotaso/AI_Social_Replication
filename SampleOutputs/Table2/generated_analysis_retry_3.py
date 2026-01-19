def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def read_data(path):
        d = pd.read_csv(path)
        d.columns = [c.strip().lower() for c in d.columns]
        return d

    def to_numeric_series(s):
        return pd.to_numeric(s, errors="coerce")

    def na_if_outside(s, valid_values):
        s2 = s.copy()
        s2[~s2.isin(valid_values)] = np.nan
        return s2

    def dislike_indicator(item_series):
        # 1 if 4/5, 0 if 1/2/3, missing otherwise
        s = to_numeric_series(item_series)
        s = na_if_outside(s, {1, 2, 3, 4, 5})
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out[s.isin([4, 5])] = 1.0
        out[s.isin([1, 2, 3])] = 0.0
        return out

    def dichotomous_from_12(s, one_values, zero_values):
        s = to_numeric_series(s)
        s = na_if_outside(s, set(one_values) | set(zero_values))
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out[s.isin(one_values)] = 1.0
        out[s.isin(zero_values)] = 0.0
        return out

    def zscore(s):
        s = s.astype(float)
        mu = np.nanmean(s.values)
        sd = np.nanstd(s.values, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_dislikes(df, items):
        # Missing if any component missing (listwise at DV construction)
        inds = []
        for col in items:
            if col not in df.columns:
                inds.append(pd.Series(np.nan, index=df.index, dtype="float64"))
            else:
                inds.append(dislike_indicator(df[col]))
        ind_df = pd.concat(inds, axis=1)
        # listwise across items (faithful to "DK treated as missing; cases excluded")
        return ind_df.sum(axis=1, min_count=len(items))

    def fit_standardized_ols(d, y_col, x_cols, model_name):
        # Keep only needed columns; drop NA/inf
        cols = [y_col] + x_cols
        dd = d[cols].copy()

        # Replace inf with nan
        dd = dd.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        n_used = int(dd.shape[0])
        if n_used < 5:
            raise ValueError(f"{model_name}: too few complete cases after cleaning (N={n_used}).")

        y = dd[y_col].astype(float)
        X = dd[x_cols].astype(float)

        # Standardize y and all X columns to obtain standardized coefficients directly
        y_z = zscore(y)
        X_z = X.apply(zscore, axis=0)

        # Drop any rows that became NaN due to zero-variance columns
        allmat = pd.concat([y_z, X_z], axis=1).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        y_z = allmat.iloc[:, 0]
        X_z = allmat.iloc[:, 1:]

        # Drop zero-variance predictors (all NaN would have been removed; here handle constant columns)
        keep_cols = [c for c in X_z.columns if np.isfinite(X_z[c]).any()]
        X_z = X_z[keep_cols]

        # Add intercept for standardized regression (intercept not interpreted as standardized)
        X_design = sm.add_constant(X_z, has_constant="add")

        if X_design.shape[0] == 0 or X_design.shape[1] == 0:
            raise ValueError(f"{model_name}: design matrix is empty after cleaning.")

        model = sm.OLS(y_z, X_design).fit()

        # Build result table (standardized betas are coefficients on standardized predictors)
        res = pd.DataFrame(
            {
                "beta": model.params,
                "se": model.bse,
                "t": model.tvalues,
                "p": model.pvalues,
            }
        )
        res.index.name = "term"

        fit = pd.DataFrame(
            {
                "N": [int(model.nobs)],
                "R2": [float(model.rsquared)],
                "Adj_R2": [float(model.rsquared_adj)],
                "DF_model": [int(model.df_model)],
                "DF_resid": [int(model.df_resid)],
            },
            index=[model_name],
        )

        return model, res, fit

    # --- Load & basic filter ---
    d = read_data(data_source)

    for required in ["year", "id"]:
        if required not in d.columns:
            raise ValueError(f"Required column missing: {required}")

    d = d.loc[to_numeric_series(d["year"]) == 1993].copy()

    # --- Construct dependent variables ---
    minority6 = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12 = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
               "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    d["dislike_minority_genres"] = build_count_dislikes(d, minority6)
    d["dislike_other12_genres"] = build_count_dislikes(d, other12)

    # --- Construct racism score (0-5) ---
    # rac1 = 1 if RACHAF==1; 0 if ==2
    d["rac1"] = dichotomous_from_12(d["rachaf"] if "rachaf" in d.columns else np.nan, one_values=[1], zero_values=[2])
    # rac2 = 1 if BUSING==2; 0 if ==1
    d["rac2"] = dichotomous_from_12(d["busing"] if "busing" in d.columns else np.nan, one_values=[2], zero_values=[1])
    # rac3 = 1 if RACDIF1==2; 0 if ==1
    d["rac3"] = dichotomous_from_12(d["racdif1"] if "racdif1" in d.columns else np.nan, one_values=[2], zero_values=[1])
    # rac4 = 1 if RACDIF3==2; 0 if ==1
    d["rac4"] = dichotomous_from_12(d["racdif3"] if "racdif3" in d.columns else np.nan, one_values=[2], zero_values=[1])
    # rac5 = 1 if RACDIF4==1; 0 if ==2
    d["rac5"] = dichotomous_from_12(d["racdif4"] if "racdif4" in d.columns else np.nan, one_values=[1], zero_values=[2])

    rac_components = ["rac1", "rac2", "rac3", "rac4", "rac5"]
    d["racism_score"] = d[rac_components].sum(axis=1, min_count=len(rac_components))

    # --- Controls ---
    d["education_years"] = to_numeric_series(d["educ"]) if "educ" in d.columns else np.nan
    d["occ_prestige"] = to_numeric_series(d["prestg80"]) if "prestg80" in d.columns else np.nan

    realinc = to_numeric_series(d["realinc"]) if "realinc" in d.columns else pd.Series(np.nan, index=d.index)
    hompop = to_numeric_series(d["hompop"]) if "hompop" in d.columns else pd.Series(np.nan, index=d.index)
    # avoid divide-by-zero / invalid
    hompop_valid = hompop.where((hompop > 0) & np.isfinite(hompop), np.nan)
    d["hh_income_per_capita"] = realinc / hompop_valid

    sex = to_numeric_series(d["sex"]) if "sex" in d.columns else pd.Series(np.nan, index=d.index)
    sex = na_if_outside(sex, {1, 2})
    d["female"] = pd.Series(np.nan, index=d.index, dtype="float64")
    d.loc[sex == 2, "female"] = 1.0
    d.loc[sex == 1, "female"] = 0.0

    d["age"] = to_numeric_series(d["age"]) if "age" in d.columns else np.nan

    race = to_numeric_series(d["race"]) if "race" in d.columns else pd.Series(np.nan, index=d.index)
    race = na_if_outside(race, {1, 2, 3})
    d["black"] = pd.Series(np.nan, index=d.index, dtype="float64")
    d.loc[race.isin([1, 3]), "black"] = 0.0
    d.loc[race == 2, "black"] = 1.0

    d["other_race"] = pd.Series(np.nan, index=d.index, dtype="float64")
    d.loc[race.isin([1, 2]), "other_race"] = 0.0
    d.loc[race == 3, "other_race"] = 1.0

    # Hispanic not available in provided variables -> use missing column so it drops out cleanly
    d["hispanic"] = np.nan

    relig = to_numeric_series(d["relig"]) if "relig" in d.columns else pd.Series(np.nan, index=d.index)
    denom = to_numeric_series(d["denom"]) if "denom" in d.columns else pd.Series(np.nan, index=d.index)

    # cons_protestant: RELIG==1 and DENOM in {1,6,7}
    d["cons_protestant"] = pd.Series(np.nan, index=d.index, dtype="float64")
    valid_relig = na_if_outside(relig, {1, 2, 3, 4, 5})
    valid_denom = na_if_outside(denom, {1, 2, 3, 4, 5, 6, 7})
    mask_known = valid_relig.notna() & valid_denom.notna()
    d.loc[mask_known, "cons_protestant"] = 0.0
    d.loc[mask_known & (valid_relig == 1) & (valid_denom.isin([1, 6, 7])), "cons_protestant"] = 1.0

    d["no_religion"] = pd.Series(np.nan, index=d.index, dtype="float64")
    d.loc[valid_relig.notna(), "no_religion"] = 0.0
    d.loc[valid_relig == 4, "no_religion"] = 1.0

    region = to_numeric_series(d["region"]) if "region" in d.columns else pd.Series(np.nan, index=d.index)
    region = na_if_outside(region, {1, 2, 3, 4})
    d["south"] = pd.Series(np.nan, index=d.index, dtype="float64")
    d.loc[region.notna(), "south"] = 0.0
    d.loc[region == 3, "south"] = 1.0

    # --- Model specs ---
    # Drop 'hispanic' from RHS because it is unavailable (all missing) -> prevents zero-size after dropna
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age",
        "black",
        # "hispanic",  # not available
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    outputs = {}

    def run_one(dv_col, model_name):
        # Use only rows where DV is observed; and needed X columns are observed
        model, tab, fit = fit_standardized_ols(d, dv_col, x_cols, model_name)

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Coefficients are from regression where DV and all predictors were z-scored (standardized betas).\n")
            f.write("- 'hispanic' dummy is omitted because it is not available in the provided extract.\n")

        tab_rounded = tab.copy()
        for c in ["beta", "se", "t", "p"]:
            tab_rounded[c] = pd.to_numeric(tab_rounded[c], errors="coerce")
        tab_rounded = tab_rounded.round({"beta": 3, "se": 3, "t": 3, "p": 4})

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_rounded.to_string())
            f.write("\n\n")
            f.write(fit.to_string())

        return tab_rounded, fit

    tabA, fitA = run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    tabB, fitB = run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    outputs["ModelA_table"] = tabA
    outputs["ModelA_fit"] = fitA
    outputs["ModelB_table"] = tabB
    outputs["ModelB_fit"] = fitB

    # Combined quick view
    combined = (
        tabA[["beta", "p"]].rename(columns={"beta": "A_beta", "p": "A_p"})
        .join(tabB[["beta", "p"]].rename(columns={"beta": "B_beta", "p": "B_p"}), how="outer")
    )
    outputs["Combined"] = combined

    with open("./output/Table2_combined_key_results.txt", "w", encoding="utf-8") as f:
        f.write(combined.round(4).to_string())

    return outputs