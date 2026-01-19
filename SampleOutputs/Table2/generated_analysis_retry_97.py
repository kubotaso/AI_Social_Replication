def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative NA-code cleaning for this extract:
        - Coerce to numeric
        - Treat common GSS NA sentinels as missing
        """
        x = to_num(x).copy()
        na_vals = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(na_vals)))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5. Dislike if 4 or 5.
        Missing if outside 1-5 or NA-coded.
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

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count(df_, items, require_complete=True):
        mat = pd.concat([likert_dislike_indicator(df_[c]).rename(c) for c in items], axis=1)
        if require_complete:
            # Complete-case on DV components (paper: DK treated as missing; missing cases excluded)
            return mat.sum(axis=1, min_count=len(items))
        # Alternative: allow partial (not used here)
        return mat.sum(axis=1, min_count=1)

    def stars_from_p(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardized_ols(df_, dv, xcols, model_name, ordered_labels):
        """
        Run OLS on z-scored y and z-scored X (including dummies) to obtain standardized betas.
        Intercept is from unstandardized regression on original scales.
        Enforces that all requested predictors are present and non-constant in the estimation sample.
        """
        needed = [dv] + xcols
        d = df_[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        # Guard: require enough cases
        if d.shape[0] < len(xcols) + 5:
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        # Guard: ensure no predictor is constant in estimation sample (prevents runtime errors / silent drops)
        zero_var = []
        for c in xcols:
            v = d[c].astype(float)
            if v.nunique(dropna=True) <= 1:
                zero_var.append(c)
        if zero_var:
            raise ValueError(f"{model_name}: zero-variance predictors in estimation sample: {zero_var}")

        # Unstandardized model (for intercept and optional diagnostics)
        y_u = d[dv].astype(float)
        X_u = sm.add_constant(d[xcols].astype(float), has_constant="add")
        model_u = sm.OLS(y_u, X_u).fit()

        # Standardized model for betas
        y_z = zscore(d[dv].astype(float))
        Xz = pd.DataFrame({c: zscore(d[c].astype(float)) for c in xcols}, index=d.index)

        # After zscoring, ensure no NaNs introduced (would indicate zero variance)
        bad = [c for c in xcols if Xz[c].isna().any() or Xz[c].std(ddof=0) == 0]
        if bad:
            raise ValueError(f"{model_name}: predictors became undefined after standardization: {bad}")

        X_z = sm.add_constant(Xz, has_constant="add")
        model_z = sm.OLS(y_z, X_z).fit()

        # Build paper-style table (standardized betas + stars); keep intercept separately (unstandardized)
        out_rows = []
        # Constant shown like paper, but not standardized
        out_rows.append(
            {
                "term": "Constant",
                "variable": "Constant",
                "beta": float(model_u.params["const"]),
                "stars": stars_from_p(float(model_u.pvalues["const"])) if np.isfinite(model_u.pvalues["const"]) else "",
            }
        )
        for var, label in ordered_labels:
            out_rows.append(
                {
                    "term": var,
                    "variable": label,
                    "beta": float(model_z.params[var]),
                    "stars": stars_from_p(float(model_z.pvalues[var])) if np.isfinite(model_z.pvalues[var]) else "",
                }
            )

        tab = pd.DataFrame(out_rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model_z.nobs),
                    "k_including_const": int(model_z.df_model + 1),
                    "r2": float(model_z.rsquared),
                    "adj_r2": float(model_z.rsquared_adj),
                    "intercept_unstandardized": float(model_u.params["const"]),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary_unstandardized.txt", "w", encoding="utf-8") as f:
            f.write(model_u.summary().as_text())

        with open(f"./output/{model_name}_summary_standardized.txt", "w", encoding="utf-8") as f:
            f.write(model_z.summary().as_text())

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("Standardized OLS coefficients (beta) for predictors; Constant is unstandardized.\n\n")
            f.write(tab.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

        return tab, fit, model_u, model_z

    # -------------------------
    # Filter to 1993
    # -------------------------
    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count(df, minority_items, require_complete=True)
    df["dislike_other12_genres"] = build_count(df, other12_items, require_complete=True)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
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

    # -------------------------
    # Controls
    # -------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing realinc and/or hompop columns.")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race indicators
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not available in provided data -> cannot be constructed faithfully
    # To keep model spec explicit and avoid silent omission, we stop with a clear error.
    if "hispanic" not in df.columns:
        # Do not proxy using 'ethnic' per instruction.
        df["hispanic"] = np.nan

    # Religion
    if "relig" not in df.columns:
        raise ValueError("Missing relig column.")
    relig = clean_na_codes(df["relig"])

    if "denom" not in df.columns:
        raise ValueError("Missing denom column.")
    denom = clean_na_codes(df["denom"])

    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Models (Table 2 spec order)
    # -------------------------
    xcols = [
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

    ordered_labels = [
        ("racism_score", "Racism score (0–5)"),
        ("education_years", "Education (years)"),
        ("hh_income_per_capita", "Household income per capita"),
        ("occ_prestige", "Occupational prestige"),
        ("female", "Female"),
        ("age_years", "Age"),
        ("black", "Black"),
        ("hispanic", "Hispanic"),
        ("other_race", "Other race"),
        ("cons_protestant", "Conservative Protestant"),
        ("no_religion", "No religion"),
        ("south", "Southern"),
    ]

    # Validate presence
    missing_constructed = [c for c in xcols + ["dislike_minority_genres", "dislike_other12_genres"] if c not in df.columns]
    if missing_constructed:
        raise ValueError(f"Missing constructed columns: {missing_constructed}")

    # Fit models (will raise if any predictor is all-missing or constant in the estimation sample)
    tabA, fitA, modelA_u, modelA_z = standardized_ols(
        df,
        dv="dislike_minority_genres",
        xcols=xcols,
        model_name="Table2_ModelA_dislike_minority6",
        ordered_labels=ordered_labels,
    )
    tabB, fitB, modelB_u, modelB_z = standardized_ols(
        df,
        dv="dislike_other12_genres",
        xcols=xcols,
        model_name="Table2_ModelB_dislike_other12",
        ordered_labels=ordered_labels,
    )

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from provided microdata)\n")
        f.write("Standardized OLS coefficients reported for predictors; constant is unstandardized.\n")
        f.write("NOTE: Provided extract does not include a direct Hispanic identifier; 'hispanic' is all-missing and will cause estimation failure unless present.\n\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n\nModel B DV: dislike_other12_genres (other 12 genres)\n")
        f.write(fitB.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }