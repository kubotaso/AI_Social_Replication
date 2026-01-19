def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def is_gss_missing(x):
        """
        Conservative missing-code handler for this extract:
        - keep only sensible ranges per-variable (handled elsewhere)
        - additionally treat common GSS sentinels as missing when they appear
        """
        if pd.isna(x):
            return True
        try:
            v = float(x)
        except Exception:
            return True
        if v in (8, 9, 98, 99, 998, 999, 9998, 9999):
            return True
        return False

    def clean_series_numeric(s):
        x = to_num(s).copy()
        # mark common sentinel codes as missing
        x = x.mask(x.apply(is_gss_missing))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items: 1-5, where 4/5 are dislike.
        Missing if not in 1..5 or sentinel.
        """
        x = clean_series_numeric(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_series_numeric(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        # Require complete data on all component items (DK treated as missing -> listwise exclusion for DV)
        return mat.sum(axis=1, min_count=len(items))

    def zscore_sample(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def standardized_ols(df, y_col, x_cols, model_name):
        """
        Standardized betas via: beta_j = b_j * sd(x_j) / sd(y), estimated on the SAME
        estimation sample (listwise on y + all x). Constant reported as unstandardized intercept.
        """
        needed = [y_col] + x_cols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < len(x_cols) + 5:
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={d.shape[0]}).")

        # Check zero-variance predictors in estimation sample (must not silently drop table variables)
        zero_var = []
        for c in x_cols:
            sd = d[c].std(ddof=0)
            if (not np.isfinite(sd)) or sd == 0:
                zero_var.append(c)
        if zero_var:
            raise ValueError(f"{model_name}: zero-variance predictors in estimation sample: {zero_var}")

        y = d[y_col].astype(float)
        X = d[x_cols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        sd_y = y.std(ddof=0)
        beta = {}
        for c in x_cols:
            beta[c] = model.params[c] * (d[c].std(ddof=0) / sd_y)

        # Assemble Table-2-like output: standardized betas for predictors + unstandardized constant
        tab = pd.DataFrame(
            {
                "term": ["const"] + x_cols,
                "coef": [model.params["const"]] + [beta[c] for c in x_cols],
            }
        ).set_index("term")

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "n": int(model.nobs),
                "k_predictors": int(model.df_model),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
            }]
        )

        # Save summaries/tables
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: coef table below reports standardized betas for predictors and the unstandardized intercept.\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(tab.to_string(float_format=lambda v: f"{v: .6f}"))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
            f.write("\n")

        return tab, fit, model

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
    # Dependent variables
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items)

    # -------------------------
    # Racism score (0-5) additive index
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism field: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half Black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' (EDUC).")
    educ = clean_series_numeric(df["educ"]).where(clean_series_numeric(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' for income per capita.")
    realinc = clean_series_numeric(df["realinc"])
    hompop = clean_series_numeric(df["hompop"]).where(clean_series_numeric(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige (PRESTG80)
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' (PRESTG80).")
    df["occ_prestige"] = clean_series_numeric(df["prestg80"])

    # Female (SEX: 1=male, 2=female)
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' (SEX).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' (AGE).")
    df["age_years"] = clean_series_numeric(df["age"]).where(clean_series_numeric(df["age"]).between(18, 89))

    # Race dummies (RACE: 1=white, 2=black, 3=other)
    if "race" not in df.columns:
        raise ValueError("Missing 'race' (RACE).")
    race = clean_series_numeric(df["race"]).where(clean_series_numeric(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: not present in provided extract (do NOT proxy using ETHNIC)
    # To keep the model runnable and faithful to available data, set to 0 (unknown/absent),
    # and document in output. (Setting to NaN would collapse N to 0 under listwise deletion.)
    df["hispanic"] = 0.0

    # Religion variables
    if "relig" not in df.columns:
        raise ValueError("Missing 'relig' (RELIG).")
    if "denom" not in df.columns:
        raise ValueError("Missing 'denom' (DENOM).")
    relig = clean_series_numeric(df["relig"])
    denom = clean_series_numeric(df["denom"])

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing 'region' (REGION).")
    region = clean_series_numeric(df["region"]).where(clean_series_numeric(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Models (Table 2 RHS list; include Hispanic as available placeholder)
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

    # Run models
    tabA, fitA, modelA = standardized_ols(df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6")
    tabB, fitB, modelB = standardized_ols(df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12")

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993 extract)\n")
        f.write("Coefficients: standardized betas for predictors; intercept is unstandardized.\n")
        f.write("Important: 'hispanic' is not available in the provided extract; it is set to 0 for all cases.\n")
        f.write("SEs/p-values are not part of the Table-2-style coefficient table (but OLS summaries are saved separately).\n\n")
        f.write("Model A DV: count of disliked genres among {rap, reggae, blues, jazz, gospel, latin}\n")
        f.write(fitA.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        f.write("\n\n")
        f.write("Model B DV: count of disliked genres among the other 12 genres in the 18-item battery\n")
        f.write(fitB.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }