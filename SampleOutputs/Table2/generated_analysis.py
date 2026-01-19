def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    # Normalize column names to lower-case to match provided extracts
    df.columns = [c.lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Expected column 'year' not found.")
    df = df.loc[df["year"] == 1993].copy()

    # Helper: coerce to numeric
    def num(s):
        return pd.to_numeric(s, errors="coerce")

    # --- Construct DVs: dislike counts (item-level missing propagates to DV by requiring all items observed) ---
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Expected music item column '{c}' not found.")

    # Dislike indicator: 1 if {4,5}, 0 if {1,2,3}, else missing
    def dislike_indicator(x):
        x = num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    for c in minority_items:
        df[f"d_{c}"] = dislike_indicator(df[c])
    for c in other12_items:
        df[f"d_{c}"] = dislike_indicator(df[c])

    # Require all component items non-missing, then sum
    df["dislike_minority_genres"] = df[[f"d_{c}" for c in minority_items]].sum(axis=1, min_count=len(minority_items))
    df["dislike_other12_genres"] = df[[f"d_{c}" for c in other12_items]].sum(axis=1, min_count=len(other12_items))

    # --- Racism score (0-5 additive) ---
    required_rac_cols = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in required_rac_cols:
        if c not in df.columns:
            raise ValueError(f"Expected racism component column '{c}' not found.")

    rachaf = num(df["rachaf"])
    busing = num(df["busing"])
    racdif1 = num(df["racdif1"])
    racdif3 = num(df["racdif3"])
    racdif4 = num(df["racdif4"])

    rac1 = pd.Series(np.nan, index=df.index)
    rac1.loc[rachaf == 1] = 1
    rac1.loc[rachaf == 2] = 0

    rac2 = pd.Series(np.nan, index=df.index)
    rac2.loc[busing == 2] = 1
    rac2.loc[busing == 1] = 0

    rac3 = pd.Series(np.nan, index=df.index)
    rac3.loc[racdif1 == 2] = 1
    rac3.loc[racdif1 == 1] = 0

    rac4 = pd.Series(np.nan, index=df.index)
    rac4.loc[racdif3 == 2] = 1
    rac4.loc[racdif3 == 1] = 0

    rac5 = pd.Series(np.nan, index=df.index)
    rac5.loc[racdif4 == 1] = 1
    rac5.loc[racdif4 == 2] = 0

    racism_components = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    racism_components.columns = ["rac1", "rac2", "rac3", "rac4", "rac5"]
    df["racism_score"] = racism_components.sum(axis=1, min_count=5)

    # --- Controls ---
    df["education_years"] = num(df.get("educ"))
    df["realinc"] = num(df.get("realinc"))
    df["hompop"] = num(df.get("hompop"))
    df["occ_prestige"] = num(df.get("prestg80"))

    # Income per capita
    df["hh_income_per_capita"] = np.where(
        (df["realinc"].notna()) & (df["hompop"].notna()) & (df["hompop"] > 0),
        df["realinc"] / df["hompop"],
        np.nan,
    )

    # Female
    sex = num(df.get("sex"))
    df["female"] = np.where(sex == 2, 1.0, np.where(sex == 1, 0.0, np.nan))

    # Age
    df["age"] = num(df.get("age"))

    # Race indicators (white is reference)
    race = num(df.get("race"))
    df["black"] = np.where(race == 2, 1.0, np.where(race.isin([1, 3]), 0.0, np.nan))
    df["other_race"] = np.where(race == 3, 1.0, np.where(race.isin([1, 2]), 0.0, np.nan))

    # Hispanic not available in provided variables -> omit from model (cannot construct faithfully)
    # Still keep a placeholder for reporting clarity
    df["hispanic"] = np.nan

    # Conservative Protestant approximation from RELIG and DENOM (per mapping instruction)
    relig = num(df.get("relig"))
    denom = num(df.get("denom"))
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = relig.notna() & denom.notna()
    consprot.loc[valid] = 0.0
    consprot.loc[valid & (relig == 1) & (denom.isin([1, 6, 7]))] = 1.0
    # If RELIG known but DENOM missing, cannot classify -> missing
    consprot.loc[relig.notna() & denom.isna()] = np.nan
    # If RELIG not Protestant, denom irrelevant; if both known, classification already 0.
    df["cons_protestant"] = consprot

    # No religion
    df["no_religion"] = np.where(relig == 4, 1.0, np.where(relig.notna(), 0.0, np.nan))

    # South
    region = num(df.get("region"))
    df["south"] = np.where(region == 3, 1.0, np.where(region.isin([1, 2, 4]), 0.0, np.nan))

    # --- Standardization ---
    # For "standardized coefficients (beta)" we z-score DV and all predictors (including dummies) in-model.
    def zscore(series):
        s = series.astype(float)
        m = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if pd.isna(sd) or sd == 0:
            return s * np.nan
        return (s - m) / sd

    # Model specs (omit hispanic due to unavailability)
    predictors = [
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

    def fit_model(dv_name):
        needed = [dv_name] + predictors
        d = df[needed].copy()

        # Listwise deletion
        d = d.dropna(axis=0, how="any").copy()

        # Standardize
        dz = d.apply(zscore)

        y = dz[dv_name]
        X = dz[predictors]
        X = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X).fit()

        # Build a compact table similar to a regression table
        out = pd.DataFrame(
            {
                "beta_std": model.params,
                "se": model.bse,
                "t": model.tvalues,
                "p": model.pvalues,
            }
        )
        out.index.name = "term"
        fit = {
            "dv": dv_name,
            "n": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        }
        return model, out, fit, d

    results = {}
    text_blocks = []

    for dv in ["dislike_minority_genres", "dislike_other12_genres"]:
        model, table, fit, d_used = fit_model(dv)

        # Save human-readable text
        title = f"OLS with standardized variables (beta): DV={dv}"
        text_blocks.append(title)
        text_blocks.append("=" * len(title))
        text_blocks.append(f"N={fit['n']}, R2={fit['r2']:.4f}, Adj.R2={fit['adj_r2']:.4f}")
        text_blocks.append("")
        text_blocks.append(table.to_string(float_format=lambda x: f"{x: .4f}"))
        text_blocks.append("")
        text_blocks.append(model.summary().as_text())
        text_blocks.append("\n\n")

        results[dv] = {
            "coef_table": table,
            "fit": pd.DataFrame([fit]),
        }

        table.to_csv(f"./output/table2_like_{dv}_coef_table.csv", index=True)
        with open(f"./output/table2_like_{dv}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

    with open("./output/table2_like_combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text_blocks))

    # Return a dict of DataFrames for programmatic access
    return {
        "minority_genres_coef": results["dislike_minority_genres"]["coef_table"],
        "minority_genres_fit": results["dislike_minority_genres"]["fit"],
        "other12_genres_coef": results["dislike_other12_genres"]["coef_table"],
        "other12_genres_fit": results["dislike_other12_genres"]["fit"],
    }