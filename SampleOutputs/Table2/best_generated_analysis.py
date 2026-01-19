def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def _clean_na_codes(series):
        """
        GSS-style numeric fields sometimes use special high codes (e.g., 8/9, 98/99, 998/999).
        We only have a subset extract here, so apply a conservative rule:
        - treat <=0 as missing for variables that should be positive (handled elsewhere),
        - treat very large sentinel-like values as missing.
        """
        x = _to_num(series).copy()
        # common sentinel bands in GSS extracts; keep conservative to avoid nuking valid data
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def _likert_dislike_indicator(item_series):
        """
        Music taste items: 1-5 scale where 4/5 indicate dislike.
        Missing if outside 1-5 or NA-coded.
        """
        x = _clean_na_codes(item_series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def _binary_from_codes(series, true_codes, false_codes):
        x = _clean_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def _zscore(s):
        s = _to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return s * np.nan
        return (s - mu) / sd

    def _build_count(df, items):
        """
        Sum of dislike indicators. To mirror 'DK treated as missing and missing cases excluded',
        require all component items observed (complete-case) for the DV.
        """
        cols = []
        for c in items:
            cols.append(_likert_dislike_indicator(df[c]).rename(c + "_dislike"))
        mat = pd.concat(cols, axis=1)
        count = mat.sum(axis=1, min_count=len(items))
        return count

    def _standardized_ols(df, dv, xcols, model_name):
        """
        Standardize y and each x (including dummies) over the estimation sample,
        then run OLS with intercept. Returns model, coefficient table, fit dict, and used index.
        """
        needed = [dv] + xcols
        d = df[needed].copy()

        # Replace inf with nan
        d = d.replace([np.inf, -np.inf], np.nan)

        # Complete-case for this model
        d = d.dropna(axis=0, how="any")
        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = _zscore(d[dv])
        Xz = {}
        for c in xcols:
            Xz[c] = _zscore(d[c])
        X = pd.DataFrame(Xz, index=d.index)

        # After z-scoring, any constant columns will become all-NaN; drop them
        keep = [c for c in X.columns if X[c].notna().all() and X[c].std(ddof=0) > 0]
        X = X[keep]
        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors dropped after standardization (likely constants).")

        # Also ensure y is finite
        ok = y.notna() & np.isfinite(y)
        X = X.loc[ok]
        y = y.loc[ok]
        if X.shape[0] < (X.shape[1] + 2):
            raise ValueError(f"{model_name}: not enough cases after standardization cleaning (n={X.shape[0]}, k={X.shape[1]}).")

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        tab = pd.DataFrame(
            {
                "coef_beta": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        fit = {
            "model": model_name,
            "n": int(model.nobs),
            "k": int(model.df_model + 1),  # includes intercept
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        }
        return model, tab, fit, d.index

    # -------------------------
    # Load and basic filtering
    # -------------------------
    df = pd.read_csv(data_source)

    # Normalize column names to lower-case for robustness
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns check
    required_base = ["year", "id"]
    for c in required_base:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    df["year"] = _to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Construct DVs
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = _build_count(df, minority_items)
    df["dislike_other12_genres"] = _build_count(df, other12_items)

    # -------------------------
    # Construct racism score (0-5)
    # -------------------------
    # Required fields
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = _binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to majority-black school
    rac2 = _binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = _binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = _binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = _binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # RHS controls
    # -------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    df["education_years"] = _clean_na_codes(df["educ"]).where(_clean_na_codes(df["educ"]).between(0, 20))

    # HH income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = _clean_na_codes(df["realinc"])
    hompop = _clean_na_codes(df["hompop"])
    hompop = hompop.where(hompop > 0)  # prevent divide by 0/neg
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = _clean_na_codes(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = _binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    df["age_years"] = _clean_na_codes(df["age"]).where(_clean_na_codes(df["age"]).between(18, 89))

    # Race indicators (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = _clean_na_codes(df["race"]).where(_clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: not available in provided variables -> create as all missing so it is excluded from models
    # (Do NOT proxy using ETHNIC per instruction.)
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = _clean_na_codes(df["relig"])
    denom = _clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = _clean_na_codes(df["region"]).where(_clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -------------------------
    # Model fitting (Table 2: two DVs, same RHS)
    # Important: Hispanic dummy not available -> drop from x list to avoid empty design matrix.
    # -------------------------
    x_cols_full = [
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

    # Ensure all exist
    for c in x_cols_full:
        if c not in df.columns:
            raise ValueError(f"Constructed predictor missing unexpectedly: {c}")

    results = {}

    def _run_one(dv_col, model_name):
        model, tab, fit, used_idx = _standardized_ols(df, dv_col, x_cols_full, model_name)
        # Save human-readable text
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")
        tab_out = tab.copy()
        tab_out.index.name = "term"
        tab_out.to_string(open(f"./output/{model_name}_table.txt", "w", encoding="utf-8"), float_format=lambda x: f"{x: .6f}")
        return tab_out, pd.DataFrame([fit])

    tabA, fitA = _run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    tabB, fitB = _run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Combined overview text
    overview_path = "./output/Table2_overview.txt"
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("Table 2 replication (1993 GSS): Standardized OLS (z-scored y and x; intercept included)\n")
        f.write("Note: Hispanic dummy not included because no direct Hispanic identifier is present in provided data.\n\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin) dislike count\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres (12 remaining genres) dislike count\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results