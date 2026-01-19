def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Helpers
    # -----------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_missing(series):
        """
        Conservative NA handling for this provided extract.
        Treat common GSS sentinel codes as missing. Keep everything else.
        """
        x = to_num(series).copy()
        x = x.mask(x.isin([7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999]))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items: 1-5 where 4/5 => dislike(1), 1/2/3 => not-dislike(0).
        Missing if not in 1..5 or NA-coded.
        """
        x = clean_missing(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Count of dislikes across items. To match "DK treated as missing and missing cases excluded",
        require complete data on all component items for the DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        # require all items observed
        count = mat.sum(axis=1, min_count=len(items))
        return count

    def standardize_series(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_ols_with_betas(df, dv, xcols, model_name):
        """
        Fit OLS on unstandardized DV (so intercept is meaningful), with intercept.
        Report:
          - unstandardized coefficients (including intercept)
          - standardized betas for predictors (intercept has no beta)
        Standardized beta computed as: b * sd(x) / sd(y) on the estimation sample.
        """
        d = df[[dv] + xcols].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")
        n = d.shape[0]
        if n < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={n}, k={len(xcols)}).")

        y = to_num(d[dv]).astype(float)
        X = pd.DataFrame({c: to_num(d[c]).astype(float) for c in xcols}, index=d.index)
        X = X.replace([np.inf, -np.inf], np.nan)

        # Ensure finite
        ok = y.notna() & np.isfinite(y)
        for c in X.columns:
            ok = ok & X[c].notna() & np.isfinite(X[c])
        y = y.loc[ok]
        X = X.loc[ok]

        # Drop constant predictors (can cause singularities and also no beta)
        keep = []
        for c in X.columns:
            if X[c].nunique(dropna=True) >= 2 and np.isfinite(X[c].std(ddof=0)) and X[c].std(ddof=0) > 0:
                keep.append(c)
        X = X[keep]

        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: design matrix empty after dropping constant predictors.")

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Compute standardized betas for predictors (not for intercept)
        y_sd = y.std(ddof=0)
        betas = {}
        if not np.isfinite(y_sd) or y_sd == 0:
            for c in X.columns:
                betas[c] = np.nan
        else:
            for c in X.columns:
                x_sd = X[c].std(ddof=0)
                if not np.isfinite(x_sd) or x_sd == 0:
                    betas[c] = np.nan
                else:
                    betas[c] = model.params[c] * (x_sd / y_sd)

        # Assemble table with term labels; include intercept
        terms = list(model.params.index)
        table = pd.DataFrame(
            {
                "b_unstd": model.params,
                "beta_std": [np.nan if t == "const" else betas.get(t, np.nan) for t in terms],
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            },
            index=terms,
        )
        table.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_params": int(len(model.params)),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

        # Human-readable table: include both b and beta; paper reports betas, but we compute both
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(table.to_string(float_format=lambda x: f"{x: .6f}"))

        return model, table, fit, d.index

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year==1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # DVs
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------
    # Racism scale (0-5)
    # -----------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls
    # -----------------------
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(clean_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_missing(df["race"]).where(clean_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic not available in provided variables: keep but cannot estimate if all missing.
    # To keep table structure faithful, we attempt to include it only if it has any non-missing.
    df["hispanic"] = np.nan

    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_missing(df["relig"])
    denom = clean_missing(df["denom"])

    consprot = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )
    df["cons_protestant"] = consprot

    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_missing(df["region"]).where(clean_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Fit models (Table 2)
    # -----------------------
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

    # If 'hispanic' has no data, drop it to avoid wiping out the sample.
    if df["hispanic"].notna().sum() == 0:
        x_cols_used = [c for c in x_cols if c != "hispanic"]
    else:
        x_cols_used = x_cols

    results = {}

    mA, tabA, fitA, idxA = fit_ols_with_betas(
        df, "dislike_minority_genres", x_cols_used, "Table2_ModelA_dislike_minority6"
    )
    mB, tabB, fitB, idxB = fit_ols_with_betas(
        df, "dislike_other12_genres", x_cols_used, "Table2_ModelB_dislike_other12"
    )

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (1993 GSS): OLS with standardized betas (computed from unstandardized OLS)\n")
        f.write("Notes:\n")
        f.write("- Model is estimated on unstandardized DV; standardized beta computed as b * sd(x)/sd(y) on estimation sample.\n")
        if "hispanic" not in x_cols_used:
            f.write("- Hispanic dummy not estimated because no Hispanic identifier is available in the provided extract (column all-missing).\n")
        f.write("\nModel A: DV = count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B: DV = count of dislikes among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results