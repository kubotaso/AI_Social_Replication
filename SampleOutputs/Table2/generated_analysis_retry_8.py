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

    def clean_gss_numeric(x):
        """
        Conservative missing-code handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        """
        x = to_num(x).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 where 4/5 = dislike.
        1/2/3 = not dislike.
        Anything else -> missing.
        """
        x = clean_gss_numeric(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_numeric(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Count of dislikes across items.
        Mirror "DK treated as missing and missing cases excluded" by requiring all items observed.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def fit_ols_with_standardized_betas(df_in, dv, xcols, model_name):
        """
        Fit OLS on unstandardized DV and predictors (with intercept),
        then compute standardized betas post-hoc:
            beta_j = b_j * sd(x_j) / sd(y)
        This preserves a meaningful intercept while producing standardized coefficients.
        """
        needed = [dv] + xcols
        d = df_in[needed].copy()

        # Ensure numeric and remove inf
        for c in needed:
            d[c] = to_num(d[c])
        d = d.replace([np.inf, -np.inf], np.nan)

        # Listwise deletion across model variables
        d = d.dropna(axis=0, how="any").copy()

        # Guard
        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: insufficient complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)
        X = d[xcols].astype(float)

        # Drop zero-variance predictors (can arise if a dummy never varies in sample)
        nunique = X.nunique(dropna=True)
        keep = [c for c in xcols if nunique.get(c, 0) > 1]
        dropped = [c for c in xcols if c not in keep]
        X = X[keep]

        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors are constant after cleaning; cannot fit.")

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (exclude intercept)
        y_sd = float(y.std(ddof=0))
        if not np.isfinite(y_sd) or y_sd == 0:
            raise ValueError(f"{model_name}: DV has zero/invalid variance; cannot standardize betas.")

        beta = {}
        for c in X.columns:
            x_sd = float(X[c].std(ddof=0))
            if not np.isfinite(x_sd) or x_sd == 0:
                beta[c] = np.nan
            else:
                beta[c] = float(model.params[c]) * x_sd / y_sd

        # Build output table with labeled rows, including constant
        rows = ["const"] + list(X.columns)
        out = pd.DataFrame(index=rows)
        out.index.name = "term"
        out["b_unstd"] = model.params.reindex(rows).astype(float)
        out["beta_std"] = pd.Series([np.nan] + [beta[c] for c in X.columns], index=rows, dtype="float64")

        # Stars based on computed p-values (table in paper shows stars but not SEs)
        p = model.pvalues.reindex(rows).astype(float)

        def stars(pv):
            if not np.isfinite(pv):
                return ""
            if pv < 0.001:
                return "***"
            if pv < 0.01:
                return "**"
            if pv < 0.05:
                return "*"
            return ""

        out["p_value"] = p
        out["sig"] = p.map(stars)

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "n": int(model.nobs),
                "k_predictors": int(model.df_model),  # excludes intercept
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "dropped_constant_predictors": ", ".join(dropped) if dropped else ""
            }]
        )

        return model, out, fit, d.index

    # -----------------------------
    # Load / filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs (counts)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music taste item column: {c}")

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
    if "educ" not in df.columns:
        raise ValueError("Missing education column: educ")
    educ = clean_gss_numeric(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing income-per-capita component column: {c}")
    realinc = clean_gss_numeric(df["realinc"])
    hompop = clean_gss_numeric(df["hompop"]).where(clean_gss_numeric(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing occupational prestige column: prestg80")
    df["occ_prestige"] = clean_gss_numeric(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing sex column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing age column: age")
    age = clean_gss_numeric(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing race column: race")
    race = clean_gss_numeric(df["race"]).where(clean_gss_numeric(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic flag not present in provided extract: create but do NOT include in models
    df["hispanic"] = np.nan

    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_numeric(df["relig"])
    denom = clean_gss_numeric(df["denom"])
    consprot = np.where((relig.isna() | denom.isna()), np.nan,
                        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float))
    df["cons_protestant"] = consprot

    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    if "region" not in df.columns:
        raise ValueError("Missing region column: region")
    region = clean_gss_numeric(df["region"]).where(clean_gss_numeric(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Models (Table 2 RHS as available; Hispanic omitted due to missing field)
    # -----------------------------
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

    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Constructed predictor missing unexpectedly: {c}")

    results = {}

    def write_outputs(model, table_df, fit_df, model_name):
        # Model summary
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit_df.to_string(index=False))
            f.write("\n\nNotes:\n")
            f.write("- Standardized betas computed post-hoc: beta = b * sd(x)/sd(y)\n")
            f.write("- Stars based on computed two-tailed p-values from this fitted model\n")

        # Human-readable regression table
        tab_to_print = table_df.copy()
        tab_to_print["beta_std_with_sig"] = tab_to_print["beta_std"].map(
            lambda v: "" if not np.isfinite(v) else f"{v: .6f}"
        ) + tab_to_print["sig"].astype(str)
        tab_to_print = tab_to_print[["b_unstd", "beta_std", "beta_std_with_sig", "p_value", "sig"]]

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_to_print.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

    # Model A
    mA, tabA, fitA, idxA = fit_ols_with_standardized_betas(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    write_outputs(mA, tabA, fitA, "Table2_ModelA_dislike_minority6")

    # Model B
    mB, tabB, fitB, idxB = fit_ols_with_standardized_betas(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )
    write_outputs(mB, tabB, fitB, "Table2_ModelB_dislike_other12")

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS extract)\n")
        f.write("Two OLS models on dislike counts; standardized betas computed post-hoc.\n")
        f.write("Important: Hispanic dummy cannot be included because no direct Hispanic identifier is present in the provided extract.\n\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin) [count]\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres (12 remaining genres) [count]\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    return results