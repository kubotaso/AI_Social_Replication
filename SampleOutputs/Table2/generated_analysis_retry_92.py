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

    def clean_na_codes(x):
        """
        Conservative NA-code cleaning for this extract:
        - keep only plausible ranges where we know them (handled per-variable)
        - treat common GSS NA codes as missing
        """
        x = to_num(x).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(item_series):
        """
        Music taste items: 1-5 scale.
        Dislike = 4 or 5
        Not-dislike = 1,2,3
        Missing if outside 1-5 or NA-coded.
        """
        x = clean_na_codes(item_series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_dv(df, items):
        """
        DV is a count of dislikes. Following the provided instructions:
        - item-level NA codes treated as missing
        - then use complete-case on all component items for the DV
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        dv = mat.sum(axis=1, min_count=len(items))
        return dv

    def standardized_betas_from_unstandardized(y, X, b_unstd):
        """
        Beta_j = b_j * sd(X_j) / sd(y)
        Intercept is not standardized; we'll keep it as the unstandardized intercept.
        Uses population sd (ddof=0) to align with typical "beta" computations.
        """
        y_sd = y.std(ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = X[col].std(ddof=0)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = b_unstd[col] * (x_sd / y_sd)
        return betas

    def stars_from_p(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_model(df, dv_col, x_cols, model_name, var_order_for_output):
        """
        OLS on unstandardized variables (count DV), then compute standardized betas
        via SD ratio. Listwise deletion on exactly DV + x_cols.
        """
        needed = [dv_col] + x_cols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Ensure we truly have variation in DV and predictors
        if d.shape[0] < (len(x_cols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}).")

        y = to_num(d[dv_col])
        X = d[x_cols].apply(to_num)
        X = sm.add_constant(X, has_constant="add")

        # Drop any zero-variance predictors (prevents weird NA rows / singularities)
        drop_cols = []
        for c in X.columns:
            if c == "const":
                continue
            sd = X[c].std(ddof=0)
            if (not np.isfinite(sd)) or sd == 0:
                drop_cols.append(c)
        if drop_cols:
            X = X.drop(columns=drop_cols)
            x_cols_effective = [c for c in x_cols if c not in drop_cols]
        else:
            x_cols_effective = list(x_cols)

        model = sm.OLS(y, X).fit()

        # Standardized betas for the predictors that remain
        betas = standardized_betas_from_unstandardized(y, X, model.params)

        # Build Table-2-like output: standardized betas + stars; plus intercept as unstandardized
        rows = []
        # Intercept first (unstandardized)
        rows.append(
            {
                "term": "Intercept",
                "beta": np.nan,
                "b_unstd": float(model.params.get("const", np.nan)),
                "p_value": float(model.pvalues.get("const", np.nan)),
                "stars": stars_from_p(float(model.pvalues.get("const", np.nan))),
            }
        )

        # Then predictors in requested order, skipping dropped predictors
        for v in var_order_for_output:
            if v not in x_cols_effective:
                # If dropped due to zero-variance or not included, mark explicitly
                rows.append(
                    {
                        "term": v,
                        "beta": np.nan,
                        "b_unstd": np.nan,
                        "p_value": np.nan,
                        "stars": "omitted",
                    }
                )
                continue
            rows.append(
                {
                    "term": v,
                    "beta": float(betas.get(v, np.nan)),
                    "b_unstd": float(model.params.get(v, np.nan)),
                    "p_value": float(model.pvalues.get(v, np.nan)),
                    "stars": stars_from_p(float(model.pvalues.get(v, np.nan))),
                }
            )

        table = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE:\n")
            f.write("- 'beta' column computed as standardized OLS coefficients via SD-ratio from the unstandardized OLS fit.\n")
            f.write("- Intercept is unstandardized (as in typical presentations of standardized-coefficient tables).\n")
            f.write("- p-values/stars are computed from this fitted model (Table 2 in the paper does not report SEs).\n")

        # Human-readable table like Table 2
        out_table = table.copy()
        out_table["beta_with_stars"] = out_table["beta"].map(
            lambda x: "" if pd.isna(x) else f"{x:.3f}"
        ) + out_table["stars"].where(out_table["stars"].isin(["*", "**", "***"]), "")
        out_table["b_unstd"] = out_table["b_unstd"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        out_table["p_value"] = out_table["p_value"].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("Standardized OLS coefficients (beta) with significance stars; intercept shown unstandardized.\n\n")
            f.write(out_table[["term", "beta_with_stars", "b_unstd", "p_value"]].to_string(index=False))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        return table, fit, model, d.index

    # -----------------------------
    # Load data, normalize columns, filter year
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must include 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct dependent variables (complete-case across items)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_count_dv(df, minority_items)   # 0-6
    df["dislike_other12_genres"] = build_count_dv(df, other12_items)     # 0-12

    # -----------------------------
    # Racism score (0-5), require all 5 items
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

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
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' column for education.")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing 'realinc' or 'hompop' needed for income per capita.")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column for occupational prestige.")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' column.")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (white reference): black, other_race
    if "race" not in df.columns:
        raise ValueError("Missing 'race' column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not present in provided variables; keep as missing
    # (This will reduce model N if included; we therefore do NOT include it to avoid forcing N to 0.)
    df["hispanic"] = np.nan

    # Conservative Protestant: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing 'relig' or 'denom' column.")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing 'region' column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Model specs (as implementable with provided fields)
    # Note: Hispanic is unavailable -> cannot include without collapsing N.
    # -----------------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # unavailable in this extract
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    var_order = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # -----------------------------
    # Fit two OLS models and export
    # -----------------------------
    results = {}

    tabA, fitA, modelA, idxA = fit_model(
        df=df,
        dv_col="dislike_minority_genres",
        x_cols=x_cols,
        model_name="Table2_ModelA_dislike_minority6",
        var_order_for_output=var_order,
    )
    tabB, fitB, modelB, idxB = fit_model(
        df=df,
        dv_col="dislike_other12_genres",
        x_cols=x_cols,
        model_name="Table2_ModelB_dislike_other12",
        var_order_for_output=var_order,
    )

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Diagnostics to explain N (where it collapses)
    diag_vars_A = ["dislike_minority_genres"] + x_cols
    diag_vars_B = ["dislike_other12_genres"] + x_cols

    diag = []
    for name, vars_ in [("ModelA", diag_vars_A), ("ModelB", diag_vars_B)]:
        tmp = df[vars_].copy().replace([np.inf, -np.inf], np.nan)
        diag.append(
            {
                "model": name,
                "n_year1993": int(df.shape[0]),
                "n_complete_cases": int(tmp.dropna().shape[0]),
                "missing_by_var": {v: int(tmp[v].isna().sum()) for v in vars_},
            }
        )

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2-style replication from provided GSS 1993 extract\n")
        f.write("Two OLS models with standardized coefficients (betas) computed from unstandardized OLS via SD-ratio.\n")
        f.write("IMPORTANT: 'hispanic' is not available in the provided variable list, so it cannot be included.\n\n")
        f.write("Model A DV: count of disliked among {rap, reggae, blues, jazz, gospel, latin} (0-6)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: count of disliked among the other 12 genres (0-12)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nDiagnostics (missingness):\n")
        for d in diag:
            f.write(f"\n{d['model']}:\n")
            f.write(f"  n_year1993: {d['n_year1993']}\n")
            f.write(f"  n_complete_cases: {d['n_complete_cases']}\n")
            f.write("  missing_by_var:\n")
            for k, v in d["missing_by_var"].items():
                f.write(f"    {k}: {v}\n")

    return results