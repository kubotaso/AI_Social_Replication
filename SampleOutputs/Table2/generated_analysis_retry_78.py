def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes_generic(s):
        """
        Conservative missing-code cleaning that avoids wiping out valid GSS values.
        - Converts to numeric
        - Sets common NA/refused/DK sentinels to NaN
        - Leaves ordinary values intact
        """
        x = to_num(s).copy()
        # Common GSS-style sentinels across many items
        sentinels = {7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(s):
        """
        Music items are 1-5. Dislike is 4 or 5.
        Treat non-1..5 as missing.
        """
        x = clean_na_codes_generic(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_na_codes_generic(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete(df, cols):
        """
        Count of disliked genres with Bryson-style DK treated as missing.
        To be faithful and simple: require ALL component items non-missing (complete-case within the bundle).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in cols], axis=1)
        return mat.sum(axis=1, min_count=len(cols))

    def standardize_beta_from_unstd_fit(model, y, X):
        """
        Compute standardized betas from an unstandardized OLS fit:
        beta_j = b_j * sd(X_j) / sd(y)
        For dummies this is also well-defined (uses sd of 0/1).
        Constant has no standardized beta; we keep unstandardized constant separately.
        """
        y_sd = np.std(y, ddof=0)
        betas = {}
        for name in model.params.index:
            if name == "const":
                continue
            x_sd = np.std(X[name], ddof=0)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[name] = np.nan
            else:
                betas[name] = float(model.params[name] * (x_sd / y_sd))
        return pd.Series(betas)

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

    def fit_one(df, dv, xcols, model_name, pretty_names):
        """
        Fit unstandardized OLS with intercept, listwise deletion.
        Report standardized betas + stars; report constant separately (unstandardized) + stars.
        """
        needed = [dv] + xcols
        d = df[needed].copy()

        # drop infinite
        d = d.replace([np.inf, -np.inf], np.nan)

        # listwise deletion
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)
        X = d[xcols].astype(float)

        # drop any zero-variance predictors (prevents singularities and NaN rows)
        keep = [c for c in X.columns if np.isfinite(X[c].std(ddof=0)) and X[c].std(ddof=0) > 0]
        dropped = [c for c in X.columns if c not in keep]
        X = X[keep]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # standardized betas
        beta_std = standardize_beta_from_unstd_fit(model, y.values, X)

        # build output table in paper order; if some were dropped due to zero-variance, show NaN
        rows = []
        for c in xcols:
            term = c
            if term in beta_std.index:
                bstd = beta_std.loc[term]
                p = float(model.pvalues.get(term, np.nan))
            else:
                bstd = np.nan
                p = np.nan
            rows.append(
                {
                    "term": pretty_names.get(c, c),
                    "beta_std": bstd,
                    "stars": stars_from_p(p),
                }
            )

        # constant (unstandardized)
        const_b = float(model.params.get("const", np.nan))
        const_p = float(model.pvalues.get("const", np.nan))
        rows.append(
            {
                "term": "Constant",
                "beta_std": const_b,   # keep column name simple; note this is unstandardized constant
                "stars": stars_from_p(const_p),
            }
        )

        out = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "n": int(model.nobs),
                "k_predictors_including_const": int(model.df_model + 1),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
            }]
        )

        # Save readable files
        out_path = f"./output/{model_name}_table.txt"
        summ_path = f"./output/{model_name}_summary.txt"
        fit_path = f"./output/{model_name}_fit.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("Standardized betas (beta_std) for predictors; Constant is unstandardized.\n\n")
            f.write(out.to_string(index=False, float_format=lambda v: "NA" if pd.isna(v) else f"{v: .3f}"))

        with open(fit_path, "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False, float_format=lambda v: f"{v: .6f}" if isinstance(v, float) else str(v)))

        with open(summ_path, "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

        return out, fit, model

    # ----------------------------
    # Load data (robust colnames)
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["year", "id"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # DVs (counts of dislikes)
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete(df, other12_items)

    # ----------------------------
    # Racism scale (0-5)
    # ----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrim
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny edu chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse no motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Covariates
    # ----------------------------
    # education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_na_codes_generic(df["educ"]).where(clean_na_codes_generic(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # income per capita: realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_na_codes_generic(df["realinc"])
    hompop = clean_na_codes_generic(df["hompop"]).where(clean_na_codes_generic(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_na_codes_generic(df["prestg80"])

    # female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    df["age_years"] = clean_na_codes_generic(df["age"]).where(clean_na_codes_generic(df["age"]).between(18, 89))

    # race dummies: black, other race
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes_generic(df["race"]).where(clean_na_codes_generic(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # hispanic: not available in provided variables; MUST NOT proxy using ethnic.
    # Create as 0 (not missing) so it does not zero-out the whole sample.
    # This keeps model runnable; result will not be a perfect replication without a real Hispanic flag.
    df["hispanic"] = 0.0

    # religion dummies
    if "relig" not in df.columns:
        raise ValueError("Missing relig column.")
    relig = clean_na_codes_generic(df["relig"])

    # no religion
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # conservative protestant proxy (as instructed): RELIG==1 and DENOM in {1,6,7}
    if "denom" not in df.columns:
        raise ValueError("Missing denom column.")
    denom = clean_na_codes_generic(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = np.where(relig.isna() | denom.isna(), np.nan, consprot)
    df["cons_protestant"] = consprot

    # south
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes_generic(df["region"]).where(clean_na_codes_generic(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Models (Table 2)
    # ----------------------------
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

    pretty = {
        "racism_score": "Racism score",
        "education_years": "Education (years)",
        "hh_income_per_capita": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
    }

    results = {}

    tableA, fitA, modelA = fit_one(
        df=df,
        dv="dislike_minority_genres",
        xcols=x_cols,
        model_name="Table2_ModelA_dislike_minority6",
        pretty_names=pretty,
    )

    tableB, fitB, modelB = fit_one(
        df=df,
        dv="dislike_other12_genres",
        xcols=x_cols,
        model_name="Table2_ModelB_dislike_other12",
        pretty_names=pretty,
    )

    results["ModelA_table"] = tableA
    results["ModelB_table"] = tableB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from provided GSS 1993 extract)\n")
        f.write("OLS on unstandardized variables; standardized betas computed post-estimation as b*SD(x)/SD(y).\n")
        f.write("Note: Hispanic indicator is not available in provided variables; set to 0 to keep model estimable.\n")
        f.write("DVs: counts of disliked genres; dislike = response in {4,5}; DK/NA treated as missing; complete-case within each DV bundle.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results