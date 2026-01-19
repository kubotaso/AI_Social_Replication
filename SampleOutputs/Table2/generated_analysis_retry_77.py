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

    def clean_gss_na(series):
        """
        Conservative NA cleaning for this extract:
        - Coerce to numeric
        - Treat common GSS NA sentinels as missing
        Note: We do NOT blanket-drop all 8/9 because some variables may legitimately take those values.
        For the variables used here, the specific recodes below further constrain valid ranges/codes.
        """
        x = to_num(series).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(series):
        """
        Music liking: valid 1..5; dislike if 4 or 5; like/neutral if 1..3.
        Anything outside 1..5 treated missing.
        """
        x = clean_gss_na(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_gss_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_complete=True):
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        if require_complete:
            # Bryson: DK treated as missing; simplest faithful approach is complete-case for the DV components
            return mat.sum(axis=1, min_count=len(items))
        # (not used)
        return mat.sum(axis=1, min_count=1)

    def standardized_betas_from_unstd_fit(model, X_raw, y_raw):
        """
        Compute standardized beta weights from an unstandardized OLS fit:
        beta_j = b_j * sd(x_j) / sd(y)
        Intercept is not standardized and is reported separately as the model intercept.
        """
        y_sd = np.std(y_raw, ddof=0)
        betas = {}
        for name in model.params.index:
            if name == "const":
                continue
            x_sd = np.std(X_raw[name], ddof=0)
            betas[name] = model.params[name] * (x_sd / y_sd) if (np.isfinite(x_sd) and x_sd > 0 and np.isfinite(y_sd) and y_sd > 0) else np.nan
        return pd.Series(betas)

    def star_from_p(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_one(df, dv_col, xcols, model_name, pretty_names):
        needed = [dv_col] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if d.shape[0] < len(xcols) + 5:
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv_col].astype(float)
        X = d[xcols].astype(float)

        # Drop any zero-variance predictors in-sample (prevents singularities / NaN coefficients)
        keep = [c for c in X.columns if np.isfinite(np.std(X[c], ddof=0)) and np.std(X[c], ddof=0) > 0]
        dropped = [c for c in X.columns if c not in keep]
        X = X[keep]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas computed from the *same estimation sample* and the *unstandardized* fit
        betas = standardized_betas_from_unstd_fit(model, X, y)
        # p-values come from unstandardized model; stars match Table 2 convention (two-tailed)
        pvals = model.pvalues.drop(labels=["const"], errors="ignore")

        out = pd.DataFrame(
            {
                "beta_std": betas,
                "p_value": pvals.reindex(betas.index),
            }
        )
        out["stars"] = out["p_value"].map(star_from_p)
        out["beta_std_star"] = out["beta_std"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "") + out["stars"]

        # Add constant (unstandardized, like in many standardized-beta tables)
        const = float(model.params.get("const", np.nan))
        const_p = float(model.pvalues.get("const", np.nan))
        const_star = star_from_p(const_p)
        const_row = pd.DataFrame(
            {
                "beta_std": [np.nan],
                "p_value": [const_p],
                "stars": [const_star],
                "beta_std_star": [f"{const:.3f}{const_star}" if np.isfinite(const) else ""],
            },
            index=["const"],
        )

        # Reindex to the Table 2 order (only those present after dropping zero-variance predictors)
        desired_order = [
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
        present = [c for c in desired_order if c in out.index]
        table = pd.concat([out.loc[present], const_row], axis=0)

        # Pretty label index for human-readable output
        table.insert(0, "variable", [pretty_names.get(ix, ix) for ix in table.index])
        table = table[["variable", "beta_std_star", "beta_std", "p_value"]]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save human-readable files
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Standardized betas computed post-estimation: b * SD(X)/SD(Y) on the estimation sample.\n")
            f.write("- Stars are from two-tailed p-values of the unstandardized OLS coefficients.\n")
            if dropped:
                f.write(f"- Dropped in-sample zero-variance predictors: {', '.join(dropped)}\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(table.to_string(index=False))

        return model, table, fit

    # -----------------------------
    # Load data and standardize names
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs: dislike counts
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_complete=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_complete=True)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
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

    # -----------------------------
    # Controls
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_gss_na(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(clean_gss_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_gss_na(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_gss_na(df["race"]).where(clean_gss_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: may not exist; prefer an explicit hispanic flag if present; otherwise do NOT proxy with ethnic.
    if "hispanic" in df.columns:
        hisp = clean_gss_na(df["hispanic"])
        # common coding: 1 yes, 2 no
        df["hispanic"] = binary_from_codes(hisp, true_codes=[1], false_codes=[2])
    else:
        # Keep as all-missing; model code will drop it (and note via diagnostics)
        df["hispanic"] = np.nan

    # Religion: No religion
    if "relig" not in df.columns:
        raise ValueError("Missing RELIG column (relig).")
    relig = clean_gss_na(df["relig"])
    # GSS RELIG: 4 = none in many codebooks; keep this as specified
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "denom" not in df.columns:
        raise ValueError("Missing DENOM column (denom).")
    denom = clean_gss_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()))
    df["cons_protestant"] = consprot

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_gss_na(df["region"]).where(clean_gss_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Model specification: Table 2 RHS
    # -----------------------------
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

    pretty = {
        "racism_score": "Racism score (0–5)",
        "education_years": "Education (years)",
        "hh_income_per_capita": "Household income per capita",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
        "const": "Constant",
    }

    # -----------------------------
    # Diagnostics: missingness in 1993 subsample
    # -----------------------------
    diag_cols = ["dislike_minority_genres", "dislike_other12_genres"] + xcols
    diag = pd.DataFrame(
        {
            "missing_n": df[diag_cols].isna().sum(),
            "nonmissing_n": df[diag_cols].notna().sum(),
            "missing_pct": (df[diag_cols].isna().mean() * 100).round(1),
        }
    ).sort_values(["missing_n", "missing_pct"], ascending=False)

    with open("./output/diagnostics_missingness.txt", "w", encoding="utf-8") as f:
        f.write("Missingness diagnostics (YEAR==1993)\n\n")
        f.write(diag.to_string())

    # -----------------------------
    # Fit both models
    # -----------------------------
    modelA, tableA, fitA = fit_one(
        df=df,
        dv_col="dislike_minority_genres",
        xcols=xcols,
        model_name="Table2_ModelA_dislike_minority6",
        pretty_names=pretty,
    )
    modelB, tableB, fitB = fit_one(
        df=df,
        dv_col="dislike_other12_genres",
        xcols=xcols,
        model_name="Table2_ModelB_dislike_other12",
        pretty_names=pretty,
    )

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS)\n")
        f.write("OLS on unstandardized variables; standardized betas computed as b * SD(X)/SD(Y).\n")
        f.write("Stars based on two-tailed p-values from the unstandardized OLS coefficients.\n\n")
        f.write("Model A: DV = count of disliked among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(tableA.to_string(index=False))
        f.write("\n\n")
        f.write("Model B: DV = count of disliked among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(tableB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "missingness_diagnostics": diag,
    }