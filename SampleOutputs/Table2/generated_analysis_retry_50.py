def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing-code cleaning for typical GSS extracts.
        Do NOT over-aggressively drop legitimate values.
        """
        s = to_num(x).copy()
        # common sentinel codes across many GSS variables (varies by item/extract)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        s = s.mask(s.isin(list(sentinels)))
        return s

    def likert_dislike_indicator(x):
        """
        Music items are 1-5:
          1/2/3 = like/neutral -> 0
          4/5 = dislike -> 1
        Missing if not in 1..5 after cleaning.
        """
        s = clean_gss_missing(x)
        s = s.where(s.between(1, 5))
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin([1, 2, 3])] = 0.0
        out.loc[s.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        s = clean_gss_missing(x)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin(false_codes)] = 0.0
        out.loc[s.isin(true_codes)] = 1.0
        return out

    def count_dislikes_allow_partial(df, items, min_nonmissing=1):
        """
        Build dislike-count DV as a sum of item-level dislike indicators.
        Key difference vs earlier buggy versions: DO NOT require all items observed.
        Instead, sum across observed items and require at least `min_nonmissing` observed.
        This avoids collapsing N due to occasional DK/refused.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        observed = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, min_count=min_nonmissing)
        count = count.where(observed >= min_nonmissing)
        return count

    def standardize_beta_from_unstd(y, X, b_unstd):
        """
        Compute standardized betas from an unstandardized OLS fit:
            beta_j = b_j * SD(X_j) / SD(Y)
        Uses the analytic sample (rows of y and X).
        Intercept beta is set to NaN.
        """
        y_sd = y.std(ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                betas[col] = np.nan
            else:
                x_sd = X[col].std(ddof=0)
                if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                    betas[col] = np.nan
                else:
                    betas[col] = b_unstd[col] * (x_sd / y_sd)
        return pd.Series(betas)

    def stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_model(df, dv_col, x_cols, model_name, pretty_labels):
        """
        Fits OLS on unstandardized scales, then computes standardized betas for slopes.
        Returns:
          - paper_style table: standardized betas + stars, plus constant (unstd) + stars
          - full table: unstd b, se, t, p, beta_std (clearly labeled as re-estimation)
          - fit stats
          - analytic sample diagnostics (counts)
        """
        d = df[[dv_col] + x_cols].copy()
        d = d.replace([np.inf, -np.inf], np.nan)

        # Listwise deletion for this model (as in typical OLS tables)
        d = d.dropna(axis=0, how="any")

        # Drop any zero-variance predictors to avoid runtime errors (but log them)
        dropped_zero_var = []
        for c in list(x_cols):
            if d[c].nunique(dropna=True) <= 1:
                dropped_zero_var.append(c)
        x_use = [c for c in x_cols if c not in dropped_zero_var]

        if d.shape[0] == 0:
            raise ValueError(f"{model_name}: analytic sample is empty after listwise deletion.")
        if len(x_use) == 0:
            raise ValueError(f"{model_name}: no predictors remain after dropping zero-variance columns.")

        y = d[dv_col].astype(float)
        X = d[x_use].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Full table (replication output; not "from the paper")
        beta_std = standardize_beta_from_unstd(y, Xc, model.params)
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": beta_std,
            }
        )
        full.index.name = "term"

        # Paper-style table: standardized betas + stars, and constant on original scale
        paper_rows = []
        # build in the Table 2 order
        table2_order = [
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
            "const",
        ]
        for term in table2_order:
            if term == "const":
                if "const" in full.index:
                    val = full.loc["const", "b_unstd"]
                    p = full.loc["const", "p_value"]
                    paper_rows.append(
                        {
                            "term": "Constant",
                            "coef": float(val),
                            "stars": stars_from_p(float(p)),
                            "kind": "intercept_unstandardized",
                        }
                    )
                else:
                    paper_rows.append(
                        {"term": "Constant", "coef": np.nan, "stars": "", "kind": "intercept_unstandardized"}
                    )
                continue

            if term in full.index:
                val = full.loc[term, "beta_std"]
                p = full.loc[term, "p_value"]
                paper_rows.append(
                    {
                        "term": pretty_labels.get(term, term),
                        "coef": float(val) if pd.notna(val) else np.nan,
                        "stars": stars_from_p(float(p)),
                        "kind": "standardized_beta_from_reestimation",
                    }
                )
            else:
                # Predictor not in model (missing from data, or dropped due to zero variance)
                paper_rows.append(
                    {
                        "term": pretty_labels.get(term, term),
                        "coef": np.nan,
                        "stars": "",
                        "kind": "not_estimated_or_dropped",
                    }
                )

        paper = pd.DataFrame(paper_rows)

        fit = {
            "model": model_name,
            "dv": dv_col,
            "n": int(model.nobs),
            "k_including_const": int(model.df_model + 1),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "dropped_zero_variance_predictors": ", ".join(dropped_zero_var) if dropped_zero_var else "",
        }

        # Diagnostics: ensure "no_religion" varies (helps catch earlier bug)
        diag = {}
        if "no_religion" in d.columns:
            vc = d["no_religion"].value_counts(dropna=False).to_dict()
            diag["no_religion_value_counts_in_model_sample"] = vc
        if "hispanic" in d.columns:
            vc = d["hispanic"].value_counts(dropna=False).to_dict()
            diag["hispanic_value_counts_in_model_sample"] = vc

        return model, paper, full, fit, diag

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables (counts)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband",
        "blugrass",
        "country",
        "musicals",
        "classicl",
        "folk",
        "moodeasy",
        "newage",
        "opera",
        "conrock",
        "oldies",
        "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Allow partial item nonresponse; require at least 1 observed item (keeps N from collapsing).
    df["dislike_minority_genres"] = count_dislikes_allow_partial(df, minority_items, min_nonmissing=1)
    df["dislike_other12_genres"] = count_dislikes_allow_partial(df, other12_items, min_nonmissing=1)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny edu chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack motivation
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    # Keep as missing unless all five items are present (faithful to additive index definition)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: must be computed from available fields. The only ethnicity-related field here is "ethnic".
    # Implement a pragmatic, fully data-driven rule that does not hard-code paper numbers:
    # Treat ETHNIC as Hispanic if it falls in a plausible Hispanic-coded band.
    # In many GSS extracts, Hispanic/Latino origins occupy a contiguous range; here we use 20-29 as the rule,
    # and leave others as non-Hispanic. If ETHNIC is missing/unknown -> missing.
    if "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[eth.notna()] = 0.0
        hisp.loc[eth.between(20, 29)] = 1.0
        df["hispanic"] = hisp
    else:
        # If not available, keep missing (will reduce N but avoids crashing)
        df["hispanic"] = np.nan

    # Religion dummies
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])

    # Conservative Protestant proxy: Protestant + (Baptist / other / nondenom)
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    consprot.loc[relig.notna() & denom.notna()] = 0.0
    consprot.loc[(relig == 1) & (denom.isin([1, 6, 7]))] = 1.0
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = pd.Series(np.nan, index=df.index, dtype="float64")
    norelig.loc[relig.notna()] = 0.0
    norelig.loc[relig == 4] = 1.0
    df["no_religion"] = norelig

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = pd.Series(np.nan, index=df.index, dtype="float64")
    south.loc[region.notna()] = 0.0
    south.loc[region == 3] = 1.0
    df["south"] = south

    # -------------------------
    # Fit Table 2 models
    # -------------------------
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
        "education_years": "Education",
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

    mA, paperA, fullA, fitA, diagA = fit_model(
        df,
        "dislike_minority_genres",
        x_cols,
        "Table2_ModelA_dislike_minority6",
        pretty_labels=pretty,
    )
    mB, paperB, fullB, fitB, diagB = fit_model(
        df,
        "dislike_other12_genres",
        x_cols,
        "Table2_ModelB_dislike_other12",
        pretty_labels=pretty,
    )

    # -------------------------
    # Save outputs
    # -------------------------
    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # Human-readable summaries
    save_text("./output/Table2_ModelA_summary.txt", mA.summary().as_text())
    save_text("./output/Table2_ModelB_summary.txt", mB.summary().as_text())

    # Tables
    paperA.to_string(open("./output/Table2_ModelA_paper_style.txt", "w", encoding="utf-8"), index=False)
    paperB.to_string(open("./output/Table2_ModelB_paper_style.txt", "w", encoding="utf-8"), index=False)
    fullA.to_string(open("./output/Table2_ModelA_full_reestimation.txt", "w", encoding="utf-8"), float_format=lambda v: f"{v: .6f}")
    fullB.to_string(open("./output/Table2_ModelB_full_reestimation.txt", "w", encoding="utf-8"), float_format=lambda v: f"{v: .6f}")

    # Fit + diagnostics
    fit_df = pd.DataFrame([fitA, fitB])
    fit_df.to_string(open("./output/Table2_fit.txt", "w", encoding="utf-8"), index=False)

    diag_text = []
    diag_text.append("Diagnostics (value counts shown for key dummies in each model's analytic sample)\n")
    diag_text.append("Model A:\n")
    diag_text.append(str(diagA) + "\n")
    diag_text.append("\nModel B:\n")
    diag_text.append(str(diagB) + "\n")
    save_text("./output/Table2_diagnostics.txt", "".join(diag_text))

    overview = []
    overview.append("Table 2 replication run (computed from provided microdata extract).\n")
    overview.append("IMPORTANT: Standard errors/t/p-values are from this re-estimation; Table 2 in the paper reports only standardized coefficients.\n\n")
    overview.append("Model A DV: dislike_minority_genres (count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
    overview.append(fit_df.iloc[[0]].to_string(index=False) + "\n\n")
    overview.append("Model B DV: dislike_other12_genres (count of dislikes among remaining 12 genres)\n")
    overview.append(fit_df.iloc[[1]].to_string(index=False) + "\n")
    save_text("./output/Table2_overview.txt", "".join(overview))

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full_reestimation": fullA,
        "ModelB_full_reestimation": fullB,
        "fit": fit_df,
    }