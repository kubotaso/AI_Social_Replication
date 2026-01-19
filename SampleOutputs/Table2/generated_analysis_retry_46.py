def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_gss_missing(s):
        """
        Conservative GSS missing handling for this extract:
        - Coerce to numeric
        - Treat common sentinel codes as missing
        Note: We DO NOT blanket-drop large values (e.g., income), only known sentinels.
        """
        x = to_num(s)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items are 1-5, with 4/5 = dislike.
        Returns float {0,1} with NA preserved.
        """
        x = clean_gss_missing(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_recode(s, true_codes, false_codes):
        x = clean_gss_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete_case(df, item_cols):
        """
        Count of disliked items requiring complete data across the items
        (listwise within DV), consistent with "DK treated as missing and cases excluded."
        """
        inds = []
        for c in item_cols:
            inds.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(inds, axis=1)
        # require all items non-missing
        count = mat.sum(axis=1, min_count=len(item_cols))
        return count

    def standardized_betas_from_unstd(result, X, y):
        """
        Compute standardized betas for slopes as: beta = b * sd(x)/sd(y)
        using the estimation sample (after listwise deletion) that produced X,y.
        Intercept beta is NA.
        """
        y_sd = float(np.nanstd(y, ddof=0))
        betas = {}
        for term, b in result.params.items():
            if term == "const":
                betas[term] = np.nan
                continue
            x_sd = float(np.nanstd(X[term], ddof=0))
            if y_sd == 0 or x_sd == 0 or (not np.isfinite(y_sd)) or (not np.isfinite(x_sd)):
                betas[term] = np.nan
            else:
                betas[term] = float(b) * (x_sd / y_sd)
        return pd.Series(betas)

    def star_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df, dv, x_terms_in_order, model_name):
        """
        OLS with intercept.
        Output:
          - paper_style: standardized betas (slopes) + intercept unstandardized, stars from p-values
          - full_table: unstandardized coef, SE, t, p, standardized beta (slopes only)
          - fit: n, r2, adj_r2
        """
        needed = [dv] + x_terms_in_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(x_terms_in_order) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # ensure predictors vary (Table2 requires them)
        zero_var = []
        for c in x_terms_in_order:
            v = d[c].values
            if np.nanstd(v, ddof=0) == 0:
                zero_var.append(c)
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Check coding/sample restrictions."
            )

        y = d[dv].astype(float)
        X = d[x_terms_in_order].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        beta_std = standardized_betas_from_unstd(res, Xc, y)

        full = pd.DataFrame(
            {
                "b_unstd": res.params,
                "std_err": res.bse,
                "t": res.tvalues,
                "p_value": res.pvalues,
                "beta_std": beta_std,
            }
        )
        full.index.name = "term"

        # paper-style: standardized slopes, unstandardized constant (as commonly reported)
        paper_rows = []
        for term in ["racism_score", "education_years", "hh_income_per_capita", "occ_prestige",
                     "female", "age_years", "black", "hispanic", "other_race",
                     "cons_protestant", "no_religion", "south"]:
            if term not in full.index:
                raise ValueError(f"{model_name}: missing term in fitted model unexpectedly: {term}")
            coef = full.loc[term, "beta_std"]
            p = full.loc[term, "p_value"]
            paper_rows.append(
                {
                    "term": term,
                    "coef": float(coef) if pd.notna(coef) else np.nan,
                    "stars": star_from_p(p),
                }
            )

        # constant
        if "const" not in full.index:
            raise ValueError(f"{model_name}: missing intercept.")
        paper_rows.append(
            {
                "term": "const",
                "coef": float(full.loc["const", "b_unstd"]),
                "stars": star_from_p(full.loc["const", "p_value"]),
            }
        )
        paper = pd.DataFrame(paper_rows)

        fit = {
            "model": model_name,
            "n": int(res.nobs),
            "k": int(res.df_model + 1),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
        }

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
            f.write("\n\nFit:\n")
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")

        with open(f"./output/{model_name}_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Paper-style table: standardized coefficients for predictors (beta_std), unstandardized intercept.\n")
            f.write("Stars based on two-tailed p-values from this re-estimated model: * p<.05, ** p<.01, *** p<.001.\n\n")
            f.write(paper.to_string(index=False, justify="left", float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        return paper, full, pd.DataFrame([fit])

    # -----------------------------
    # Load and filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.shape[0] == 0:
        raise ValueError("No rows found for YEAR==1993.")

    # -----------------------------
    # Dependent variables (counts)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # -----------------------------
    # Racism score (0-5 additive)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_recode(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half black school
    rac2 = binary_recode(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_recode(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_recode(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_recode(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    rac_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = rac_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income per capita: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_recode(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: not available in this extract per mapping instruction -> use ETHNIC as last-resort heuristic,
    # but KEEP THIS EXPLICIT and conservative:
    # - If ETHNIC exists and is coded with values that include Hispanic/Latino group codes in this extract, define it.
    # - Otherwise set as missing, which will fail fast with a clear error at fit-time (Table 2 requires it).
    if "hispanic" in df.columns:
        # If the dataset already includes a hispanic column, use it as 0/1 with cleaning.
        df["hispanic"] = clean_gss_missing(df["hispanic"]).where(clean_gss_missing(df["hispanic"]).isin([0, 1]))
    elif "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        # Heuristic: treat common "Hispanic" coding as 1 if ETHNIC==1 (often "Mexican/Chicano/Spanish")
        # ONLY if that code exists; else set missing.
        if (eth == 1).any():
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))
        else:
            df["hispanic"] = np.nan
    else:
        df["hispanic"] = np.nan

    # Religion dummies
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing relig and/or denom columns needed for religion dummies.")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    consprot = np.where((relig == 1) & (denom.isin([1, 6, 7])), 1.0, 0.0).astype(float)
    consprot[(relig.isna()) | (denom.isna())] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = np.where(relig == 4, 1.0, 0.0).astype(float)
    norelig[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = np.where(region == 3, 1.0, 0.0).astype(float)
    south[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Fit models
    # -----------------------------
    x_order = [
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

    # If Hispanic is all-missing, we cannot estimate Table 2 faithfully; raise a clear error.
    if df["hispanic"].notna().sum() == 0:
        raise ValueError(
            "This dataset extract does not provide a usable Hispanic indicator (hispanic is entirely missing). "
            "Table 2 requires a Hispanic dummy. Provide an input with a proper Hispanic flag."
        )

    paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from provided microdata)\n")
        f.write("OLS with intercept. Standardized coefficients computed as b * SD(x)/SD(y) on estimation sample.\n")
        f.write("Stars from two-tailed p-values: * p<.05, ** p<.01, *** p<.001.\n\n")
        f.write("Model A DV: dislike_minority_genres (count of dislikes among rap, reggae, blues, jazz, gospel, latin)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(paperA.to_string(index=False, justify="left", float_format=lambda x: f"{x: .6f}"))
        f.write("\n\n")
        f.write("Model B DV: dislike_other12_genres (count of dislikes among the other 12 genres)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(paperB.to_string(index=False, justify="left", float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }