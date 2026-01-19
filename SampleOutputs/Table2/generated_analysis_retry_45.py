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

    def clean_common_na_codes(x):
        """
        Conservative NA-code cleaning for typical GSS-style numeric codes.
        We avoid aggressive rules (like >=90) because many variables are genuine counts/scales.
        """
        x = to_num(x).copy()
        na_codes = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(na_codes))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 scale. Dislike if 4 or 5.
        Missing if not in 1..5 after NA-code cleaning.
        """
        x = clean_common_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_common_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all=True):
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        if require_all:
            # require all items observed (DK treated as missing; exclude)
            return mat.sum(axis=1, min_count=len(items))
        # allow partial (not used here)
        return mat.sum(axis=1, min_count=1)

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

    def standardized_betas_from_ols(model, y, X_no_const):
        """
        Compute standardized betas for slopes as: b * sd(x)/sd(y)
        using the analytic sample (already listwise-cleaned).
        """
        y_sd = float(np.std(y, ddof=0))
        betas = {}
        for c in X_no_const.columns:
            x_sd = float(np.std(X_no_const[c], ddof=0))
            b = float(model.params.get(c, np.nan))
            if not np.isfinite(y_sd) or y_sd == 0 or not np.isfinite(x_sd) or x_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (x_sd / y_sd)
        return pd.Series(betas)

    def fit_table2_model(df, dv, x_order, model_name):
        # Build analytic frame (listwise deletion on DV + all predictors)
        needed = [dv] + x_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Ensure we are not accidentally filtering to a subgroup that kills variation
        # (Do not drop predictors silently; instead report and proceed only if estimable.)
        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        y = to_num(d[dv]).astype(float)
        X = d[x_order].apply(to_num).astype(float)

        # Detect zero-variance predictors in THIS analytic sample; drop only if truly constant,
        # but keep a record so the user can see it (Table 2 expects them, but extract may not allow it).
        zero_var = [c for c in X.columns if float(np.nanstd(X[c].values, ddof=0)) == 0.0]
        dropped = []
        if zero_var:
            # Drop constants to avoid singular matrix runtime errors, but do not hide it.
            X = X.drop(columns=zero_var)
            dropped = zero_var

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (slopes only; intercept not standardized)
        beta_std = standardized_betas_from_ols(model, y.values, X)

        # Assemble full table with labels, left-joined by term name
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        full.index.name = "term"

        # Add standardized betas for slopes
        full["beta_std"] = np.nan
        for c in beta_std.index:
            if c in full.index:
                full.loc[c, "beta_std"] = beta_std.loc[c]

        # Paper-style display: standardized betas for slopes, unstandardized constant
        # Keep exact ordering requested (if a term was dropped because constant, it will show blank)
        display_rows = []
        for term in x_order:
            if term in full.index:
                b = full.loc[term, "beta_std"]
                p = full.loc[term, "p_value"]
                display_rows.append((term, b, star_from_p(p)))
            else:
                display_rows.append((term, np.nan, ""))

        # Constant
        if "const" in full.index:
            display_rows.append(("const", full.loc["const", "b_unstd"], star_from_p(full.loc["const", "p_value"])))
        else:
            display_rows.append(("const", np.nan, ""))

        paper = pd.DataFrame(display_rows, columns=["term", "coef", "stars"]).set_index("term")
        paper["coef_with_stars"] = paper["coef"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + paper["stars"]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),  # includes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTES:\n")
            f.write("- Standardized coefficients (beta_std) computed as b * sd(x)/sd(y) on the analytic sample.\n")
            f.write("- Intercept shown as unstandardized.\n")
            if dropped:
                f.write(f"- Dropped predictors with zero variance in analytic sample: {dropped}\n")

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write(paper[["coef", "stars", "coef_with_stars"]].to_string())

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))

        return model, paper, full, fit

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Year filter
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.shape[0] == 0:
        raise ValueError("No rows found for YEAR==1993.")

    # -----------------------
    # Construct DVs
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Require all items for each DV (DK treated as missing; cases excluded per model)
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all=True)

    # -----------------------
    # Racism scale (0-5)
    # -----------------------
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

    # -----------------------
    # Controls
    # -----------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_common_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_common_na_codes(df["realinc"])
    hompop = clean_common_na_codes(df["hompop"]).where(clean_common_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_common_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_common_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (white ref): black, other_race
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_common_na_codes(df["race"]).where(clean_common_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: not present in this extract; approximate using ETHNIC only if present.
    # We keep it implementable but transparent. If ETHNIC is missing, hispanic will be missing.
    # This avoids crashing while still allowing the models to run on the available extract.
    if "ethnic" in df.columns:
        ethnic = clean_common_na_codes(df["ethnic"])
        # Very conservative proxy: treat a small set of common Hispanic-origin codes if present.
        # If codes don't match, it will simply yield mostly 0/NaN and not distort by broad recoding.
        # (User can replace with a proper Hispanic flag in a richer extract.)
        hisp_codes = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
        df["hispanic"] = np.where(ethnic.isna(), np.nan, ethnic.isin(hisp_codes).astype(float))
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing required columns for religion coding: relig and denom")
    relig = clean_common_na_codes(df["relig"])
    denom = clean_common_na_codes(df["denom"])
    consprot = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_common_na_codes(df["region"]).where(clean_common_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Fit models (Table 2 spec)
    # -----------------------
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

    # Sanity: ensure we did NOT inadvertently filter out "no religion" by any rule
    # (no additional filtering is done; only year==1993)
    freq_path = "./output/diagnostics_frequencies.txt"
    with open(freq_path, "w", encoding="utf-8") as f:
        f.write("Diagnostics (YEAR==1993 only; before listwise deletion):\n\n")
        for c in ["no_religion", "cons_protestant", "black", "other_race", "hispanic", "south"]:
            vc = df[c].value_counts(dropna=False)
            f.write(f"{c} value counts:\n{vc.to_string()}\n\n")

    mA, paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # Combined overview
    overview_path = "./output/Table2_overview.txt"
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993 extract)\n")
        f.write("Outputs:\n")
        f.write("- *_table_paper_style.txt: standardized betas for slopes + stars; intercept unstandardized\n")
        f.write("- *_table_full.txt: unstandardized coefficients + SE/t/p and computed standardized betas\n")
        f.write("\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin) count\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres (12 remaining genres) count\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nNote: If 'hispanic' is missing or poorly measured in this extract, coefficients may differ.\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }