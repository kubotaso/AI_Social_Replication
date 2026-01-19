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

    def clean_gss_missing(x):
        """
        Conservative cleaning of common GSS missing codes across numeric series.
        We only coerce obvious sentinel codes; keep everything else.
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(s):
        """
        1-5 like/dislike scale; dislike if in {4,5}; like/neutral if in {1,2,3}.
        Anything else -> missing.
        """
        x = clean_gss_missing(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_gss_missing(s)
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

    def build_dislike_count(df, items, require_complete=True):
        """
        Build a dislike count from given items (0/1 each).
        Paper summary says DK treated as missing and cases excluded; default require complete.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_complete:
            return mat.sum(axis=1, min_count=len(items))
        # fallback: allow partial (not used by default)
        return mat.sum(axis=1, min_count=1)

    def add_stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardized_ols_with_unstd_intercept(df, dv, xcols, model_name):
        """
        Table-2 style: standardized betas (z-scored y and x) for predictors,
        plus *unstandardized* intercept from model on original DV scale.

        Returns:
          - beta_table: standardized betas + stars (computed from standardized model p-values),
          - intercept_value: unstandardized intercept (original DV scale),
          - fit dict: N, R2, Adj R2 from standardized model + N from unstandardized model (should match)
        """
        needed = [dv] + xcols
        d = df[needed].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={d.shape[0]}).")

        # Standardized model (for betas and fit)
        y_z = zscore(d[dv])
        Xz = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)

        # Drop any predictor that became constant/NaN after standardization, but do not silently
        # proceed without warning in the output.
        bad = [c for c in Xz.columns if Xz[c].isna().any() or Xz[c].std(ddof=0) == 0]
        good = [c for c in Xz.columns if c not in bad]
        if len(good) == 0:
            raise ValueError(f"{model_name}: all predictors invalid after standardization (constants or missing).")
        Xz = Xz[good]

        Xz_c = sm.add_constant(Xz, has_constant="add")
        std_model = sm.OLS(y_z.loc[Xz.index], Xz_c).fit()

        # Unstandardized intercept model (original DV scale) on the same estimation sample/predictors
        # Use the same set of predictors that survived standardization.
        Xu = sm.add_constant(d[good], has_constant="add")
        unstd_model = sm.OLS(d[dv], Xu).fit()
        intercept = float(unstd_model.params.get("const", np.nan))

        # Build Table-2-like output: standardized betas (no SE/t/p columns)
        betas = std_model.params.drop(labels=["const"], errors="ignore").copy()
        pvals = std_model.pvalues.drop(labels=["const"], errors="ignore").copy()

        out = pd.DataFrame(
            {
                "beta_std": betas,
                "stars": [add_stars_from_p(pvals.get(k, np.nan)) for k in betas.index],
            }
        )
        out.index.name = "term"
        out = out.loc[good]  # enforce requested ordering (good follows xcols order subset)

        fit = {
            "model": model_name,
            "n": int(std_model.nobs),
            "k_predictors": int(std_model.df_model),
            "r2": float(std_model.rsquared),
            "adj_r2": float(std_model.rsquared_adj),
            "intercept_unstandardized": intercept,
            "dropped_predictors_after_standardization": ", ".join(bad) if bad else "",
        }
        return out, fit, std_model, unstd_model

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required basics
    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Dependent variables
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Per paper summary: DK treated as missing and missing cases excluded
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_complete=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_complete=True)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    df["age_years"] = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))

    # Race indicators (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not available in provided variables.
    # To keep the model faithful without fabricating a proxy, create a missing column and then
    # explicitly OMIT it from estimation (and note this omission in outputs).
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype("float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype("float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index).astype("float64")
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Models (Table 2 RHS)
    # NOTE: Hispanic is unavailable here -> cannot estimate its coefficient.
    # We do NOT include it, and we state this explicitly in the saved outputs.
    # -----------------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # unavailable in provided data extract
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Sanity checks for invariance in full 1993 sample (not model sample)
    diag = {}
    for c in ["dislike_minority_genres", "dislike_other12_genres"] + x_cols:
        s = df[c]
        diag[c] = {
            "nonmissing": int(s.notna().sum()),
            "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
            "std": float(s.std(skipna=True, ddof=0)) if s.notna().any() else np.nan,
            "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
            "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
        }
    diag_df = pd.DataFrame(diag).T
    diag_df.index.name = "variable"
    diag_df.to_string(open("./output/diagnostics_1993.txt", "w", encoding="utf-8"), float_format=lambda v: f"{v: .6f}")

    # Run both models
    betaA, fitA, stdA, unstdA = standardized_ols_with_unstd_intercept(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    betaB, fitB, stdB, unstdB = standardized_ols_with_unstd_intercept(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )

    # -----------------------------
    # Save human-readable outputs
    # -----------------------------
    def write_table2_style(beta_df, fit, path_table, path_summary):
        with open(path_table, "w", encoding="utf-8") as f:
            f.write("Table-2-style output: Standardized coefficients (beta) for predictors; stars from two-tailed p-values of the standardized OLS fit.\n")
            f.write("Note: Intercept is NOT standardized and is reported separately in the summary file.\n\n")
            f.write(beta_df.to_string(float_format=lambda v: f"{v: .3f}"))
            f.write("\n")

        with open(path_summary, "w", encoding="utf-8") as f:
            f.write("Model fit and notes\n")
            f.write("-------------------\n")
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")

    write_table2_style(
        betaA, fitA,
        "./output/Table2_ModelA_table.txt",
        "./output/Table2_ModelA_fit_and_notes.txt"
    )
    write_table2_style(
        betaB, fitB,
        "./output/Table2_ModelB_table.txt",
        "./output/Table2_ModelB_fit_and_notes.txt"
    )

    # Save full regression summaries too (explicitly re-estimated from microdata)
    with open("./output/Table2_ModelA_full_summary_standardized.txt", "w", encoding="utf-8") as f:
        f.write(stdA.summary().as_text())
    with open("./output/Table2_ModelA_full_summary_unstandardized.txt", "w", encoding="utf-8") as f:
        f.write(unstdA.summary().as_text())

    with open("./output/Table2_ModelB_full_summary_standardized.txt", "w", encoding="utf-8") as f:
        f.write(stdB.summary().as_text())
    with open("./output/Table2_ModelB_full_summary_unstandardized.txt", "w", encoding="utf-8") as f:
        f.write(unstdB.summary().as_text())

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Replication from microdata (computed; not copied from paper)\n")
        f.write("=========================================================\n\n")
        f.write("Important: The provided data extract has no direct Hispanic identifier; therefore the 'Hispanic' dummy in the published Table 2 cannot be estimated here.\n")
        f.write("We omit that term rather than proxying with ETHNIC.\n\n")

        f.write("Model A DV: dislike_minority_genres (0-6)\n")
        for k, v in fitA.items():
            f.write(f"{k}: {v}\n")
        f.write("\nModel A standardized betas:\n")
        f.write(betaA.to_string(float_format=lambda v: f"{v: .3f}"))
        f.write("\n\n")

        f.write("Model B DV: dislike_other12_genres (0-12)\n")
        for k, v in fitB.items():
            f.write(f"{k}: {v}\n")
        f.write("\nModel B standardized betas:\n")
        f.write(betaB.to_string(float_format=lambda v: f"{v: .3f}"))
        f.write("\n")

    # Return results as dict of DataFrames for programmatic use
    fit_df = pd.DataFrame([fitA, fitB])
    return {
        "ModelA_betas": betaA,
        "ModelB_betas": betaB,
        "Fit": fit_df,
        "Diagnostics_1993": diag_df,
    }