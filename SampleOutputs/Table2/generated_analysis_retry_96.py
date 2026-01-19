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

    def clean_gss_numeric(x):
        """
        Conservative missing-code handling for this extract:
        - convert to numeric
        - treat common GSS sentinel codes as missing
        """
        x = to_num(x).copy()
        # common GSS-style missing codes in numeric extracts
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items expected 1-5:
        1 like very much, 2 like, 3 neither, 4 dislike, 5 dislike very much
        Dislike indicator = 1 if 4/5, 0 if 1/2/3, missing otherwise.
        """
        x = clean_gss_numeric(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_gss_numeric(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        # require all component items observed (DK etc. treated as missing -> exclude)
        return mat.sum(axis=1, min_count=len(items))

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

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

    def standardized_ols(df, dv, xcols, model_name):
        """
        Standardized coefficients (beta) computed by running OLS on z-scored Y and z-scored X.
        Intercept is taken from an unstandardized OLS on the original DV scale.
        Listwise deletion on dv + all xcols (as replication default).
        """
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if d.shape[0] < len(xcols) + 5:
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={d.shape[0]}).")

        # Check for zero-variance predictors in the estimation sample
        zero_var = [c for c in xcols if d[c].nunique(dropna=True) <= 1]
        if zero_var:
            # Fail loudly: Table 2 variables should not drop silently
            raise ValueError(f"{model_name}: zero-variance predictors in estimation sample: {zero_var}")

        # Unstandardized model for intercept / fit diagnostics on original DV scale
        Xu = sm.add_constant(d[xcols], has_constant="add")
        yu = d[dv]
        model_u = sm.OLS(yu, Xu).fit()

        # Standardized model for betas
        yz = zscore(yu)
        Xz = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)
        # After zscore, ensure no NaNs introduced (shouldn't happen if variance > 0)
        bad = [c for c in xcols if Xz[c].isna().any()]
        if bad:
            raise ValueError(f"{model_name}: predictors became undefined after standardization: {bad}")
        Xz_c = sm.add_constant(Xz, has_constant="add")
        model_z = sm.OLS(yz, Xz_c).fit()

        # Build Table-2-like output: standardized betas + stars; constant from unstandardized
        rows = []
        # Constant row (paper reports it)
        rows.append(
            {
                "term": "Constant",
                "beta": float(model_u.params["const"]),
                "stars": stars_from_p(float(model_u.pvalues["const"])),
            }
        )
        for c in xcols:
            rows.append(
                {
                    "term": c,
                    "beta": float(model_z.params[c]),
                    "stars": stars_from_p(float(model_z.pvalues[c])),
                }
            )

        out = pd.DataFrame(rows)
        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model_u.nobs),
                    "k_including_const": int(model_u.df_model + 1),
                    "r2": float(model_u.rsquared),
                    "adj_r2": float(model_u.rsquared_adj),
                }
            ]
        )

        # Save human-readable summaries
        with open(f"./output/{model_name}_summary_unstandardized.txt", "w", encoding="utf-8") as f:
            f.write(model_u.summary().as_text())
        with open(f"./output/{model_name}_summary_standardized.txt", "w", encoding="utf-8") as f:
            f.write(model_z.summary().as_text())

        # Save table
        tbl_path = f"./output/{model_name}_table.txt"
        with open(tbl_path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("Note: 'beta' is standardized for predictors; Constant is unstandardized on DV scale.\n")
            f.write(out.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
            f.write("\n\nFit (from unstandardized model on DV scale):\n")
            f.write(fit.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
            f.write("\n")

        return out, fit, model_u, model_z

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.empty:
        raise ValueError("No rows found for YEAR==1993")

    # ----------------------------
    # Construct dependent variables
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music taste column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # ----------------------------
    # Construct racism score (0-5)
    # ----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism component column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Controls
    # ----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_numeric(df["educ"]).where(clean_gss_numeric(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_gss_numeric(df["realinc"])
    hompop = clean_gss_numeric(df["hompop"]).where(clean_gss_numeric(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_numeric(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age"] = clean_gss_numeric(df["age"]).where(clean_gss_numeric(df["age"]).between(18, 89))

    # Race dummies: black and other_race from RACE (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_numeric(df["race"]).where(clean_gss_numeric(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables -> keep as missing; cannot estimate Table-2 exact model
    # Still include in table pipeline as a placeholder? No: would force listwise deletion to zero.
    # Therefore: create a column but do NOT include in xcols.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_gss_numeric(df["relig"])
    denom = clean_gss_numeric(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_numeric(df["region"]).where(clean_gss_numeric(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Models (Table 2 RHS, excluding Hispanic due to unavailability in provided extract)
    # ----------------------------
    # Keep the ordering aligned with the paper as much as possible.
    xcols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age",
        "black",
        # "hispanic",  # not usable here
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Ensure all predictors exist
    for c in xcols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # Quick diagnostics saved to file
    diag = []
    diag.append(("rows_year1993", int(df.shape[0])))
    diag.append(("nonmissing_dv_minority6", int(df["dislike_minority_genres"].notna().sum())))
    diag.append(("nonmissing_dv_other12", int(df["dislike_other12_genres"].notna().sum())))
    for c in xcols:
        diag.append((f"nonmissing_{c}", int(df[c].notna().sum())))
    diag_df = pd.DataFrame(diag, columns=["metric", "value"])
    diag_df.to_csv("./output/Table2_diagnostics.csv", index=False)
    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write(diag_df.to_string(index=False))

    # Fit both models
    tabA, fitA, modelA_u, modelA_z = standardized_ols(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    tabB, fitB, modelB_u, modelB_z = standardized_ols(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )

    # Overview text
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from provided GSS 1993 extract)\n")
        f.write("Standardized betas are estimated by OLS on z-scored DV and predictors; constant from unstandardized OLS.\n")
        f.write("IMPORTANT: 'Hispanic' predictor from the paper is not available in the provided variable list, so it is not included.\n\n")
        f.write("Model A: DV = count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin (0-6)\n")
        f.write(fitA.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
        f.write("\n\n")
        f.write("Model B: DV = count of dislikes among the remaining 12 genres (0-12)\n")
        f.write(fitB.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "Diagnostics": diag_df,
    }