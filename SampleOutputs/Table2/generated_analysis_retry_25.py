def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    # Conservative NA-code handling:
    # - For GSS-style small categorical vars, 8/9 often mean DK/NA.
    # - For larger numeric vars, 98/99/998/999/9998/9999 often mean DK/NA.
    # We apply both, but only after reading numeric.
    NA_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def clean_na(series):
        x = to_num(series)
        return x.mask(x.isin(list(NA_CODES)))

    def likert_dislike_indicator(series):
        """
        Music taste 1-5 scale. Dislike is 4/5, Like/neutral is 1/2/3.
        DK/NA/outside 1..5 -> missing.
        """
        x = clean_na(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete_case(df, items):
        """
        Sum of dislike indicators across listed music items.
        Require all component items observed (complete-case) for the DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardized_betas_from_unstd(y, X, b_unstd):
        """
        Standardized beta for each non-constant regressor:
        beta_j = b_j * sd(x_j) / sd(y)
        Uses estimation-sample SDs with ddof=0.
        """
        sd_y = np.std(y, ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            sd_x = np.std(X[col], ddof=0)
            if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                betas[col] = np.nan
            else:
                betas[col] = b_unstd[col] * (sd_x / sd_y)
        return betas

    def sig_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df_in, dv_col, x_cols, model_name):
        """
        Fit unstandardized OLS on DV (count), then compute standardized betas for slopes.
        Keep intercept unstandardized (to match Table 2 style).
        Listwise deletion over DV + RHS.
        """
        needed = [dv_col] + x_cols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_cols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = d[dv_col].astype(float)
        X = d[x_cols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        m = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(y.values, Xc, m.params)
        rows = []

        # Table in paper order: standardized slopes + stars, constant (unstandardized) + stars
        for col in x_cols:
            b = float(m.params.get(col, np.nan))
            p = float(m.pvalues.get(col, np.nan))
            rows.append(
                {
                    "term": col,
                    "beta_std": float(betas.get(col, np.nan)),
                    "b_unstd": b,
                    "p_value_reest": p,
                    "sig": sig_stars(p),
                }
            )
        # constant row
        rows.append(
            {
                "term": "const",
                "beta_std": np.nan,
                "b_unstd": float(m.params.get("const", np.nan)),
                "p_value_reest": float(m.pvalues.get("const", np.nan)),
                "sig": sig_stars(float(m.pvalues.get("const", np.nan))),
            }
        )

        tab = pd.DataFrame(rows).set_index("term")
        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m.nobs),
                    "k_predictors": int(m.df_model),  # excludes intercept
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                }
            ]
        )
        return m, tab, fit

    # -----------------------
    # Load and standardize names
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Year filter
    if "year" not in df.columns:
        raise ValueError("Required column missing: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Build DVs (complete-case on the DV items only)
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # -----------------------
    # Racism score (0-5)
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
    # RHS variables
    # -----------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_na(df["educ"]).where(clean_na(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # HH income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = clean_na(df["age"]).where(clean_na(df["age"]).between(18, 89))

    # Race indicators: Black, Hispanic, Other race.
    # Provided extract lacks a clean Hispanic flag; we approximate using ETHNIC as a proxy:
    # treat ETHNIC values typically corresponding to "Mexican, Puerto Rican, other Spanish"
    # as Hispanic. If ETHNIC is missing, hispanic is missing.
    # (This is a pragmatic approximation to avoid omitting Hispanic as required by Table 2.)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))

    df["black"] = np.where(race.notna(), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.notna(), (race == 3).astype(float), np.nan)

    # Hispanic proxy from ETHNIC (if present); else set missing
    if "ethnic" in df.columns:
        ethnic = clean_na(df["ethnic"])
        # Common GSS ETHNIC codes: 1=Mexican, 2=Puerto Rican, 3=Other Spanish.
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[ethnic.notna()] = ethnic.isin([1, 2, 3]).astype(float)
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan

    # Make race/ethnicity dummies mutually exclusive as in table concept:
    # If Hispanic==1, set black/other_race to 0 unless race indicates black/other explicitly.
    # (Keeps model identifiable; avoids double-counting Hispanic as "other".)
    # If race is missing, leave missing.
    mask_hisp1 = df["hispanic"] == 1
    mask_race_obs = race.notna()
    df.loc[mask_hisp1 & mask_race_obs, "other_race"] = 0.0  # avoid counting Hispanics as "other" by default

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    consprot.loc[relig.notna() & denom.notna()] = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    df["cons_protestant"] = consprot

    # No religion: RELIG == 4
    norelig = pd.Series(np.nan, index=df.index, dtype="float64")
    norelig.loc[relig.notna()] = (relig == 4).astype(float)
    df["no_religion"] = norelig

    # Southern: REGION == 3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    south = pd.Series(np.nan, index=df.index, dtype="float64")
    south.loc[region.notna()] = (region == 3).astype(float)
    df["south"] = south

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

    # Ensure predictors exist
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Constructed predictor missing: {c}")

    # Model A
    mA, tabA, fitA = fit_table2_model(df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6")

    # Model B
    mB, tabB, fitB = fit_table2_model(df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12")

    # -----------------------
    # Save outputs (human-readable)
    # -----------------------
    def write_model_outputs(model, tab, fit, prefix):
        with open(f"./output/{prefix}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n\nNotes:\n")
            f.write("- Standardized coefficients (beta_std) computed as b_unstd * SD(x)/SD(y) on estimation sample.\n")
            f.write("- p-values/stars are from this re-estimation (Table 2 in the paper does not report SEs).\n")

        # Pretty table
        t = tab.copy()
        # reorder to match paper: predictors then constant
        order = x_cols + ["const"]
        t = t.reindex(order)
        # format columns
        pretty = t[["beta_std", "sig", "b_unstd", "p_value_reest"]].copy()
        with open(f"./output/{prefix}_table.txt", "w", encoding="utf-8") as f:
            f.write(pretty.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

    write_model_outputs(mA, tabA, fitA, "Table2_ModelA")
    write_model_outputs(mB, tabB, fitB, "Table2_ModelB")

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS on count DV; standardized betas computed post-estimation.\n")
        f.write("Important: Hispanic indicator is approximated from ETHNIC codes (1/2/3) if ETHNIC is present in this extract.\n")
        f.write("If your data includes a direct Hispanic variable, replace this proxy for closer replication.\n\n")
        f.write("Model A: DV = count of dislikes among [rap, reggae, blues, jazz, gospel, latin]\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B: DV = count of dislikes among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }