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

    def clean_gss_na(x):
        """
        Conservative missing-value handler for this extract:
        - Coerce to numeric
        - Drop common GSS NA codes (7/8/9, 97/98/99, 997/998/999, 9997/9998/9999)
        - Keep everything else
        """
        x = to_num(x).copy()
        na_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}
        x = x.mask(x.isin(list(na_codes)))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5. Code 1 if 4/5 (dislike/dislike very much),
        0 if 1/2/3, missing otherwise.
        """
        x = clean_gss_na(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_na(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete_case(df, items):
        """
        Build count DV as sum of binary dislike indicators.
        Complete-case across the items (DK treated as missing; cases with missing excluded).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        # require all items present
        return mat.sum(axis=1, min_count=len(items))

    def standardize_series(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def compute_standardized_betas_from_unstd(model, d, ycol, xcols):
        """
        Standardized beta for slope j: beta_j = b_j * sd(x_j) / sd(y)
        (computed on the model estimation sample d).
        Intercept is left unstandardized (as in typical 'standardized coefficients' tables).
        """
        y_sd = d[ycol].std(ddof=0)
        betas = {}
        for c in xcols:
            if c in model.params.index:
                x_sd = d[c].std(ddof=0)
                if np.isfinite(x_sd) and x_sd > 0 and np.isfinite(y_sd) and y_sd > 0:
                    betas[c] = model.params[c] * (x_sd / y_sd)
                else:
                    betas[c] = np.nan
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

    def fit_table2_model(df, ycol, xcols, model_name, dv_label):
        """
        - Listwise deletion on y and all Xs (as typical for published OLS tables)
        - OLS on unstandardized variables
        - Report standardized betas for slopes + unstandardized constant
        """
        needed = [ycol] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Drop zero-variance predictors (but keep a record)
        dropped_zero_var = []
        x_keep = []
        for c in xcols:
            v = d[c].std(ddof=0)
            if not np.isfinite(v) or v == 0:
                dropped_zero_var.append(c)
            else:
                x_keep.append(c)

        if d.shape[0] < (len(x_keep) + 5):
            raise ValueError(
                f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}), "
                f"predictors kept={len(x_keep)}, dropped_zero_var={dropped_zero_var}."
            )

        y = d[ycol].astype(float)
        X = sm.add_constant(d[x_keep].astype(float), has_constant="add")
        m = sm.OLS(y, X).fit()

        betas = compute_standardized_betas_from_unstd(m, d, ycol, x_keep)

        # Paper-style table: standardized betas (slopes) + constant (unstandardized)
        order_rows = [
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
        # Build standardized beta column aligned to full intended order (NaN if not estimated)
        beta_full = pd.Series({c: np.nan for c in order_rows}, dtype="float64")
        for c in betas.index:
            beta_full.loc[c] = betas.loc[c]

        # p-values for slopes come from unstandardized b tests (same inference)
        p_full = pd.Series({c: np.nan for c in order_rows}, dtype="float64")
        for c in order_rows:
            if c in m.pvalues.index:
                p_full.loc[c] = float(m.pvalues.loc[c])

        star_full = p_full.apply(stars_from_p)

        paper_tbl = pd.DataFrame(
            {
                "std_beta": beta_full,
                "std_beta_star": beta_full.map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + star_full,
                "p_value": p_full,
            }
        )
        # Add constant separately (unstandardized)
        const_val = float(m.params.get("const", np.nan))
        const_p = float(m.pvalues.get("const", np.nan))
        const_star = stars_from_p(const_p)
        const_row = pd.DataFrame(
            {
                "std_beta": [np.nan],
                "std_beta_star": [f"{const_val:.3f}{const_star}"],
                "p_value": [const_p],
            },
            index=["constant_unstd"],
        )
        paper_tbl = pd.concat([paper_tbl, const_row], axis=0)

        # Full (replication) table with b/SE/t/p (explicitly computed, not from paper)
        full_tbl = pd.DataFrame(
            {
                "b": m.params,
                "std_err": m.bse,
                "t": m.tvalues,
                "p_value": m.pvalues,
            }
        )
        # Add standardized beta for estimated slopes into full table for convenience
        full_tbl["std_beta"] = np.nan
        for c in betas.index:
            if c in full_tbl.index:
                full_tbl.loc[c, "std_beta"] = float(betas.loc[c])

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_label,
                    "n": int(m.nobs),
                    "k_including_const": int(m.df_model + 1),
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped_zero_var) if dropped_zero_var else "",
                }
            ]
        )

        # Diagnostics: quick frequency for key dummies in the estimation sample
        diag_cols = ["female", "black", "hispanic", "other_race", "cons_protestant", "no_religion", "south"]
        diag = {}
        for c in diag_cols:
            if c in d.columns:
                vc = d[c].value_counts(dropna=False).to_dict()
                diag[c] = vc
        diag_txt = "\n".join([f"{k}: {v}" for k, v in diag.items()])

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\nDV: {dv_label}\n\n")
            f.write(m.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Standardized betas (slopes) computed as b * sd(x)/sd(y) on the estimation sample.\n")
            f.write("- Constant reported as unstandardized intercept.\n")
            f.write("- p-values and stars are from this re-estimation (not extracted from the paper).\n")
            if dropped_zero_var:
                f.write(f"- Dropped zero-variance predictors: {', '.join(dropped_zero_var)}\n")
            f.write("\nEstimation-sample dummy distributions (value_counts):\n")
            f.write(diag_txt)
            f.write("\n")

        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            f.write("Table 2-style output: standardized coefficients (betas) for slopes + unstandardized constant\n")
            f.write("(Stars based on p-values from this re-estimation.)\n\n")
            f.write(paper_tbl.to_string(float_format=lambda x: f"{x:.6g}"))
            f.write("\n")

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write("Replication regression output (computed from microdata): unstandardized b, SE, t, p, and std_beta\n\n")
            f.write(full_tbl.to_string(float_format=lambda x: f"{x:.6g}"))
            f.write("\n")

        return m, paper_tbl, full_tbl, fit, d.index

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

    # ----------------------------
    # DVs (counts of dislikes)
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # ----------------------------
    # Racism score (0-5)
    # ----------------------------
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

    # ----------------------------
    # Controls
    # ----------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_na(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita = REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing column(s) for income per capita: realinc and/or hompop")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(lambda x: x > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_gss_na(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies from RACE (no Hispanic field available in provided extract)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_na(df["race"]).where(lambda x: x.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in the provided variables -> keep as missing (will reduce N if included)
    # To keep the model runnable and avoid collapsing N to ~0, we omit it from estimation
    # but we still compute and report it as unavailable.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing column(s): relig and/or denom")
    relig = clean_gss_na(df["relig"])
    denom = clean_gss_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4 (None)
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna(), np.nan)
    df["no_religion"] = norelig

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_na(df["region"]).where(lambda x: x.isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Fit models (Table 2): same RHS
    # Note: Hispanic is not available in this dataset extract; including it would force N to 0.
    # We keep it out of the estimation X list but we keep a placeholder in output tables.
    # ----------------------------
    xcols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # unavailable; would zero out sample via listwise deletion
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    results = {}

    mA, paperA, fullA, fitA, idxA = fit_table2_model(
        df,
        "dislike_minority_genres",
        xcols,
        "Table2_ModelA_dislike_Rap_Reggae_Blues_Jazz_Gospel_Latin",
        "Dislike count: Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin",
    )

    mB, paperB, fullB, fitB, idxB = fit_table2_model(
        df,
        "dislike_other12_genres",
        xcols,
        "Table2_ModelB_dislike_other12_genres",
        "Dislike count: 12 remaining genres (excluding Rap, Reggae, Blues, Jazz, Gospel, Latin)",
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (computed from provided 1993 GSS extract)\n\n")
        f.write("Important implementation notes:\n")
        f.write("- Standardized coefficients (betas) are computed from unstandardized OLS slopes via b*sd(x)/sd(y).\n")
        f.write("- Intercept is reported unstandardized.\n")
        f.write("- Stars are based on p-values from this re-estimation; the paper does not report SEs.\n")
        f.write("- Hispanic dummy is not estimable because no direct Hispanic identifier exists in the provided variable list.\n")
        f.write("  (Therefore, models here omit Hispanic; paper includes it.)\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results["ModelA_paper_style"] = paperA
    results["ModelB_paper_style"] = paperB
    results["ModelA_full"] = fullA
    results["ModelB_full"] = fullB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    return results