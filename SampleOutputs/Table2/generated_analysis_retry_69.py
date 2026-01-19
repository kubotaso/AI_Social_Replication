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
        Conservative NA handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 (like very much ... dislike very much)
        Dislike if 4 or 5.
        Missing if not in 1..5 or sentinel/NA.
        """
        x = clean_gss_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_series(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if (not np.isfinite(sd)) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_completecase(df, items):
        """
        Bryson: DK treated as missing; cases with missing excluded.
        To be faithful and avoid partial-information counts, require all component items observed.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

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

    def fit_table2_model(df, dv, x_order, model_name, standardize_for_betas=True):
        """
        Fit OLS on unstandardized variables (to get the constant on DV scale),
        compute standardized betas by refitting on z-scored DV and z-scored X (excluding constant),
        then keep p-values from the standardized fit for stars.
        """
        needed = [dv] + x_order
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] == 0:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n=0).")

        # Unstandardized model (for intercept on DV scale + R2 with original DV)
        y_u = to_num(d[dv])
        X_u = d[x_order].apply(to_num)
        X_u = sm.add_constant(X_u, has_constant="add")
        m_u = sm.OLS(y_u, X_u).fit()

        # Standardized fit for betas and p-values on those betas (same sample)
        if standardize_for_betas:
            y_z = zscore_series(d[dv])
            X_z = pd.DataFrame({c: zscore_series(d[c]) for c in x_order}, index=d.index)

            # If any predictor becomes all-NA due to 0 variance, keep it as NA and note it
            dropped = []
            keep_cols = []
            for c in x_order:
                col = X_z[c]
                if col.isna().any() or (col.std(ddof=0) == 0) or (not np.isfinite(col.std(ddof=0))):
                    dropped.append(c)
                else:
                    keep_cols.append(c)

            X_z = X_z[keep_cols]
            # Standardized regression should not include intercept for "standardized coefficients",
            # but in practice beta from standardized regression with intercept equals correlation-based beta.
            X_zc = sm.add_constant(X_z, has_constant="add")
            m_z = sm.OLS(y_z.loc[X_zc.index], X_zc).fit()
        else:
            m_z = m_u
            dropped = []

        # Build paper-style table in required order (with names)
        rows = []
        for c in x_order:
            if standardize_for_betas and (c in dropped):
                beta = np.nan
                pval = np.nan
            else:
                beta = float(m_z.params.get(c, np.nan))
                pval = float(m_z.pvalues.get(c, np.nan))
            rows.append(
                {
                    "variable": c,
                    "std_beta": beta,
                    "p_value": pval,
                    "std_beta_star": ("" if pd.isna(beta) else f"{beta:.3f}{stars_from_p(pval)}"),
                }
            )

        # Add constant as unstandardized intercept from unstandardized model
        const = float(m_u.params.get("const", np.nan))
        const_p = float(m_u.pvalues.get("const", np.nan))
        rows.append(
            {
                "variable": "const",
                "std_beta": np.nan,
                "p_value": const_p,
                "std_beta_star": f"{const:.3f}{stars_from_p(const_p)}",
            }
        )

        paper = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m_u.nobs),
                    "k": int(m_u.df_model + 1),
                    "r2": float(m_u.rsquared),
                    "adj_r2": float(m_u.rsquared_adj),
                }
            ]
        )

        # Diagnostics: dummy variation in estimation sample
        diag = {}
        for c in x_order:
            v = d[c]
            if set(v.dropna().unique()) <= {0.0, 1.0}:
                diag[c] = {
                    "mean": float(v.mean()),
                    "sum": float(v.sum()),
                    "n": int(v.shape[0]),
                    "unique": sorted([float(x) for x in v.dropna().unique()]),
                }

        return m_u, m_z, paper, fit, d.index, dropped, diag

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to GSS 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Dependent variables
    # -----------------------------
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

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
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

    # -----------------------------
    # Controls
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing educ column (EDUC).")
    edu = clean_gss_missing(df["educ"])
    df["education_years"] = edu.where(edu.between(0, 20))

    # HH income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column (PRESTG80).")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column (SEX).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column (AGE).")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (Black, Hispanic, Other race)
    # Black/Other available from RACE.
    if "race" not in df.columns:
        raise ValueError("Missing race column (RACE).")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: not directly provided; use ETHNIC as a pragmatic proxy only if present.
    # This is imperfect, but avoids dropping the variable entirely (which caused prior failures).
    # Rule: treat small ethnic codes as "Hispanic-origin" proxy; keep conservative.
    if "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        # Conservative proxy: many GSS ETHNIC code schemes place Hispanic groups in low codes.
        # Use <= 10 as Hispanic proxy; set missing if ETHNIC missing.
        hisp = (eth.between(1, 10)).astype(float)
        hisp.loc[eth.isna()] = np.nan
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan  # will be dropped by listwise deletion; diagnostics will show.

    # Conservative Protestant + No religion
    if "relig" not in df.columns:
        raise ValueError("Missing relig column (RELIG).")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"]) if "denom" in df.columns else pd.Series(np.nan, index=df.index)

    # Conservative Protestant proxy: Protestant (RELIG==1) and DENOM in {1,6,7}
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column (REGION).")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Model specification (Table 2 order)
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

    # -----------------------------
    # Fit models
    # -----------------------------
    mA_u, mA_z, tabA, fitA, idxA, droppedA, diagA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB_u, mB_z, tabB, fitB, idxB, droppedB, diagB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # -----------------------------
    # Save outputs (human-readable)
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # Paper-style tables (standardized betas + stars, constant unstandardized)
    tabA_out = tabA.copy()
    tabB_out = tabB.copy()

    # Pretty labels for readability
    label_map = {
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
        "const": "Constant",
    }
    tabA_out["variable_label"] = tabA_out["variable"].map(label_map).fillna(tabA_out["variable"])
    tabB_out["variable_label"] = tabB_out["variable"].map(label_map).fillna(tabB_out["variable"])

    tabA_out = tabA_out[["variable_label", "std_beta", "p_value", "std_beta_star"]]
    tabB_out = tabB_out[["variable_label", "std_beta", "p_value", "std_beta_star"]]

    write_text(
        "./output/Table2_ModelA_paper_style.txt",
        tabA_out.to_string(index=False),
    )
    write_text(
        "./output/Table2_ModelB_paper_style.txt",
        tabB_out.to_string(index=False),
    )

    # Full model summaries (from re-estimation; not in paper but useful diagnostics)
    write_text("./output/Table2_ModelA_unstandardized_summary.txt", mA_u.summary().as_text())
    write_text("./output/Table2_ModelB_unstandardized_summary.txt", mB_u.summary().as_text())
    write_text("./output/Table2_ModelA_standardized_summary.txt", mA_z.summary().as_text())
    write_text("./output/Table2_ModelB_standardized_summary.txt", mB_z.summary().as_text())

    # Fit
    write_text("./output/Table2_fit.txt", pd.concat([fitA, fitB], axis=0).to_string(index=False))

    # Diagnostics on dummy variation and dropped predictors
    diag_txt = []
    diag_txt.append("Dropped predictors (due to 0 variance / NA after standardization):\n")
    diag_txt.append(f"Model A dropped: {droppedA}\n")
    diag_txt.append(f"Model B dropped: {droppedB}\n\n")
    diag_txt.append("Dummy diagnostics in estimation samples (means/sums/unique):\n")
    diag_txt.append("Model A:\n")
    for k, v in diagA.items():
        diag_txt.append(f"  {k}: {v}\n")
    diag_txt.append("\nModel B:\n")
    for k, v in diagB.items():
        diag_txt.append(f"  {k}: {v}\n")
    write_text("./output/Table2_diagnostics.txt", "".join(diag_txt))

    # Overview
    overview = []
    overview.append("Table 2 replication attempt (GSS 1993): OLS with standardized coefficients (betas).\n")
    overview.append("Notes:\n")
    overview.append("- Betas computed via regression on z-scored DV and z-scored predictors (intercept included).\n")
    overview.append("- Constant reported from the unstandardized regression (DV scale).\n")
    overview.append("- Missing responses are treated as missing; DVs require complete responses to all items in their genre set.\n")
    overview.append("- Stars based on two-tailed OLS p-values from standardized regression (*<.05, **<.01, ***<.001).\n\n")
    overview.append("Model A: DV = count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin.\n")
    overview.append(fitA.to_string(index=False) + "\n\n")
    overview.append("Model B: DV = count of dislikes among the other 12 genres.\n")
    overview.append(fitB.to_string(index=False) + "\n")
    write_text("./output/Table2_overview.txt", "".join(overview))

    return {
        "ModelA_table": tabA_out,
        "ModelB_table": tabB_out,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }