def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def dislike_indicator(series):
        """
        1 if response is 4/5 (dislike/dislike very much),
        0 if response is 1/2/3,
        missing otherwise.
        """
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        """Map to {0,1}; anything else missing."""
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

    def dummy_eq(series, value):
        """Binary indicator; missing if input missing."""
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        m = x.notna()
        out.loc[m] = (x.loc[m] == value).astype(float)
        return out

    def zscore(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

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

    def fmt(x, nd=3):
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

    def safe_unique_count(s):
        s = to_num(s)
        return int(s.dropna().nunique())

    def labelled_missingness(dfin, cols, labels):
        miss = dfin[cols].isna().mean()
        out = pd.DataFrame(
            {"variable": cols, "label": [labels.get(c, c) for c in cols], "share_missing": miss.values}
        ).sort_values("share_missing", ascending=False)
        return out

    # -----------------------------
    # Variable lists (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    core_required = [
        "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race",
        "relig", "denom", "region"
    ] + minority_genres + remaining_genres + racism_items

    missing_cols = [c for c in core_required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Numeric coercion (keep id as-is)
    for c in core_required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables: strict dislike counts
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    dv1_col = "dv1_minority6_dislikes"
    dv2_col = "dv2_remaining12_dislikes"
    df[dv1_col] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df[dv2_col] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0–5) strict sum of 5 dichotomies (listwise on components)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Occupational prestige
    df["prestg80"] = to_num(df["prestg80"])

    # Female
    df["female"] = dummy_eq(df["sex"], 2)

    # Age
    df["age"] = to_num(df["age"])

    # Race dummies (reference: White)
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic: best possible from 'ethnic' if present; otherwise remain missing (and will be dropped if unusable)
    # We intentionally do NOT coerce missing to 0 (to preserve listwise logic).
    if "ethnic" in df.columns:
        df["ethnic"] = to_num(df["ethnic"])
        df["hispanic"] = pd.Series(np.nan, index=df.index, dtype="float64")
        m = df["ethnic"].notna()
        # Common recode in some extracts: ETHNIC==1 indicates Hispanic; otherwise 0.
        df.loc[m, "hispanic"] = (df.loc[m, "ethnic"] == 1).astype(float)
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy using available RELIG + DENOM (broad):
    # Use RELIG==1 (Protestant) & DENOM==1 (Baptist) as a conservative Protestant indicator.
    # Keep missing as missing (do not impute 0).
    df["cons_prot"] = pd.Series(np.nan, index=df.index, dtype="float64")
    m = df["relig"].notna() & df["denom"].notna()
    df.loc[m, "cons_prot"] = ((df.loc[m, "relig"] == 1) & (df.loc[m, "denom"] == 1)).astype(float)

    # No religion
    df["no_religion"] = dummy_eq(df["relig"], 4)

    # Southern (REGION==3)
    df["southern"] = dummy_eq(df["region"], 3)

    predictors_full = [
        "racism_score",
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    labels = {
        dv1_col: "Dislike of minority-linked genres (Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin) (count of 6)",
        dv2_col: "Dislike of the 12 remaining genres (count of 12)",
        "racism_score": "Racism score (0–5; strict sum of 5 dichotomies)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (from ETHNIC==1 if present; else missing)",
        "other_race": "Other race (1=other)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1; missing retained)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting
    # -----------------------------
    def fit_table2_model(dv_col, model_name):
        cols = [dv_col] + predictors_full
        d_raw = df[cols].copy()

        # Drop predictors that are 100% missing in the 1993 extract BEFORE listwise deletion
        usable_predictors = []
        dropped_all_missing = []
        for p in predictors_full:
            if d_raw[p].notna().sum() == 0:
                dropped_all_missing.append(p)
            else:
                usable_predictors.append(p)

        cols2 = [dv_col] + usable_predictors
        d = d_raw.dropna(subset=cols2).copy()

        # Drop non-varying predictors (avoid singular matrices)
        kept, dropped_no_var = [], []
        for p in usable_predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        # If still empty, write an error file and return
        if len(d) == 0 or (len(kept) == 0 and d[dv_col].notna().sum() == 0):
            miss = labelled_missingness(df, cols2, labels)
            miss_fmt = miss.copy()
            miss_fmt["share_missing"] = miss_fmt["share_missing"].map(lambda v: fmt(v, 3))
            msg = []
            msg.append(model_name)
            msg.append("=" * len(model_name))
            msg.append("")
            msg.append("ERROR: Analytic sample is empty after preprocessing + listwise deletion.")
            msg.append(f"DV: {labels.get(dv_col, dv_col)}")
            msg.append("")
            if dropped_all_missing:
                msg.append("Predictors dropped because they are 100% missing in the extract:")
                for p in dropped_all_missing:
                    msg.append(f"- {p}: {labels.get(p, p)}")
                msg.append("")
            msg.append("Missingness (share missing) among DV and remaining predictors (1993):")
            msg.append(miss_fmt.to_string(index=False))
            msg.append("")
            with open(f"./output/{model_name}_ERROR.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(msg) + "\n")

            table = pd.DataFrame(
                {
                    "Independent Variable": [labels.get(p, p) for p in predictors_full],
                    "Std_Beta": [np.nan] * len(predictors_full),
                    "Replication_Sig": [""] * len(predictors_full),
                    "Included_in_model": [False] * len(predictors_full),
                }
            )
            fit_stats = pd.DataFrame(
                {
                    "DV": [labels.get(dv_col, dv_col)],
                    "N": [0],
                    "R2": [np.nan],
                    "Adj_R2": [np.nan],
                    "Constant": [np.nan],
                    "Constant_Replication_Sig": [""],
                    "Dropped_all_missing": [", ".join(dropped_all_missing)],
                    "Dropped_no_variation": [", ".join(dropped_no_var)],
                },
                index=[model_name],
            )
            diag = pd.DataFrame(
                {
                    "variable": [dv_col] + predictors_full,
                    "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors_full],
                    "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors_full],
                    "unique_values_in_analytic_sample": [0] + [0] * len(predictors_full),
                }
            )
            table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
            fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)
            diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)
            return table, fit_stats, d, diag

        y = d[dv_col].astype(float)

        # Unstandardized OLS (R2, intercept, p-values)
        X_unstd = d[kept].astype(float) if len(kept) > 0 else pd.DataFrame(index=d.index)
        Xc = sm.add_constant(X_unstd, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas: regress z(Y) on z(X) without intercept
        betas = pd.Series(index=kept, dtype="float64")
        fit_beta = None
        if len(kept) > 0:
            y_z = zscore(y).rename("y_z")
            X_z = pd.DataFrame({p: zscore(d[p]) for p in kept}, index=d.index)
            dz = pd.concat([y_z, X_z], axis=1).dropna()
            if len(dz) > 0 and dz["y_z"].std(ddof=0) > 0:
                # Ensure no zero-variance standardized predictors
                if all(dz[p].std(ddof=0) > 0 for p in kept):
                    fit_beta = sm.OLS(dz["y_z"].astype(float), dz[kept].astype(float)).fit()
                    betas = fit_beta.params.reindex(kept)

        # Stars from replication p-values (unstandardized model)
        pvals = fit_unstd.pvalues if hasattr(fit_unstd, "pvalues") else pd.Series(dtype=float)
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        # Table in Table-2 order (include all predictors_full, marking inclusion)
        rows = []
        for p in predictors_full:
            inc = p in kept
            rows.append(
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": float(betas.get(p, np.nan)) if inc else np.nan,
                    "Replication_Sig": stars.get(p, "") if inc else "",
                    "Included_in_model": bool(inc),
                }
            )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(getattr(fit_unstd, "rsquared", np.nan))],
                "Adj_R2": [float(getattr(fit_unstd, "rsquared_adj", np.nan))],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Replication_Sig": [const_star],
                "Dropped_all_missing": [", ".join(dropped_all_missing) if dropped_all_missing else ""],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        # Diagnostics
        diag = pd.DataFrame(
            {
                "variable": [dv_col] + predictors_full,
                "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors_full],
                "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors_full],
                "unique_values_in_analytic_sample": [safe_unique_count(d[dv_col])] + [safe_unique_count(d.get(p, pd.Series(dtype=float))) for p in predictors_full],
            }
        )

        # Save outputs
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS.")
        lines.append("Reported coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: regress z(Y) on z(X) with no intercept (analytic listwise sample).")
        lines.append("Stars: two-tailed p-values from unstandardized OLS (replication stars).")
        lines.append("")
        lines.append("Construction notes:")
        lines.append("- Dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV construction: strict sum across component genres (missing if any component missing within that DV).")
        lines.append("- Racism scale: strict sum of 5 dichotomies (missing if any component missing).")
        lines.append("- Missing data: listwise deletion on DV + included predictors (after dropping predictors that are 100% missing).")
        if dropped_all_missing:
            lines.append("")
            lines.append("Dropped predictors (100% missing in extract): " + ", ".join(dropped_all_missing))
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped predictors (no variation in analytic sample): " + ", ".join(dropped_no_var))
        lines.append("")
        lines.append("Standardized coefficients (Table-2 style; replication)")
        lines.append("-----------------------------------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp.to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS; replication)")
        lines.append("----------------------------------------------")
        fs = fit_stats[
            ["N", "R2", "Adj_R2", "Constant", "Constant_Replication_Sig", "Dropped_all_missing", "Dropped_no_variation"]
        ].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        with open(f"./output/{model_name}_ols_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write("Unstandardized OLS (fit stats + p-values):\n")
            f.write(fit_unstd.summary().as_text())
            f.write("\n\n")
            if fit_beta is not None:
                f.write("Standardized-beta regression (zY on zX, no intercept):\n")
                f.write(fit_beta.summary().as_text())
                f.write("\n")
            else:
                f.write("Standardized-beta regression not estimated (insufficient usable predictors after standardization).\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)
        diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)

        return table, fit_stats, d, diag

    # Run models
    m1_table, m1_fit, m1_frame, m1_diag = fit_table2_model(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame, m2_diag = fit_table2_model(dv2_col, "Table2_ModelB_Remaining12")

    # Combined summary outputs
    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "ModelA_Std_Beta": m1_table["Std_Beta"],
            "ModelA_Replication_Sig": m1_table["Replication_Sig"],
            "ModelB_Std_Beta": m2_table["Std_Beta"],
            "ModelB_Replication_Sig": m2_table["Replication_Sig"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # Missingness tables (pre-listwise)
    miss_A = labelled_missingness(df, [dv1_col] + predictors_full, labels)
    miss_B = labelled_missingness(df, [dv2_col] + predictors_full, labels)

    # Human-readable combined summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Implementation summary")
    lines.append("----------------------")
    lines.append("- Year filter: year == 1993")
    lines.append("- DVs: strict dislike counts (missing if any component genre rating missing within the DV set)")
    lines.append("- Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing")
    lines.append("- Racism score: strict sum of 5 dichotomies (0–5; missing if any component missing)")
    lines.append("- Estimation: OLS; standardized coefficients computed as z(Y) on z(X), no intercept")
    lines.append("- Regression missing data: listwise deletion after dropping predictors that are 100% missing in the extract")
    lines.append("")
    lines.append("Combined standardized coefficients (replication)")
    lines.append("----------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; replication)")
    lines.append("----------------------------------------------")
    fit_disp = combined_fit.copy()
    for col in ["N"]:
        fit_disp[col] = fit_disp[col].map(lambda v: fmt(v, 0))
    for col in ["R2", "Adj_R2", "Constant"]:
        fit_disp[col] = fit_disp[col].map(lambda v: fmt(v, 3))
    lines.append(fit_disp.to_string())
    lines.append("")
    lines.append("Missingness shares (Model A variables; before listwise)")
    lines.append("------------------------------------------------------")
    missA = miss_A.copy()
    missA["share_missing"] = missA["share_missing"].map(lambda v: fmt(v, 3))
    lines.append(missA.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares (Model B variables; before listwise)")
    lines.append("------------------------------------------------------")
    missB = miss_B.copy()
    missB["share_missing"] = missB["share_missing"].map(lambda v: fmt(v, 3))
    lines.append(missB.to_string(index=False))
    lines.append("")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss_A.to_csv("./output/missingness_modelA_labelled.csv", index=False)
    miss_B.to_csv("./output/missingness_modelB_labelled.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "missingness_modelA": miss_A,
        "missingness_modelB": miss_B,
        "modelA_analytic_frame": m1_frame,
        "modelB_analytic_frame": m2_frame,
        "modelA_diagnostics": m1_diag,
        "modelB_diagnostics": m2_diag,
    }