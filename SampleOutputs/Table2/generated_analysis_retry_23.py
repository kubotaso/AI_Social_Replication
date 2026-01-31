def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + normalize columns
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    if "id" not in df.columns:
        raise ValueError("Expected column 'id' in the input CSV.")

    df = df.loc[df["year"] == 1993].copy()

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    # Convert all except id to numeric where possible
    for c in df.columns:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Helpers
    # -----------------------------
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
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

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

    def z_beta_from_unstd(unstd_params, xcols, d_analytic, ycol):
        """
        Standardized betas computed from unstandardized OLS slopes:
            beta_std_j = b_j * sd(x_j) / sd(y)
        Uses sample SD (ddof=1) on the SAME analytic sample.
        """
        y_sd = d_analytic[ycol].astype(float).std(ddof=1)
        betas = {}
        for c in xcols:
            x_sd = d_analytic[c].astype(float).std(ddof=1)
            b = unstd_params.get(c, np.nan)
            if pd.isna(b) or pd.isna(x_sd) or pd.isna(y_sd) or x_sd == 0 or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(b) * float(x_sd) / float(y_sd)
        return pd.Series(betas)

    def stepwise_case_counts(dfin, required_cols, title):
        """
        Counts remaining N after cumulatively requiring non-missing for each variable in order.
        Helps diagnose N collapse and "no_religion dropped" issues.
        """
        rows = []
        mask = pd.Series(True, index=dfin.index)
        rows.append({"step": "start (YEAR==1993)", "N": int(mask.sum())})
        for col in required_cols:
            mask = mask & dfin[col].notna()
            rows.append({"step": f"+ non-missing {col}", "N": int(mask.sum())})
        out = pd.DataFrame(rows)
        out.to_csv(f"./output/{title}_case_count_steps.csv", index=False)
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

    required_cols = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # -----------------------------
    # Dependent variables (STRICT, per Table 2 summary)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    dv1_col = "dv1_minority6_dislikes"
    dv2_col = "dv2_remaining12_dislikes"
    dv1_items = [f"d_{c}" for c in minority_genres]
    dv2_items = [f"d_{c}" for c in remaining_genres]

    # Strict sums: missing if any genre missing (matches summary/instructions)
    df[dv1_col] = strict_sum(df, dv1_items)
    df[dv2_col] = strict_sum(df, dv2_items)

    # -----------------------------
    # Racism score (0–5): STRICT sum of 5 dichotomies; missing if any missing
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no (not discrimination) -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no (not education chance) -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes (willpower) -> 1

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # 0..5 if complete

    # -----------------------------
    # Controls / indicators (NO "missing to 0" imputation; preserve missing for listwise deletion)
    # -----------------------------
    df["education"] = df["educ"]

    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies (reference = White)
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic from ETHNIC (best available in this extract; do NOT force missing to 0)
    df["hispanic"] = np.where(df["ethnic"].notna(), (df["ethnic"] == 1).astype(float), np.nan)

    # Conservative Protestant proxy (limited): RELIG==1 (protestant) & DENOM==1 (baptist)
    # Keep missing as missing for faithful listwise deletion.
    df["cons_prot"] = np.where(
        df["relig"].notna() & df["denom"].notna(),
        ((df["relig"] == 1) & (df["denom"] == 1)).astype(float),
        np.nan,
    )

    # No religion
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
        "racism_score",
        "education",
        "income_pc",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    present_labels = {
        "racism_score": "Racism score",
        "education": "Education",
        "income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant",
        dv1_col: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count)",
        dv2_col: "Dislike of the 12 Remaining Genres (count)",
    }

    # -----------------------------
    # Model fitting (faithful listwise)
    # -----------------------------
    def fit_model(dv_col, model_name):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # Drop no-variation predictors to avoid singular matrices, but record them
        kept, dropped_no_var = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas (Table 2 style): convert from unstandardized slopes
        betas = z_beta_from_unstd(fit_unstd.params, kept, d, dv_col)

        # Build coefficient table aligned to full predictor list
        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": present_labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit_unstd.pvalues.get(p, np.nan)),
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": present_labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Sig": "",
                    }
                )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [present_labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [star_from_p(fit_unstd.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        # Save human-readable table (no SEs printed; table 2 doesn't show them)
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {present_labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS regression.")
        lines.append("Coefficients shown: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_std_j = b_j * SD(X_j) / SD(Y), computed on the analytic (listwise) sample; SD uses ddof=1.")
        lines.append("Stars: two-tailed p-values from the unstandardized OLS regression (replication-derived).")
        lines.append("")
        lines.append("Coding/Construction:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV construction: STRICT count; missing if any component genre missing")
        lines.append("- Racism score: STRICT sum of 5 dichotomies; missing if any component missing (0–5)")
        lines.append("- Missing data in regression: listwise deletion on DV + all predictors used")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped predictors due to no variation in analytic sample:")
            for p in dropped_no_var:
                lines.append(f"- {p}: {present_labels.get(p, p)}")

        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp.to_string(index=False))

        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())

        # Also save full regression summary separately for debugging
        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(f"./output/{model_name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit_unstd.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Diagnostics: value counts for key dummies within analytic sample (to catch "dropped no_religion")
        diag_lines = []
        diag_lines.append(f"{model_name} analytic sample diagnostics")
        diag_lines.append("=" * (len(model_name) + 28))
        diag_lines.append(f"N analytic: {len(d)}")
        diag_lines.append("")
        for v in ["hispanic", "no_religion", "cons_prot", "black", "other_race", "southern"]:
            if v in d.columns:
                vc = d[v].value_counts(dropna=False).sort_index()
                diag_lines.append(f"{v} value counts (analytic):")
                diag_lines.append(vc.to_string())
                diag_lines.append("")
        with open(f"./output/{model_name}_analytic_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(diag_lines).strip() + "\n")

        return table, fit_stats, d

    # Stepwise N collapse diagnostics (key fix for previous runtime/sample issues)
    _steps_A = stepwise_case_counts(df, [dv1_col] + predictors, "ModelA")
    _steps_B = stepwise_case_counts(df, [dv2_col] + predictors, "ModelB")

    m1_table, m1_fit, m1_frame = fit_model(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame = fit_model(dv2_col, "Table2_ModelB_Remaining12")

    # -----------------------------
    # Combined outputs
    # -----------------------------
    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "ModelA_Std_Beta": m1_table["Std_Beta"],
            "ModelA_Sig": m1_table["Sig"],
            "ModelB_Std_Beta": m2_table["Std_Beta"],
            "ModelB_Sig": m2_table["Sig"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # DV descriptives before listwise deletion (strict DVs)
    dv_desc = pd.DataFrame(
        {
            "stat": ["N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"],
            "DV1": [
                int(df[dv1_col].notna().sum()),
                df[dv1_col].mean(),
                df[dv1_col].std(ddof=0),
                df[dv1_col].min(),
                df[dv1_col].quantile(0.25),
                df[dv1_col].quantile(0.50),
                df[dv1_col].quantile(0.75),
                df[dv1_col].max(),
            ],
            "DV2": [
                int(df[dv2_col].notna().sum()),
                df[dv2_col].mean(),
                df[dv2_col].std(ddof=0),
                df[dv2_col].min(),
                df[dv2_col].quantile(0.25),
                df[dv2_col].quantile(0.50),
                df[dv2_col].quantile(0.75),
                df[dv2_col].max(),
            ],
        }
    )

    # Missingness shares in 1993 before listwise
    miss = df[[dv1_col, dv2_col] + predictors].isna().mean().reset_index()
    miss.columns = ["variable", "share_missing_1993"]
    miss["label"] = miss["variable"].map(lambda v: present_labels.get(v, v))
    miss = miss.sort_values("share_missing_1993", ascending=False)

    # Frequency checks in 1993 (raw, not analytic)
    freq_vars = ["race", "ethnic", "relig", "denom", "region"]
    freq_lines = []
    for v in freq_vars:
        if v in df.columns:
            vc = df[v].value_counts(dropna=False).sort_index()
            freq_lines.append(f"{v} value counts (1993):")
            freq_lines.append(vc.to_string())
            freq_lines.append("")

    # Human-readable combined summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (computed from provided 1993 GSS extract)")
    lines.append("================================================================================")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) with replication-derived stars")
    lines.append("-------------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; replication)")
    lines.append("----------------------------------------------")
    fd = combined_fit.copy()
    fd["N"] = fd["N"].map(lambda v: fmt(v, 0))
    for c in ["R2", "Adj_R2", "Constant"]:
        fd[c] = fd[c].map(lambda v: fmt(v, 3))
    lines.append(fd.to_string())
    lines.append("")
    lines.append("DV descriptives (constructed counts; strict; before listwise deletion)")
    lines.append("---------------------------------------------------------------------")
    dv_disp = dv_desc.copy()
    dv_disp["DV1"] = dv_disp["DV1"].map(lambda v: str(int(v)) if isinstance(v, (int, np.integer)) else fmt(v, 3))
    dv_disp["DV2"] = dv_disp["DV2"].map(lambda v: str(int(v)) if isinstance(v, (int, np.integer)) else fmt(v, 3))
    lines.append(dv_disp.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares in 1993 (before listwise deletion)")
    lines.append("----------------------------------------------------")
    miss_disp = miss.copy()
    miss_disp["share_missing_1993"] = miss_disp["share_missing_1993"].map(lambda v: fmt(v, 3))
    lines.append(miss_disp.to_string(index=False))
    lines.append("")
    lines.append("Analytic sample sizes (after listwise deletion)")
    lines.append("----------------------------------------------")
    lines.append(f"Model A analytic N: {len(m1_frame)}")
    lines.append(f"Model B analytic N: {len(m2_frame)}")
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("- Standard errors are not printed in Table 2; this script saves full OLS summaries separately for debugging.")
    lines.append("- Hispanic is constructed from ETHNIC==1 (best available in this extract); if this differs from the original GSS Hispanic flag, exact replication may differ.")
    lines.append("- Conservative Protestant uses a limited proxy (RELIG==1 & DENOM==1) because finer religious tradition codes are not present in this extract.")
    lines.append("- All missing are left as missing; no 'missing to 0' imputation is used (to match listwise deletion behavior).")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss.to_csv("./output/missingness_1993.csv", index=False)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "missingness_1993": miss,
        "dv_descriptives": dv_desc,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_fit": m1_fit,
        "modelB_fit": m2_fit,
        "modelA_analytic_frame": m1_frame,
        "modelB_analytic_frame": m2_frame,
        "modelA_case_count_steps": _steps_A,
        "modelB_case_count_steps": _steps_B,
    }