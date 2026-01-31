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

    # Coerce all but id to numeric (keeps NA as NaN)
    for c in df.columns:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Column presence checks
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    # Note: ETHNIC is present in provided extract; use it to construct Hispanic.
    # RELIG/DENOM are present; we implement the best-available cons_prot proxy,
    # but do NOT impute missing to 0 (listwise deletion like the paper).
    required = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

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

    def standardized_betas_from_unstd(fit, d_analytic, y_col, x_cols):
        """
        Compute standardized betas from unstandardized slopes:
            beta_std_j = b_j * SD(X_j) / SD(Y)
        using ddof=1 on the analytic sample.
        """
        y = d_analytic[y_col].astype(float)
        y_sd = y.std(ddof=1)
        betas = {}
        for c in x_cols:
            x = d_analytic[c].astype(float)
            x_sd = x.std(ddof=1)
            b = fit.params.get(c, np.nan)
            if pd.isna(b) or pd.isna(x_sd) or pd.isna(y_sd) or x_sd == 0 or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(b) * float(x_sd) / float(y_sd)
        return pd.Series(betas, index=x_cols)

    def write_text(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    # -----------------------------
    # Construct DVs (STRICT; missing if any component missing)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1_col = "dv1_minority6_dislikes"
    dv2_col = "dv2_remaining12_dislikes"
    dv1_items = [f"d_{g}" for g in minority_genres]
    dv2_items = [f"d_{g}" for g in remaining_genres]

    df[dv1_col] = strict_sum(df, dv1_items)  # 0..6
    df[dv2_col] = strict_sum(df, dv2_items)  # 0..12

    # -----------------------------
    # Racism score (0-5), STRICT sum (missing if any missing)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no edu chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # 0..5 if complete

    # -----------------------------
    # Controls / indicators (preserve missing; listwise deletion later)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # -----------------------------
    # Race/ethnicity: mutually exclusive dummies (White non-Hispanic reference)
    # Using ETHNIC==1 as Hispanic flag in this extract.
    # If ETHNIC missing -> race/ethnicity dummies missing (forces listwise deletion like the paper).
    # -----------------------------
    hisp = np.where(df["ethnic"].notna(), (df["ethnic"] == 1).astype(float), np.nan)  # best available
    df["hispanic"] = hisp

    race_ok = df["race"].isin([1, 2, 3])
    # Initialize as missing when we can't classify
    df["black"] = np.nan
    df["other_race"] = np.nan

    can_classify = race_ok & df["hispanic"].notna()
    # Hispanic overrides race to make categories mutually exclusive (common practice)
    df.loc[can_classify, "black"] = np.where(
        df.loc[can_classify, "hispanic"] == 1.0, 0.0, (df.loc[can_classify, "race"] == 2).astype(float)
    )
    df.loc[can_classify, "other_race"] = np.where(
        df.loc[can_classify, "hispanic"] == 1.0, 0.0, (df.loc[can_classify, "race"] == 3).astype(float)
    )

    # -----------------------------
    # Religion: best-available conservative Protestant proxy; preserve missing
    # -----------------------------
    df["cons_prot"] = np.where(
        df["relig"].notna() & df["denom"].notna(),
        ((df["relig"] == 1) & (df["denom"] == 1)).astype(float),
        np.nan,
    )

    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)
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

    labels = {
        dv1_col: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count)",
        dv2_col: "Dislike of the 12 Remaining Genres (count)",
        "racism_score": "Racism score (0–5)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age_years": "Age (years)",
        "black": "Black (mutually exclusive; Hispanic overrides)",
        "hispanic": "Hispanic (ETHNIC==1; mutually exclusive)",
        "other_race": "Other race (mutually exclusive; Hispanic overrides)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting: faithful listwise deletion on DV + all predictors
    # -----------------------------
    def fit_table2(dv_col, model_name):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # Drop no-variation predictors (true no-variation only)
        kept = []
        dropped_no_var = []
        for p in predictors:
            nun = d[p].nunique(dropna=True)
            if nun <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(fit, d, dv_col, kept)

        # Build "Table 2 style" coefficient table: standardized betas + stars
        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit.pvalues.get(p, np.nan)),
                    }
                )
            else:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": ""})
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Sig": [star_from_p(fit.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        # Write human-readable table output
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS regression (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_std_j = b_j * SD(X_j) / SD(Y) computed on analytic (listwise) sample (ddof=1).")
        lines.append("Stars: two-tailed p-values from the unstandardized OLS regression.")
        lines.append("")
        lines.append("Construction:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Race/ethnicity: Hispanic overrides race so Black/Hispanic/Other are mutually exclusive; White non-Hispanic is reference")
        lines.append("- Missing data: listwise deletion on DV + all predictors used")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped predictors due to no variation in analytic sample:")
            for p in dropped_no_var:
                lines.append(f"- {p}: {labels.get(p, p)}")
        lines.append("")
        lines.append("Standardized coefficients (Table 2 style)")
        lines.append("---------------------------------------")
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

        write_text(f"./output/{model_name}_table2_style.txt", lines)

        # Save full OLS summary for debugging
        with open(f"./output/{model_name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Diagnostics: counts for key dummies within analytic sample
        diag_lines = []
        diag_lines.append(f"{model_name} analytic diagnostics")
        diag_lines.append("=" * (len(model_name) + 20))
        diag_lines.append(f"N analytic: {len(d)}")
        diag_lines.append("")
        for v in ["black", "hispanic", "other_race", "cons_prot", "no_religion", "southern"]:
            vc = d[v].value_counts(dropna=False).sort_index()
            diag_lines.append(f"{v} value counts (analytic):")
            diag_lines.append(vc.to_string())
            diag_lines.append("")
        write_text(f"./output/{model_name}_analytic_diagnostics.txt", diag_lines)

        return table, fit_stats, d

    m1_table, m1_fit, m1_frame = fit_table2(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame = fit_table2(dv2_col, "Table2_ModelB_Remaining12")

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

    # DV descriptives for (a) all 1993 with nonmissing DV, and (b) analytic samples
    def dv_desc_for(series):
        s = series.dropna().astype(float)
        return {
            "N": int(s.shape[0]),
            "Mean": float(s.mean()) if s.shape[0] else np.nan,
            "SD": float(s.std(ddof=0)) if s.shape[0] else np.nan,
            "Min": float(s.min()) if s.shape[0] else np.nan,
            "P25": float(s.quantile(0.25)) if s.shape[0] else np.nan,
            "Median": float(s.quantile(0.50)) if s.shape[0] else np.nan,
            "P75": float(s.quantile(0.75)) if s.shape[0] else np.nan,
            "Max": float(s.max()) if s.shape[0] else np.nan,
        }

    dv_desc = pd.DataFrame(
        [
            {"Sample": "1993 (DV available)", "DV": labels[dv1_col], **dv_desc_for(df[dv1_col])},
            {"Sample": "1993 (DV available)", "DV": labels[dv2_col], **dv_desc_for(df[dv2_col])},
            {"Sample": "Model A analytic", "DV": labels[dv1_col], **dv_desc_for(m1_frame[dv1_col])},
            {"Sample": "Model B analytic", "DV": labels[dv2_col], **dv_desc_for(m2_frame[dv2_col])},
        ]
    )

    miss = df[[dv1_col, dv2_col] + predictors].isna().mean().reset_index()
    miss.columns = ["variable", "share_missing_1993"]
    miss["label"] = miss["variable"].map(lambda v: labels.get(v, v))
    miss = miss.sort_values("share_missing_1993", ascending=False)

    # Summary text
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
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
    lines.append("DV descriptives")
    lines.append("--------------")
    dd = dv_desc.copy()
    for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
        dd[c] = dd[c].map(lambda v: fmt(v, 3))
    dd["N"] = dd["N"].astype(int)
    lines.append(dd.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares in 1993 (before listwise deletion)")
    lines.append("----------------------------------------------------")
    md = miss.copy()
    md["share_missing_1993"] = md["share_missing_1993"].map(lambda v: fmt(v, 3))
    lines.append(md.to_string(index=False))
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("- Hispanic is constructed from ETHNIC==1 in this extract; this may differ from the original GSS Hispanic flag used in the paper.")
    lines.append("- Conservative Protestant is approximated with RELIG==1 & DENOM==1 given the available variables; no missing-to-0 imputation is used.")
    lines.append("- Race/ethnicity dummies are mutually exclusive (Hispanic overrides race) to match the table’s implied category structure.")
    lines.append("- Stars are computed from this replication’s OLS p-values (Table 2 reports stars but not SEs).")

    write_text("./output/combined_summary.txt", lines)

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss.to_csv("./output/missingness_1993.csv", index=False)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_1993": miss,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_fit": m1_fit,
        "modelB_fit": m2_fit,
    }