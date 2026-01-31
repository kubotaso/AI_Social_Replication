def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + normalize
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    if "id" not in df.columns:
        # allow missing id, but warn via output file
        df["id"] = np.arange(len(df))

    df = df.loc[df["year"] == 1993].copy()

    # Coerce to numeric (except id which may be string-like)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Required fields
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_required = [
        "hompop", "educ", "realinc", "prestg80", "sex", "age", "race",
        "relig", "denom", "region"
    ]
    # ethnic is optional but used for a best-effort hispanic; if absent, hispanic becomes missing
    optional = ["ethnic"]

    required = base_required + minority_genres + remaining_genres + racism_items_raw
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    for c in optional:
        if c not in df.columns:
            df[c] = np.nan

    # -----------------------------
    # Helpers
    # -----------------------------
    def write_text(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    def dislike_indicator(x):
        """
        1 if 4/5, 0 if 1/2/3, else missing.
        """
        x = pd.to_numeric(x, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        return dfin[cols].sum(axis=1, skipna=False)

    def star(p):
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

    def safe_vc(s):
        return s.value_counts(dropna=False)

    def standardized_betas_from_unstd(fit, d_analytic, y_col, x_cols):
        """
        Compute standardized betas from unstandardized slopes:
            beta_std_j = b_j * SD(X_j) / SD(Y)
        using analytic sample SDs with ddof=1.
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

    # -----------------------------
    # DV construction (strict: any missing item => DV missing)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1_col = "dv1_dislike_minority_linked_6"
    dv2_col = "dv2_dislike_remaining_12"
    df[dv1_col] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2_col] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (0–5), partial completion to avoid N collapse
    # Require at least 4 of 5 answered; rescale to 0..5 metric.
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no edu chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]

    nonmiss = df[racism_comp].notna().sum(axis=1)
    ssum = df[racism_comp].sum(axis=1, skipna=True)
    df["racism_score"] = np.where(nonmiss >= 4, ssum * (5.0 / nonmiss), np.nan)

    # -----------------------------
    # Controls (preserve missing for listwise deletion)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies: White reference category; preserve missing if race unknown
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: best-effort using ETHNIC if present. We do NOT force missing to 0.
    # Because codebook for this extract isn't provided, we try two common patterns:
    # (A) ETHNIC in {1,2,3,4} indicates Hispanic-origin (as in prior attempt)
    # (B) ETHNIC equals a single "Hispanic" category (often 20 or 21 in other recodes) is not known here.
    # We implement (A) only, but we save a diagnostic frequency table for review.
    df["hispanic"] = np.nan
    eth = df["ethnic"]
    hisp_codes = {1, 2, 3, 4}
    df.loc[eth.notna() & eth.isin(list(hisp_codes)), "hispanic"] = 1.0
    df.loc[eth.notna() & (~eth.isin(list(hisp_codes))), "hispanic"] = 0.0

    # Conservative Protestant: per mapping instruction, approximate as RELIG==1 & DENOM==1; preserve missing
    df["cons_prot"] = np.nan
    known_rel_denom = df["relig"].notna() & df["denom"].notna()
    df.loc[known_rel_denom, "cons_prot"] = (
        (df.loc[known_rel_denom, "relig"] == 1) & (df.loc[known_rel_denom, "denom"] == 1)
    ).astype(float)

    # No religion: RELIG==4; preserve missing
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern: REGION==3; preserve missing
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictor order per Table 2
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
        dv1_col: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music",
        dv2_col: "Dislike of the 12 Remaining Genres",
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
    }

    # -----------------------------
    # Model fitting (listwise deletion on DV + all predictors)
    # -----------------------------
    def fit_table2(dv_col, model_name, model_number):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # Drop no-variation predictors (singular design protection); record them
        kept, dropped_no_var = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n"
                f"Missingness shares (1993) for model columns:\n{d0.isna().mean().sort_values(ascending=False).to_string()}\n"
            )
            write_text(f"./output/{model_name}_ERROR.txt", [msg])
            raise ValueError(msg)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(fit, d, dv_col, kept)

        table_rows = []
        for p in predictors:
            if p in kept:
                table_rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star(fit.pvalues.get(p, np.nan)),
                    }
                )
            else:
                table_rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Sig": "",
                    }
                )
        table = pd.DataFrame(table_rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [f"Model {model_number}"],
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Sig": [star(fit.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[f"Model_{model_number}"],
        )

        # Human-readable summary
        lines = []
        title = f"Bryson (1996) Table 2 replication (GSS 1993 extract) — Model {model_number}"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_std_j = b_j * SD(X_j) / SD(Y), computed on the analytic (listwise) sample (ddof=1).")
        lines.append("Stars: two-tailed p-values from the unstandardized OLS regression shown in ./output/*_ols_summary.txt.")
        lines.append("")
        lines.append("Construction rules used:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: 5 dichotomies; requires >=4 answered; sum rescaled to 0–5; otherwise missing")
        lines.append("- Hispanic: ETHNIC in {1,2,3,4} => 1; other non-missing ETHNIC => 0; missing preserved")
        lines.append("- Conservative Protestant: (RELIG==1 & DENOM==1), missing preserved")
        lines.append("- No religion: (RELIG==4), missing preserved")
        lines.append("- Southern: (REGION==3), missing preserved")
        lines.append("- Missing data: strict listwise deletion on DV + all predictors")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped due to no variation in analytic sample:")
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
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Sig"]].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())

        write_text(f"./output/{model_name}_table2_style.txt", lines)

        with open(f"./output/{model_name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Diagnostics for key mismatches: N drivers + dummy variation
        diag_lines = []
        diag_lines.append(f"{model_name} diagnostics")
        diag_lines.append("=" * (len(model_name) + 12))
        diag_lines.append(f"N_1993_total: {int(df.shape[0])}")
        diag_lines.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag_lines.append(f"N_analytic_listwise: {int(d.shape[0])}")
        diag_lines.append("")
        diag_lines.append("Missingness shares in 1993 for model columns (descending):")
        diag_lines.append(d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())
        diag_lines.append("")
        diag_lines.append("Unique values in analytic sample:")
        for v in [dv_col] + predictors:
            diag_lines.append(f"- {v}: unique={int(d[v].nunique(dropna=True))}")
        diag_lines.append("")
        for v in ["black", "hispanic", "other_race", "cons_prot", "no_religion", "southern", "female"]:
            diag_lines.append(f"{v} value counts (analytic):")
            diag_lines.append(safe_vc(d[v]).to_string())
            diag_lines.append("")
        write_text(f"./output/{model_name}_diagnostics.txt", diag_lines)

        return table, fit_stats, d

    m1_table, m1_fit, m1_frame = fit_table2(dv1_col, "Table2_ModelA_MinorityLinked6", 1)
    m2_table, m2_fit, m2_frame = fit_table2(dv2_col, "Table2_ModelB_Remaining12", 2)

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

    # -----------------------------
    # Descriptives + QC outputs
    # -----------------------------
    def dv_desc(series):
        s = series.dropna().astype(float)
        if s.shape[0] == 0:
            return {"N": 0, "Mean": np.nan, "SD": np.nan, "Min": np.nan, "P25": np.nan, "Median": np.nan, "P75": np.nan, "Max": np.nan}
        return {
            "N": int(s.shape[0]),
            "Mean": float(s.mean()),
            "SD": float(s.std(ddof=0)),
            "Min": float(s.min()),
            "P25": float(s.quantile(0.25)),
            "Median": float(s.quantile(0.50)),
            "P75": float(s.quantile(0.75)),
            "Max": float(s.max()),
        }

    dv_desc_df = pd.DataFrame(
        [
            {"Sample": "1993 (DV available)", "DV": labels[dv1_col], **dv_desc(df[dv1_col])},
            {"Sample": "1993 (DV available)", "DV": labels[dv2_col], **dv_desc(df[dv2_col])},
            {"Sample": "Model 1 analytic (listwise)", "DV": labels[dv1_col], **dv_desc(m1_frame[dv1_col])},
            {"Sample": "Model 2 analytic (listwise)", "DV": labels[dv2_col], **dv_desc(m2_frame[dv2_col])},
        ]
    )

    miss = df[[dv1_col, dv2_col] + predictors].isna().mean().reset_index()
    miss.columns = ["variable", "share_missing_1993"]
    miss["label"] = miss["variable"].map(lambda v: labels.get(v, v))
    miss = miss.sort_values("share_missing_1993", ascending=False)

    music_items = [f"d_{g}" for g in (minority_genres + remaining_genres)]
    music_missing = pd.DataFrame({"item": music_items, "share_missing": df[music_items].isna().mean().values})

    racism_missing = pd.DataFrame(
        {"item": racism_comp + ["racism_score"], "share_missing": df[racism_comp + ["racism_score"]].isna().mean().values}
    )

    # Key QC: show ETHNIC distribution and constructed Hispanic distribution
    qc = []
    qc.append(pd.DataFrame({"variable": "ethnic", "value": safe_vc(df["ethnic"]).index.astype(str), "count": safe_vc(df["ethnic"]).values}))
    qc.append(pd.DataFrame({"variable": "hispanic", "value": safe_vc(df["hispanic"]).index.astype(str), "count": safe_vc(df["hispanic"]).values}))
    qc.append(pd.DataFrame({"variable": "no_religion", "value": safe_vc(df["no_religion"]).index.astype(str), "count": safe_vc(df["no_religion"]).values}))
    qc.append(pd.DataFrame({"variable": "cons_prot", "value": safe_vc(df["cons_prot"]).index.astype(str), "count": safe_vc(df["cons_prot"]).values}))
    qc.append(pd.DataFrame({"variable": "southern", "value": safe_vc(df["southern"]).index.astype(str), "count": safe_vc(df["southern"]).values}))
    qc_df = pd.concat(qc, axis=0, ignore_index=True)

    # -----------------------------
    # Combined human-readable summary
    # -----------------------------
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars (from this replication)")
    lines.append("-------------------------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))

    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; this replication)")
    lines.append("-----------------------------------------------------")
    fs = combined_fit[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs.to_string(index=False))

    lines.append("")
    lines.append("DV descriptives")
    lines.append("--------------")
    dd = dv_desc_df.copy()
    dd["N"] = dd["N"].astype(int)
    for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
        dd[c] = dd[c].map(lambda v: fmt(v, 3))
    lines.append(dd.to_string(index=False))

    lines.append("")
    lines.append("Missingness shares in 1993 (before listwise deletion)")
    lines.append("----------------------------------------------------")
    md = miss.copy()
    md["share_missing_1993"] = md["share_missing_1993"].map(lambda v: fmt(v, 3))
    lines.append(md.to_string(index=False))

    lines.append("")
    lines.append("Per-item missingness: music dislike indicators")
    lines.append("--------------------------------------------")
    mi = music_missing.copy()
    mi["share_missing"] = mi["share_missing"].map(lambda v: fmt(v, 3))
    lines.append(mi.to_string(index=False))

    lines.append("")
    lines.append("Per-item missingness: racism components and scale")
    lines.append("------------------------------------------------")
    ri = racism_missing.copy()
    ri["share_missing"] = ri["share_missing"].map(lambda v: fmt(v, 3))
    lines.append(ri.to_string(index=False))

    lines.append("")
    lines.append("Quick-check distributions (pre-listwise)")
    lines.append("----------------------------------------")
    lines.append(qc_df.to_string(index=False))

    write_text("./output/combined_summary.txt", lines)

    # Save artifacts
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss.to_csv("./output/missingness_1993.csv", index=False)
    dv_desc_df.to_csv("./output/dv_descriptives.csv", index=False)
    music_missing.to_csv("./output/item_missingness_music.csv", index=False)
    racism_missing.to_csv("./output/item_missingness_racism.csv", index=False)
    qc_df.to_csv("./output/quickcheck_distributions.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc_df,
        "missingness_1993": miss,
        "item_missingness_music": music_missing,
        "item_missingness_racism": racism_missing,
        "quickcheck_distributions": qc_df,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_fit": m1_fit,
        "modelB_fit": m2_fit,
    }