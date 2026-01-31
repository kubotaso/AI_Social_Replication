def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    df = df.loc[df["year"] == 1993].copy()

    # Coerce non-id columns to numeric
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Variable lists (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
        + minority_genres
        + remaining_genres
        + racism_raw
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing expected columns: " + ", ".join(missing_cols))

    # -----------------------------
    # Helpers
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")

    def fmt(x, nd=3):
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

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

    def dislike_indicator(series):
        """
        1 if response is 4/5 (dislike/dislike very much),
        0 if response is 1/2/3,
        missing otherwise.
        """
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        """Map to {0,1}; anything else -> missing."""
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

    def standardized_betas_from_fit(fit, d, ycol, xcols):
        """
        Standardized beta weights:
            beta_j = b_j * sd(X_j) / sd(Y)
        computed on analytic sample used for fit.
        """
        y = pd.to_numeric(d[ycol], errors="coerce").astype(float)
        y_sd = y.std(ddof=0)
        out = {}
        for p in xcols:
            x = pd.to_numeric(d[p], errors="coerce").astype(float)
            x_sd = x.std(ddof=0)
            if pd.isna(y_sd) or y_sd == 0 or pd.isna(x_sd) or x_sd == 0:
                out[p] = np.nan
            else:
                out[p] = float(fit.params[p] * (x_sd / y_sd))
        return out

    def vc(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.value_counts(dropna=False)

    # -----------------------------
    # Dependent variables: strict complete-case counts
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6, missing if any missing
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12, missing if any missing

    # -----------------------------
    # Racism score: strict 5/5 items, dichotomized per mapping
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])        # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])        # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])      # 2=no (discrim) -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])      # 2=no (educ) -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])      # 1=yes (willpower) -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)                # 0..5, missing if any missing

    # -----------------------------
    # Controls (preserve missingness; listwise deletion in model)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies (reference = White; preserve missingness)
    race = df["race"]
    df["black"] = np.where(race.isin([1, 2, 3]), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.isin([1, 2, 3]), (race == 3).astype(float), np.nan)

    # Hispanic: attempt best-effort construction from ETHNIC if it appears to be a Hispanic-origin code.
    # If this heuristic cannot identify any Hispanic cases, we leave as missing (unavailable).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = df["ethnic"]
        # Heuristic: if ETHNIC is a numeric ancestry/origin code, common Hispanic-origin codes are often in a distinct range.
        # We implement a conservative rule: treat ETHNIC in [20..39] or [40..49] as Hispanic-origin if present in data.
        # (If this rule yields no 1s, the regressor will be considered unavailable and dropped.)
        hisp_flag = np.where(eth.notna(), eth.isin(list(range(20, 50))).astype(float), np.nan)
        # Only accept if it actually yields variation in the 1993 data
        if pd.Series(hisp_flag).dropna().nunique() > 1 and pd.Series(hisp_flag).sum(skipna=True) > 0:
            df["hispanic"] = hisp_flag

    # Religion dummies (preserve missingness)
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: best available from broad DENOM
    # (Still an approximation; preserve missingness and let listwise deletion handle it.)
    df["cons_prot"] = np.nan
    m_rel = df["relig"].notna() & df["denom"].notna()
    df.loc[m_rel, "cons_prot"] = ((df.loc[m_rel, "relig"] == 1) & (df.loc[m_rel, "denom"] == 1)).astype(float)

    # Southern: per mapping REGION==3; preserve missingness
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
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music",
        dv2: "Dislike of the 12 Remaining Genres",
        "racism_score": "Racism score (0–5; strict 5/5 items)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age_years": "Age (years)",
        "black": "Black (RACE==2)",
        "hispanic": "Hispanic (best-effort from ETHNIC; may be unavailable)",
        "other_race": "Other race (RACE==3)",
        "cons_prot": "Conservative Protestant (approx: RELIG==1 & DENOM==1)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting with strict listwise deletion
    # -----------------------------
    def fit_model(dv_col, model_name, stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Drop predictors that are all-missing in the raw 1993 data (cannot be used)
        all_missing = [p for p in predictors if d0[p].isna().all()]
        usable_predictors = [p for p in predictors if p not in all_missing]

        # Listwise deletion on DV + usable predictors
        d = d0[[dv_col] + usable_predictors].dropna(axis=0, how="any").copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                "Missingness shares in 1993 for model columns:\n"
                + d0.isna().mean().sort_values(ascending=False).to_string()
                + "\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop non-varying predictors to avoid singular matrix
        kept, dropped_no_var = [], []
        for p in usable_predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_fit(fit, d, dv_col, kept)

        rows = []
        for p in predictors:
            if p in all_missing:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": "dropped (unavailable)"})
            elif p in dropped_no_var:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": "dropped (no variation)"})
            elif p in kept:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star(fit.pvalues.get(p, np.nan)),
                        "Status": "included",
                    }
                )
            else:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": "dropped (other)"})

        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Sig": [star(fit.pvalues.get("const", np.nan))],
                "Dropped_unavailable": [", ".join(all_missing) if all_missing else ""],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable report
        title = f"Bryson (1996) Table 2 — {model_name} (computed from provided GSS 1993 extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)} (count)")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_j = b_j * sd(X_j)/sd(Y), computed on the analytic sample.")
        lines.append("Stars: two-tailed p-values from conventional OLS in this run (replication stars).")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Missing data: strict listwise deletion on DV + included predictors")
        if all_missing:
            lines.append("")
            lines.append("Dropped predictors because they are unavailable (all missing) in this extract:")
            for p in all_missing:
                lines.append(f"- {p}: {labels.get(p, p)}")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped predictors due to no variation in analytic sample:")
            for p in dropped_no_var:
                lines.append(f"- {p}: {labels.get(p, p)}")

        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp[["Independent Variable", "Std_Beta", "Sig", "Status"]].to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats.copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(
            fs[
                ["Model", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_unavailable", "Dropped_no_variation"]
            ].to_string(index=False)
        )
        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))

        # Save full OLS summary
        with open(f"./output/{stub}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Diagnostics
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
        diag_lines.append("Value counts (analytic sample) for key dummies:")
        for v in ["female", "black", "hispanic", "other_race", "cons_prot", "no_religion", "southern"]:
            if v in d.columns:
                diag_lines.append(f"\n{v} ({labels.get(v, v)}):")
                diag_lines.append(vc(d[v]).to_string())
        diag_lines.append("")
        diag_lines.append("Underlying raw distributions (1993, pre-listwise):")
        diag_lines.append("\nRELIG value counts:\n" + vc(df["relig"]).to_string())
        diag_lines.append("\nDENOM value counts:\n" + vc(df["denom"]).to_string())
        diag_lines.append("\nREGION value counts:\n" + vc(df["region"]).to_string())
        diag_lines.append("\nRACE value counts:\n" + vc(df["race"]).to_string())
        if "ethnic" in df.columns:
            diag_lines.append("\nETHNIC value counts:\n" + vc(df["ethnic"]).to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d, all_missing, dropped_no_var

    m1_table, m1_fit, m1_d, m1_unavail, m1_novar = fit_model(dv1, "Model A (Minority-linked genres: 6)", "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_d, m2_unavail, m2_novar = fit_model(dv2, "Model B (Remaining genres: 12)", "Table2_ModelB_Remaining12")

    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "ModelA_Std_Beta": m1_table["Std_Beta"],
            "ModelA_Sig": m1_table["Sig"],
            "ModelA_Status": m1_table["Status"],
            "ModelB_Std_Beta": m2_table["Std_Beta"],
            "ModelB_Sig": m2_table["Sig"],
            "ModelB_Status": m2_table["Status"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    def dv_desc(series):
        s = pd.to_numeric(series, errors="coerce")
        return {
            "N": int(s.notna().sum()),
            "Mean": float(s.mean()) if s.notna().any() else np.nan,
            "SD": float(s.std(ddof=0)) if s.notna().any() else np.nan,
            "Min": float(s.min()) if s.notna().any() else np.nan,
            "P25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "Median": float(s.quantile(0.50)) if s.notna().any() else np.nan,
            "P75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "Max": float(s.max()) if s.notna().any() else np.nan,
        }

    dv_desc_df = pd.DataFrame(
        [
            {"Sample": "All 1993 (DV nonmissing)", "DV": labels[dv1], **dv_desc(df[dv1])},
            {"Sample": "All 1993 (DV nonmissing)", "DV": labels[dv2], **dv_desc(df[dv2])},
            {"Sample": "Model A analytic sample", "DV": labels[dv1], **dv_desc(m1_d[dv1])},
            {"Sample": "Model B analytic sample", "DV": labels[dv2], **dv_desc(m2_d[dv2])},
        ]
    )

    # Combined summary text
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Key implementation choices")
    lines.append("--------------------------")
    lines.append("- Year restriction: YEAR==1993")
    lines.append("- DVs: strict complete-case counts of dislikes (4/5 => dislike) across specified genres")
    lines.append("- Racism: strict complete-case sum of 5 dichotomized items (0..5)")
    lines.append("- Missing data in regressions: strict listwise deletion")
    lines.append("- Stars: from this run's OLS p-values (not copied from the paper)")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars (from this run)")
    lines.append("--------------------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; from this run)")
    lines.append("--------------------------------------------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_unavailable", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("DV descriptives (counts)")
    lines.append("------------------------")
    dvf = dv_desc_df.copy()
    dvf["N"] = dvf["N"].map(lambda v: fmt(v, 0))
    for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
        dvf[c] = dvf[c].map(lambda v: fmt(v, 3))
    lines.append(dvf[["Sample", "DV", "N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"]].to_string(index=False))
    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)
    dv_desc_df.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc_df,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_analytic_sample": m1_d,
        "modelB_analytic_sample": m2_d,
    }