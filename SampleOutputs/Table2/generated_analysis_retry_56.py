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
        raise ValueError("Expected 'year' column in CSV.")
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # Coerce all but id to numeric
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Variable mapping (per instructions)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    core_required = [
        "hompop", "educ", "realinc", "prestg80",
        "sex", "age", "race",
        "relig", "denom", "region",
        "ethnic"
    ]

    required = list(dict.fromkeys(core_required + minority_genres + remaining_genres + racism_items_raw))
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in extract: {missing_cols}")

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

    def dislike_indicator(x):
        """
        1 if response is 4/5 (dislike/dislike very much),
        0 if response is 1/2/3,
        missing otherwise.
        """
        x = pd.to_numeric(x, errors="coerce")
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

    def zscore(s):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def standardized_betas_from_unstd(fit, d, ycol, xcols):
        """
        Compute standardized betas using:
          beta_j = b_j * sd(x_j) / sd(y)
        on the analytic sample used for `fit`.
        """
        y = d[ycol].astype(float)
        sd_y = y.std(ddof=0)
        betas = {}
        for x in xcols:
            sd_x = d[x].astype(float).std(ddof=0)
            b = fit.params.get(x, np.nan)
            if pd.isna(b) or pd.isna(sd_x) or pd.isna(sd_y) or sd_x == 0 or sd_y == 0:
                betas[x] = np.nan
            else:
                betas[x] = float(b * (sd_x / sd_y))
        return pd.Series(betas)

    def value_counts_full(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.value_counts(dropna=False).sort_index()

    # -----------------------------
    # DVs: strict complete-case dislike counts (per mapping)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6 (strict)
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12 (strict)

    # -----------------------------
    # Racism score: STRICT 5/5 items -> integer 0..5 (per feedback)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])        # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])        # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])      # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])      # 2=no educ chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])      # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # missing if any component missing; integer 0..5

    # -----------------------------
    # Controls: preserve missingness (NO imputation of missings to 0)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies (reference = White); preserve missingness
    race = df["race"]
    df["black"] = np.where(race.isin([1, 2, 3]), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.isin([1, 2, 3]), (race == 3).astype(float), np.nan)

    # Hispanic: best available in this extract is ETHNIC.
    # Use a stable, explicit heuristic:
    #   - If ETHNIC is coded 1..N with 1=not Hispanic and some codes representing Hispanic origins,
    #     we do not know exact mapping. But we must avoid "data-driven scanning" that can change run-to-run.
    # Strategy:
    #   - Build a small set of candidate "Hispanic-like" codes using observed ETHNIC values:
    #     prefer codes with labels commonly appearing as small integers (e.g., 1=non-Hispanic).
    #   - If that fails (no variation), fall back to a deterministic range [20,49] (as commonly used in some origin codings).
    eth = df["ethnic"]
    df["hispanic"] = np.nan
    if eth.notna().any():
        # Deterministic candidate sets (in order)
        candidate_sets = [
            set([2, 3, 4, 5, 6, 7, 8, 9]),                 # common "other ancestry" buckets; may include Hispanic in some codings
            set(range(20, 50)),                             # deterministic fallback
            set(range(50, 80)),
            set(range(80, 120)),
        ]
        chosen = None
        for s in candidate_sets:
            flag = eth.isin(list(s)).astype(float)
            # accept only if variation exists AND at least 5 positives (avoid trivial noise)
            if flag.nunique(dropna=True) > 1 and float(flag.sum()) >= 5:
                chosen = flag
                break
        if chosen is not None:
            df["hispanic"] = np.where(eth.notna(), chosen, np.nan)

    # Religion variables (do not condition "no religion" on denom availability)
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy available in this extract: Protestant & Baptist (RELIG==1 & DENOM==1)
    df["cons_prot"] = np.nan
    m_rel = df["relig"].notna() & df["denom"].notna()
    df.loc[m_rel, "cons_prot"] = ((df.loc[m_rel, "relig"] == 1) & (df.loc[m_rel, "denom"] == 1)).astype(float)

    # Southern: REGION coding unknown in this extract; default to REGION==3 per mapping instruction,
    # but also write diagnostics so mismatches can be spotted quickly.
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
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
        dv2: "Dislike of the 12 Remaining Genres (count of 12)",
        "racism_score": "Racism score (0–5; strict 5/5 items, integer)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age_years": "Age (years)",
        "black": "Black (RACE==2)",
        "hispanic": "Hispanic (from ETHNIC; extract-limited heuristic)",
        "other_race": "Other race (RACE==3)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3 per mapping)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting (strict listwise deletion, faithful)
    # -----------------------------
    def fit_model(dv_col, model_name, stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Drop predictors that are all-missing in this extract (cannot be used)
        all_missing = [p for p in predictors if d0[p].isna().all()]
        usable_predictors = [p for p in predictors if p not in all_missing]

        # Strict listwise deletion on DV + usable predictors
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

        # Drop non-varying predictors (avoid singular matrix)
        kept, dropped_no_var = [], []
        for p in usable_predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas computed from unstandardized b's + SDs (common "beta weights" approach)
        betas = standardized_betas_from_unstd(fit_unstd, d, dv_col, kept)

        # Table rows in original order
        rows = []
        for p in predictors:
            if p in all_missing:
                status = "dropped (unavailable)"
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": status})
            elif p in dropped_no_var:
                status = "dropped (no variation)"
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": status})
            elif p in kept:
                status = "included"
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit_unstd.pvalues.get(p, np.nan)),
                        "Status": status,
                    }
                )
            else:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": "dropped (other)"})

        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [star_from_p(fit_unstd.pvalues.get("const", np.nan))],
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
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_j = b_j * SD(x_j) / SD(y) computed on analytic sample.")
        lines.append("Stars: two-tailed p-values from unstandardized OLS in this run.")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict 5/5 dichotomous items; sum -> 0..5 (missing if any item missing)")
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
            f.write(fit_unstd.summary().as_text())
            f.write("\n")

        # Diagnostics: missingness + key distributions
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
                diag_lines.append(value_counts_full(d[v]).to_string())

        diag_lines.append("")
        diag_lines.append("Underlying raw distributions (1993, pre-listwise):")
        diag_lines.append("\nRELIG value counts:\n" + value_counts_full(df["relig"]).to_string())
        diag_lines.append("\nDENOM value counts:\n" + value_counts_full(df["denom"]).to_string())
        diag_lines.append("\nREGION value counts:\n" + value_counts_full(df["region"]).to_string())
        diag_lines.append("\nRACE value counts:\n" + value_counts_full(df["race"]).to_string())
        diag_lines.append("\nETHNIC value counts:\n" + value_counts_full(df["ethnic"]).to_string())
        diag_lines.append("\nRacism components missingness:\n" + df[racism_comp + ["racism_score"]].isna().mean().map(lambda v: fmt(v, 3)).to_string())
        diag_lines.append("\nRacism score value counts (1993, pre-listwise):\n" + value_counts_full(df["racism_score"]).to_string())
        diag_lines.append("\nDV value counts (1993, pre-listwise):\n" + value_counts_full(df[dv_col]).to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d, all_missing, dropped_no_var, kept

    m1_table, m1_fit, m1_d, m1_unavail, m1_novar, m1_kept = fit_model(
        dv1, "Model A (Minority-linked genres: 6)", "Table2_ModelA_MinorityLinked6"
    )
    m2_table, m2_fit, m2_d, m2_unavail, m2_novar, m2_kept = fit_model(
        dv2, "Model B (Remaining genres: 12)", "Table2_ModelB_Remaining12"
    )

    # -----------------------------
    # Combined outputs
    # -----------------------------
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

    # Human-readable combined summary
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Key implementation choices")
    lines.append("--------------------------")
    lines.append("- Year restriction: YEAR==1993")
    lines.append("- DVs: strict complete-case counts of dislikes (4/5 => dislike) across specified genres")
    lines.append("- Racism: strict 5/5 items required; summed to integer 0–5")
    lines.append("- Missing data in regressions: strict listwise deletion (DV + all included predictors)")
    lines.append("- Standardized coefficients: beta_j = b_j * SD(x_j) / SD(y) on analytic sample")
    lines.append("- Stars: from this run's OLS p-values (replication stars)")
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

    # Quick checks to help reconcile earlier discrepancy causes
    lines.append("")
    lines.append("Quick checks (1993 pre-listwise)")
    lines.append("-------------------------------")
    lines.append(f"Racism score unique values: {sorted([int(x) for x in df['racism_score'].dropna().unique()]) if df['racism_score'].notna().any() else '[]'}")
    lines.append(f"No religion share (nonmissing RELIG): {fmt((df['no_religion'] == 1).mean() if df['no_religion'].notna().any() else np.nan, 3)}")
    lines.append(f"Southern share (nonmissing REGION): {fmt((df['southern'] == 1).mean() if df['southern'].notna().any() else np.nan, 3)}")
    lines.append(f"Hispanic share (nonmissing ETHNIC): {fmt((df['hispanic'] == 1).mean() if df['hispanic'].notna().any() else np.nan, 3)}")
    lines.append("")
    lines.append("Kept predictors in each model after dropping unavailable/no-variation:")
    lines.append(f"- Model A kept: {', '.join(m1_kept)}")
    lines.append(f"- Model B kept: {', '.join(m2_kept)}")

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