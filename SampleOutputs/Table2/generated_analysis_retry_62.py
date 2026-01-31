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
        df["id"] = np.arange(len(df), dtype=int)

    # Keep 1993 only
    df = df.loc[df["year"] == 1993].copy()

    # Coerce numeric (except id)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
        1 if response in {4,5} (dislike/dislike very much),
        0 if response in {1,2,3},
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

    def value_counts_full(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.value_counts(dropna=False).sort_index()

    def standardized_betas_from_unstd(fit, d, ycol, xcols):
        """
        Standardized betas computed from unstandardized OLS on the analytic sample:
          beta_j = b_j * SD(x_j) / SD(y)
        Uses ddof=0 (population SD), matching typical beta-weight computation.
        """
        y = pd.to_numeric(d[ycol], errors="coerce").astype(float)
        sd_y = y.std(ddof=0)
        betas = {}
        for x in xcols:
            sx = pd.to_numeric(d[x], errors="coerce").astype(float)
            sd_x = sx.std(ddof=0)
            b = fit.params.get(x, np.nan)
            if pd.isna(b) or pd.isna(sd_x) or pd.isna(sd_y) or sd_x == 0 or sd_y == 0:
                betas[x] = np.nan
            else:
                betas[x] = float(b * (sd_x / sd_y))
        return pd.Series(betas)

    # -----------------------------
    # Required columns (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    core_cols = ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "region", "ethnic", "denom"]
    needed = ["year", "id"] + core_cols + minority_genres + remaining_genres + racism_items_raw
    missing_needed = [c for c in needed if c not in df.columns]
    if missing_needed:
        # We require ethnic+denom here to avoid fabricating Hispanic/ConsProt and to prevent silent model mismatch
        raise ValueError(
            "Missing expected columns needed for Table 2 replication: "
            + ", ".join(missing_needed)
            + ".\nThis function requires ETHNIC and DENOM to construct Hispanic and Conservative Protestant."
        )

    # -----------------------------
    # Dependent variables (strict complete-case within each DV set)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (STRICT 5/5 items; sum 0..5) per mapping
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls (keep missing as missing; listwise deletion in models)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies: reference is White (RACE==1)
    race = df["race"]
    df["black"] = np.where(race.isin([1, 2, 3]), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.isin([1, 2, 3]), (race == 3).astype(float), np.nan)

    # Southern: REGION==3 per mapping (do not collapse missing to 0)
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # No religion: RELIG==4 per mapping (do not collapse missing to 0)
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Hispanic: derive from ETHNIC (ancestry/origin code); keep unknowns missing.
    # Rule: mark Hispanic if ETHNIC is in a Latin-origin band (15..38) or equals 15 as seen in sample.
    # Treat 97 (other/NA in sample) as missing. Leave other special codes missing as well.
    e = df["ethnic"]
    df["hispanic"] = np.nan
    e_known = e.notna() & (~e.isin([97, 98, 99]))
    df.loc[e_known, "hispanic"] = 0.0
    hisp_codes = set([15]) | set(range(16, 39))
    df.loc[e_known & e.isin(list(hisp_codes)), "hispanic"] = 1.0

    # Conservative Protestant: cannot perfectly match Bryson without original scheme.
    # Use minimal defensible proxy based on available broad DENOM:
    # RELIG==1 (Protestant) & DENOM==1 (Baptist) => 1; all others => 0; missing if RELIG/DENOM missing.
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.nan
    m = rel.notna() & den.notna()
    df.loc[m, "cons_prot"] = 0.0
    df.loc[m & (rel == 1) & (den == 1), "cons_prot"] = 1.0

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
        "racism_score": "Racism score (0–5; strict 5 items, sum)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (SEX==2)",
        "age_years": "Age (years)",
        "black": "Black (RACE==2)",
        "hispanic": "Hispanic (derived from ETHNIC in {15..38}; unknown ETHNIC missing)",
        "other_race": "Other race (RACE==3)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1 (Baptist))",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting (strict listwise deletion; keep all predictors, drop only no-variation)
    # -----------------------------
    def fit_model(dv_col, model_name, stub, target_n=None):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(axis=0, how="any").copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                "Missingness shares in 1993 for model columns:\n"
                + d0.isna().mean().sort_values(ascending=False).to_string()
                + "\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

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

        betas = standardized_betas_from_unstd(fit_unstd, d, dv_col, kept)

        rows = []
        for p in predictors:
            status = "included" if p in kept else "dropped (no variation)"
            rows.append(
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": float(betas.get(p, np.nan)) if p in kept else np.nan,
                    "Sig": star_from_p(fit_unstd.pvalues.get(p, np.nan)) if p in kept else "",
                    "Status": status,
                }
            )
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
        lines.append("Standardization: beta_j = b_j * SD(x_j) / SD(y) on the analytic sample (ddof=0).")
        lines.append("Stars: two-tailed p-values from unstandardized OLS in this run (replication-run stars).")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict 5/5 dichotomous items summed to 0..5 (missing if any item missing)")
        lines.append("- Missing data in regressions: strict listwise deletion on DV + all predictors")
        lines.append(f"- Hispanic: {labels['hispanic']}")
        lines.append(f"- Conservative Protestant: {labels['cons_prot']}")
        lines.append(f"- Southern: {labels['southern']}")
        lines.append(f"- No religion: {labels['no_religion']}")
        if target_n is not None:
            lines.append(f"- Target N from paper (for reference): {int(target_n)}")
        lines.append("")
        lines.append("Standardized coefficients (Table 2 style)")
        lines.append("---------------------------------------")
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
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].to_string(index=False))
        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))

        with open(f"./output/{stub}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit_unstd.summary().as_text())
            f.write("\n")

        # Diagnostics (key for troubleshooting N mismatches)
        diag_lines = []
        diag_lines.append(f"{model_name} diagnostics")
        diag_lines.append("=" * (len(model_name) + 12))
        diag_lines.append(f"N_1993_total: {int(df.shape[0])}")
        diag_lines.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag_lines.append(f"N_analytic_listwise: {int(d.shape[0])}")
        if target_n is not None:
            diag_lines.append(f"N_target_from_paper: {int(target_n)}")
            diag_lines.append(f"N_gap (analytic - target): {int(d.shape[0]) - int(target_n)}")
        diag_lines.append("")
        diag_lines.append("Missingness shares in 1993 for model columns (descending):")
        diag_lines.append(d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())

        diag_lines.append("\n\nRaw value counts (1993, pre-listwise; including NA):")
        diag_lines.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag_lines.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        diag_lines.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag_lines.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag_lines.append("\nETHNIC:\n" + value_counts_full(df["ethnic"]).to_string())

        diag_lines.append("\nHispanic indicator used:\n" + value_counts_full(df["hispanic"]).to_string())
        diag_lines.append("\nNo religion indicator used:\n" + value_counts_full(df["no_religion"]).to_string())
        diag_lines.append("\nSouthern indicator used:\n" + value_counts_full(df["southern"]).to_string())
        diag_lines.append("\nConservative Protestant indicator used:\n" + value_counts_full(df["cons_prot"]).to_string())

        diag_lines.append("\nRacism components missingness:\n" + df[racism_comp + ["racism_score"]].isna().mean().map(lambda v: fmt(v, 3)).to_string())
        diag_lines.append("\nRacism score value counts:\n" + value_counts_full(df["racism_score"]).to_string())

        diag_lines.append("\nDV value counts:\n" + value_counts_full(df[dv_col]).to_string())

        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d, kept, dropped_no_var, fit_unstd

    m1_table, m1_fit, m1_d, m1_kept, m1_novar, m1_fit_unstd = fit_model(
        dv1, "Model 1 (Minority-linked genres: 6)", "Table2_Model1_MinorityLinked6", target_n=644
    )
    m2_table, m2_fit, m2_d, m2_kept, m2_novar, m2_fit_unstd = fit_model(
        dv2, "Model 2 (Remaining genres: 12)", "Table2_Model2_Remaining12", target_n=605
    )

    # -----------------------------
    # Combined outputs
    # -----------------------------
    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "Model1_Std_Beta": m1_table["Std_Beta"],
            "Model1_Sig": m1_table["Sig"],
            "Model2_Std_Beta": m2_table["Std_Beta"],
            "Model2_Sig": m2_table["Sig"],
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
            {"Sample": "Model 1 analytic sample", "DV": labels[dv1], **dv_desc(m1_d[dv1])},
            {"Sample": "Model 2 analytic sample", "DV": labels[dv2], **dv_desc(m2_d[dv2])},
        ]
    )

    title = "Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)"
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Specification implemented")
    lines.append("------------------------")
    lines.append("- Year restriction: YEAR==1993")
    lines.append("- DV1 (Model 1): strict count of dislikes across RAP, REGGAE, BLUES, JAZZ, GOSPEL, LATIN (dislike=4/5)")
    lines.append("- DV2 (Model 2): strict count of dislikes across BIGBAND, BLUGRASS, COUNTRY, MUSICALS, CLASSICL, FOLK, MOODEASY, NEWAGE, OPERA, CONROCK, OLDIES, HVYMETAL (dislike=4/5)")
    lines.append("- Racism score: strict 5/5 dichotomous items summed to 0..5 per mapping")
    lines.append("- Controls: education, income_pc, prestige, female, age, race dummies, hispanic, cons_prot, no_religion, southern")
    lines.append("- Missing data: strict model-wise listwise deletion (no imputing missing to 0)")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and stars (from this run)")
    lines.append("--------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["Model1_Std_Beta"] = tmp["Model1_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["Model2_Std_Beta"] = tmp["Model2_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; from this run)")
    lines.append("--------------------------------------------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("DV descriptives (counts)")
    lines.append("------------------------")
    dvf = dv_desc_df.copy()
    dvf["N"] = dvf["N"].map(lambda v: fmt(v, 0))
    for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
        dvf[c] = dvf[c].map(lambda v: fmt(v, 3))
    lines.append(dvf[["Sample", "DV", "N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"]].to_string(index=False))
    lines.append("")
    lines.append("Analytic-sample N checkpoints (paper targets: Model 1 N=644; Model 2 N=605)")
    lines.append("----------------------------------------------------------------------------")
    lines.append(f"Model 1 analytic N: {int(m1_d.shape[0])}")
    lines.append(f"Model 2 analytic N: {int(m2_d.shape[0])}")
    lines.append("")
    lines.append("Kept predictors (after dropping only no-variation columns)")
    lines.append("----------------------------------------------------------")
    lines.append(f"Model 1 kept: {', '.join(m1_kept) if m1_kept else '(none)'}")
    lines.append(f"Model 2 kept: {', '.join(m2_kept) if m2_kept else '(none)'}")
    lines.append("")
    lines.append("Notes / limitations")
    lines.append("-------------------")
    lines.append("- Hispanic is derived from ETHNIC codes available in this extract; if your ETHNIC coding differs from Bryson's, coefficients/N may differ.")
    lines.append("- Conservative Protestant is approximated using RELIG and the broad DENOM recode (Baptist); if Bryson used a richer classification, results may differ.")
    lines.append("- Southern uses REGION==3 per provided mapping; if REGION coding differs from Census South in Bryson, adjust accordingly.")

    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)
    dv_desc_df.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc_df,
        "model1_table": m1_table,
        "model2_table": m2_table,
        "model1_analytic_sample": m1_d,
        "model2_analytic_sample": m2_d,
    }