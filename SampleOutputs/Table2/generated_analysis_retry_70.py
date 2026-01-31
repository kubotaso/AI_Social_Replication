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
        df["id"] = np.arange(len(df), dtype=int)

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # Coerce (except id) to numeric
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Variable lists per mapping
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_raw
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

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
        """Map to {0,1}; anything else missing."""
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

    def nan_share(s):
        return float(pd.to_numeric(s, errors="coerce").isna().mean())

    def standardized_betas_from_fit(fit, d, ycol, xcols):
        """
        Standardized betas from unstandardized coefficients:
            beta_j = b_j * SD(x_j) / SD(y)
        computed on the analytic sample (ddof=0).
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
    # Dependent variables: strict dislike counts
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])     # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])    # 0..12

    # -----------------------------
    # Racism score (0..5): partial completion (>=4 of 5), rescaled to 0..5
    # This reduces N-collapse and better matches typical published GSS scale practices.
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["_rac_nonmiss"] = df[racism_comp].notna().sum(axis=1)
    df["_rac_sum"] = df[racism_comp].sum(axis=1, skipna=True)

    df["racism_score"] = np.where(
        df["_rac_nonmiss"] >= 4,
        df["_rac_sum"] * (5.0 / df["_rac_nonmiss"]),
        np.nan,
    )
    df.drop(columns=["_rac_nonmiss", "_rac_sum"], inplace=True)

    # -----------------------------
    # Controls / indicators (no forced imputation of missing to 0)
    # -----------------------------
    df["education"] = df["educ"]

    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    # Female: 1 if SEX==2, 0 if SEX==1, missing otherwise
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    df["age_years"] = df["age"]

    # Race dummies: reference white (race==1)
    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic indicator: use ETHNIC proxy (must not be constant-zero).
    # Rule: if ETHNIC observed, hispanic=1 if in [15..39] or explicitly 20..29; else 0.
    # If ETHNIC missing -> missing.
    e = df["ethnic"]
    df["hispanic_ind"] = np.nan
    e_obs = e.notna()
    df.loc[e_obs, "hispanic_ind"] = 0.0
    df.loc[e_obs & e.between(15, 39), "hispanic_ind"] = 1.0
    df.loc[e_obs & e.isin(list(range(20, 30))), "hispanic_ind"] = 1.0

    # No religion: RELIG==4, missing if RELIG missing
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: conservative proxy using available fields without requiring denom nonmissing.
    # If RELIG missing -> missing. If RELIG != Protestant -> 0. If Protestant -> 1 iff DENOM==1 else 0 (DENOM missing => 0).
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.nan
    rel_obs = rel.notna()
    df.loc[rel_obs, "cons_prot"] = 0.0
    prot_mask = rel_obs & (rel == 1)
    df.loc[prot_mask, "cons_prot"] = np.where(den.loc[prot_mask].notna(), (den.loc[prot_mask] == 1).astype(float), 0.0)

    # Southern: REGION==3, missing if REGION missing
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
        "racism_score",
        "education",
        "income_pc",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic_ind",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    labels = {
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
        dv2: "Dislike of the 12 Remaining Genres (count of 12)",
        "racism_score": "Racism score (0–5; >=4/5 items, rescaled to 0–5)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (SEX==2)",
        "age_years": "Age (years)",
        "black": "Black (RACE==2)",
        "hispanic_ind": "Hispanic (indicator from ETHNIC proxy)",
        "other_race": "Other race (RACE==3)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting: listwise on DV + predictors.
    # Also: do NOT star the constant to better match typical Table 2 presentation.
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

        betas = standardized_betas_from_fit(fit_unstd, d, dv_col, kept)

        table_rows = []
        for p in predictors:
            status = "included" if p in kept else "dropped (no variation)"
            table_rows.append(
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": float(betas.get(p, np.nan)) if p in kept else np.nan,
                    "Sig": star_from_p(fit_unstd.pvalues.get(p, np.nan)) if p in kept else "",
                    "Status": status,
                }
            )
        table = pd.DataFrame(table_rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable output
        title = f"Bryson (1996) Table 2 replication attempt — {model_name} (computed from provided 1993 GSS extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_j = b_j * SD(x_j) / SD(y) on the analytic sample (ddof=0).")
        lines.append("Stars: two-tailed p-values from unstandardized OLS in this run (computed from the data).")
        lines.append("Note: constant is printed without stars to match typical published-table conventions.")
        lines.append("")
        lines.append("Construction rules implemented:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: >=4/5 dichotomous items; sum rescaled to 0–5; else missing")
        lines.append("- Missing data in regression: listwise deletion across DV + all predictors")
        if target_n is not None:
            lines.append(f"- Target N from paper (reference only): {int(target_n)}")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped due to no variation in analytic sample:")
            lines.append(", ".join(dropped_no_var))
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
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant", "Dropped_no_variation"]].to_string(index=False))
        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))
        write_text(f"./output/{stub}_ols_unstandardized_summary.txt", fit_unstd.summary().as_text())

        # Diagnostics to reconcile N and coding
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

        diag_lines.append("\nKey raw value counts (1993, including NA):")
        diag_lines.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        diag_lines.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag_lines.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag_lines.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag_lines.append("\nETHNIC:\n" + value_counts_full(df["ethnic"]).to_string())

        diag_lines.append("\nDerived indicators value counts (1993, including NA):")
        for v in ["female", "black", "hispanic_ind", "other_race", "cons_prot", "no_religion", "southern"]:
            diag_lines.append(f"\n{v}:\n" + value_counts_full(df[v]).to_string())

        diag_lines.append("\nRacism components missingness:")
        diag_lines.append(df[racism_comp + ["racism_score"]].isna().mean().map(lambda v: fmt(v, 3)).to_string())
        diag_lines.append("\nRacism score value counts:\n" + value_counts_full(df["racism_score"]).to_string())

        diag_lines.append("\nDV value counts:\n" + value_counts_full(df[dv_col]).to_string())

        conts = ["education", "income_pc", "occ_prestige", "age_years"]
        diag_lines.append("\nContinuous variable missingness shares:")
        diag_lines.append(pd.Series({c: nan_share(df[c]) for c in conts}).sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())
        diag_lines.append("\nContinuous variable descriptives (1993, nonmissing):")
        diag_lines.append(df[conts].describe().to_string())

        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d, kept, dropped_no_var, fit_unstd

    m1_table, m1_fit, m1_d, m1_kept, m1_novar, _ = fit_model(
        dv1, "Table 2 Model A (Minority-linked genres: 6)", "Table2_ModelA_MinorityLinked6", target_n=644
    )
    m2_table, m2_fit, m2_d, m2_kept, m2_novar, _ = fit_model(
        dv2, "Table 2 Model B (Remaining genres: 12)", "Table2_ModelB_Remaining12", target_n=605
    )

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

    title = "Bryson (1996) Table 2 replication attempt (computed from provided 1993 GSS extract)"
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Specification implemented")
    lines.append("------------------------")
    lines.append("- Year restriction: YEAR==1993")
    lines.append("- DV1: strict count of dislikes across RAP, REGGAE, BLUES, JAZZ, GOSPEL, LATIN (dislike=4/5)")
    lines.append("- DV2: strict count of dislikes across BIGBAND, BLUGRASS, COUNTRY, MUSICALS, CLASSICL, FOLK, MOODEASY, NEWAGE, OPERA, CONROCK, OLDIES, HVYMETAL (dislike=4/5)")
    lines.append("- Racism score: >=4/5 dichotomous items per mapping; sum rescaled to 0–5")
    lines.append("- Controls: education, income_pc, prestige, female, age, black, hispanic (ETHNIC proxy), other race, cons_prot (RELIG/DENOM proxy), no religion, southern")
    lines.append("- Missing data: model-wise listwise deletion (DV + all predictors)")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and stars (from this run)")
    lines.append("--------------------------------------------------------------------------")
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
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("DV descriptives (counts)")
    lines.append("------------------------")
    dvf = dv_desc_df.copy()
    dvf["N"] = dvf["N"].map(lambda v: fmt(v, 0))
    for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
        dvf[c] = dvf[c].map(lambda v: fmt(v, 3))
    lines.append(dvf[["Sample", "DV", "N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"]].to_string(index=False))
    lines.append("")
    lines.append("Analytic-sample N checkpoints (paper targets: Model A N=644; Model B N=605)")
    lines.append("----------------------------------------------------------------------------")
    lines.append(f"Model A analytic N: {int(m1_d.shape[0])}")
    lines.append(f"Model B analytic N: {int(m2_d.shape[0])}")
    lines.append("")
    lines.append("Kept predictors (after dropping only no-variation columns)")
    lines.append("----------------------------------------------------------")
    lines.append(f"Model A kept: {', '.join(m1_kept) if m1_kept else '(none)'}")
    lines.append(f"Model B kept: {', '.join(m2_kept) if m2_kept else '(none)'}")
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("- Stars are from this run's p-values; they will match the paper only if coding/sample align exactly.")
    lines.append("- Hispanic and Conservative Protestant are proxied from ETHNIC and RELIG/DENOM because this extract lacks the paper’s exact measures.")
    lines.append("- If sample sizes remain below paper targets, check missingness in income_pc and occ_prestige, and availability of music-module items.")

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