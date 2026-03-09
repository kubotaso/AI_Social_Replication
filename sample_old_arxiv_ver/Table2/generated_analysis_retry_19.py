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

    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

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

    # -----------------------------
    # Variables (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_cols = [
        "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race",
        "ethnic", "relig", "denom", "region"
    ]

    required = [c for c in base_cols if c in df.columns] + minority_genres + remaining_genres + racism_items
    missing_required = [c for c in (minority_genres + remaining_genres + racism_items + ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"])
                        if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing expected columns: {missing_required}")

    # Numeric coercion
    for c in df.columns:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # DVs: strict dislike counts (missing if any component missing)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    dv1_col = "dv1_minority6_dislikes"
    dv2_col = "dv2_remaining12_dislikes"
    df[dv1_col] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df[dv2_col] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0–5): strict sum of 5 dichotomies (missing if any component missing)
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
    # Education
    df["educ"] = to_num(df["educ"])

    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Occupational prestige
    df["prestg80"] = to_num(df["prestg80"])

    # Female (SEX: 1=male, 2=female)
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Age
    df["age"] = to_num(df["age"])

    # Race dummies (reference = white)
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic: derive from ETHNIC if available; code non-Hispanic as 0 (NOT missing) when ETHNIC is observed.
    # This avoids the previous "many missing => N collapse" problem.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        m = df["ethnic"].notna()
        # Best-available assumption for this extract (consistent with earlier attempts):
        # ETHNIC==1 indicates Hispanic; other observed values indicate not Hispanic.
        df.loc[m, "hispanic"] = (df.loc[m, "ethnic"] == 1).astype(float)

    # Conservative Protestant: cannot be perfectly reconstructed from this extract.
    # To stay faithful to "do not impute missing to 0" while avoiding catastrophic N collapse,
    # we code a conservative proxy using RELIG and DENOM but treat missing as missing.
    # (If extract has more detailed denom info, this should be replaced.)
    df["cons_prot"] = np.nan
    m = df["relig"].notna() & df["denom"].notna()
    # Proxy: Protestant & Baptist
    df.loc[m, "cons_prot"] = ((df.loc[m, "relig"] == 1) & (df.loc[m, "denom"] == 1)).astype(float)

    # No religion (RELIG==4)
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern (REGION==3)
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
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
        "hispanic": "Hispanic (ETHNIC==1 if observed; other observed ETHNIC -> 0; missing ETHNIC -> missing)",
        "other_race": "Other race (1=other)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting (faithful + robust to missingness)
    # -----------------------------
    def fit_model(dv_col, model_name):
        model_cols = [dv_col] + predictors
        d_raw = df[model_cols].copy()

        # Drop predictors that are entirely missing (so we can still run and diagnose)
        usable = []
        dropped_all_missing = []
        for p in predictors:
            if d_raw[p].notna().sum() == 0:
                dropped_all_missing.append(p)
            else:
                usable.append(p)

        # Listwise deletion on DV + usable predictors (faithful to published listwise approach)
        d = d_raw.dropna(subset=[dv_col] + usable).copy()

        # Drop no-variation predictors (singularity protection)
        kept = []
        dropped_no_var = []
        for p in usable:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        # If empty sample, write diagnostic and return placeholders
        if len(d) == 0:
            msg = []
            msg.append(model_name)
            msg.append("=" * len(model_name))
            msg.append("")
            msg.append("ERROR: Analytic sample is empty after listwise deletion.")
            msg.append(f"DV: {labels.get(dv_col, dv_col)}")
            msg.append("")
            msg.append("Missingness shares in 1993 (before listwise):")
            miss = df[[dv_col] + usable].isna().mean().reset_index()
            miss.columns = ["variable", "share_missing"]
            miss["label"] = miss["variable"].map(lambda v: labels.get(v, v))
            miss = miss[["variable", "label", "share_missing"]].sort_values("share_missing", ascending=False)
            miss["share_missing"] = miss["share_missing"].map(lambda v: fmt(v, 3))
            msg.append(miss.to_string(index=False))
            with open(f"./output/{model_name}_ERROR.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(msg) + "\n")

            table = pd.DataFrame(
                {
                    "Independent Variable": [labels.get(p, p) for p in predictors],
                    "Std_Beta": [np.nan] * len(predictors),
                    "Replication_Sig": [""] * len(predictors),
                    "Included_in_model": [False] * len(predictors),
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
                    "variable": [dv_col] + predictors,
                    "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors],
                    "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors],
                    "unique_values_in_analytic_sample": [0] + [0] * len(predictors),
                }
            )
            table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
            fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)
            diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)
            return table, fit_stats, d, diag

        y = d[dv_col].astype(float)

        # Unstandardized OLS for intercept/R2/p-values (stars are "replication stars")
        X = d[kept].astype(float) if kept else pd.DataFrame(index=d.index)
        Xc = sm.add_constant(X, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas: z(Y) on z(X), no intercept
        betas = pd.Series(index=kept, dtype="float64")
        fit_beta = None
        if kept:
            y_z = zscore(y).rename("y_z")
            X_z = pd.DataFrame({p: zscore(d[p]) for p in kept}, index=d.index)
            dz = pd.concat([y_z, X_z], axis=1).dropna()
            # remove any zero-variance z predictors after listwise
            kept2 = [p for p in kept if dz[p].std(ddof=0) > 0]
            if len(dz) > 0 and dz["y_z"].std(ddof=0) > 0 and kept2:
                fit_beta = sm.OLS(dz["y_z"].astype(float), dz[kept2].astype(float)).fit()
                betas = fit_beta.params.reindex(kept2)

        # Stars from unstandardized p-values
        pvals = fit_unstd.pvalues
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        # Table in original predictor order
        rows = []
        for p in predictors:
            inc = p in betas.index
            rows.append(
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": float(betas.get(p, np.nan)) if inc else np.nan,
                    "Replication_Sig": stars.get(p, "") if (p in kept) else "",
                    "Included_in_model": bool(p in kept),
                }
            )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Replication_Sig": [const_star],
                "Dropped_all_missing": [", ".join(dropped_all_missing) if dropped_all_missing else ""],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        diag = pd.DataFrame(
            {
                "variable": [dv_col] + predictors,
                "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors],
                "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors],
                "unique_values_in_analytic_sample": [safe_unique_count(d[dv_col])]
                + [safe_unique_count(d[p]) if p in d.columns else 0 for p in predictors],
            }
        )

        # Write human-readable outputs
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS.")
        lines.append("Reported coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: regress z(Y) on z(X) with no intercept, using the analytic (listwise) sample.")
        lines.append("Significance: two-tailed p-values from unstandardized OLS (replication stars; may differ from paper).")
        lines.append("")
        lines.append("Construction notes:")
        lines.append("- Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV is strict count across component items (missing if any component missing within that DV).")
        lines.append("- Racism score is strict sum of 5 dichotomies (missing if any component missing).")
        lines.append("- Missing data: listwise deletion on DV + usable predictors; predictors that are 100% missing are dropped.")
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
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Replication_Sig", "Dropped_all_missing", "Dropped_no_variation"]].copy()
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

    # Fit both models
    m1_table, m1_fit, m1_frame, m1_diag = fit_model(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame, m2_diag = fit_model(dv2_col, "Table2_ModelB_Remaining12")

    # Combined output
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

    # Write combined summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Implementation summary")
    lines.append("----------------------")
    lines.append("- Year filter: year == 1993")
    lines.append("- DVs: strict dislike counts (missing if any component rating missing within DV set)")
    lines.append("- Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing")
    lines.append("- Racism score: strict sum of 5 dichotomies (0–5; missing if any component missing)")
    lines.append("- Estimation: OLS; standardized coefficients computed as z(Y) on z(X), no intercept")
    lines.append("- Stars: from unstandardized OLS p-values (replication stars; not the paper's stars)")
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
    fit_disp["N"] = fit_disp["N"].map(lambda v: fmt(v, 0))
    for c in ["R2", "Adj_R2", "Constant"]:
        fit_disp[c] = fit_disp[c].map(lambda v: fmt(v, 3))
    lines.append(fit_disp.to_string())
    lines.append("")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    m1_diag.to_csv("./output/Table2_ModelA_MinorityLinked6_diagnostics.csv", index=False)
    m2_diag.to_csv("./output/Table2_ModelB_Remaining12_diagnostics.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_fit": m1_fit,
        "modelB_fit": m2_fit,
        "modelA_analytic_frame": m1_frame,
        "modelB_analytic_frame": m2_frame,
        "modelA_diagnostics": m1_diag,
        "modelB_diagnostics": m2_diag,
    }