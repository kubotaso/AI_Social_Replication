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

    for col in ["year", "id"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in the input CSV.")

    df = df.loc[df["year"] == 1993].copy()

    # Coerce to numeric where appropriate
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
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
        + minority_genres
        + remaining_genres
        + racism_items
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

    def standardized_betas_from_unstd(fit, d_analytic, y_col, x_cols):
        """
        Standardized betas derived from unstandardized slopes:
            beta_std_j = b_j * SD(X_j) / SD(Y)
        computed on analytic (listwise) sample with ddof=1.
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
    # Construct DVs (strict per mapping)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1_col = "dv1_minority6_dislikes"
    dv2_col = "dv2_remaining12_dislikes"
    df[dv1_col] = strict_sum(df, [f"d_{g}" for g in minority_genres])     # 0..6
    df[dv2_col] = strict_sum(df, [f"d_{g}" for g in remaining_genres])    # 0..12

    # -----------------------------
    # Racism score (0–5), strict sum (per mapping)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no education chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

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

    # Race dummies: White is reference; preserve missing when race unknown
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: derive from ETHNIC if available (best effort within provided variables)
    # Treat only clearly-Hispanic codes as 1; treat non-missing non-Hispanic as 0.
    # If ETHNIC is missing, leave missing (do not impute).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = df["ethnic"]
        # Common GSS ancestry/ethnic recode often uses:
        # 1=Mexican, 2=Puerto Rican, 3=Other Spanish, 4=Central/South American (varies).
        # We only flag 1..4 as Hispanic; anything else non-missing -> 0.
        hisp_codes = set([1, 2, 3, 4])
        df.loc[eth.notna() & eth.isin(list(hisp_codes)), "hispanic"] = 1.0
        df.loc[eth.notna() & (~eth.isin(list(hisp_codes))), "hispanic"] = 0.0

    # Conservative Protestant proxy using available broad denom recode: Protestant & Baptist
    df["cons_prot"] = np.nan
    rel = df["relig"]
    den = df["denom"]
    known = rel.notna() & den.notna()
    df.loc[known, "cons_prot"] = ((rel.loc[known] == 1) & (den.loc[known] == 1)).astype(float)

    # No religion
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictors in Table 2 order
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
        dv1_col: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (0–6 dislike count)",
        dv2_col: "Dislike of the 12 Remaining Genres (0–12 dislike count)",
        "racism_score": "Racism score (0–5; strict sum of 5 items)",
        "education": "Education",
        "income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic (derived from ETHNIC codes 1–4 when available)",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy: Protestant & Baptist)",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting: strict listwise deletion on DV + all predictors
    # (faithful to paper-style listwise handling; no missing->0 shortcuts)
    # -----------------------------
    def fit_table2(dv_col, model_name, model_number):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # If any predictor is entirely missing, we cannot estimate Table 2 faithfully.
        all_missing = [p for p in predictors if d0[p].notna().sum() == 0]

        # Listwise on all predictors regardless; this matches "same RHS in both models".
        # (If a predictor is all-missing, this will yield N=0; we handle with explicit error.)
        d = d0.dropna(subset=model_cols).copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n"
                f"Predictors with all values missing in this extract: {all_missing}\n"
                f"Fix: include/derive the missing variable(s) in the input dataset."
            )
            write_text(f"./output/{model_name}_ERROR.txt", [msg])
            raise ValueError(msg)

        # Drop true no-variation predictors to avoid singular matrix (rare; document if happens)
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

        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star(fit.pvalues.get(p, np.nan)),
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Sig": "",
                    }
                )
        table = pd.DataFrame(rows)

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
                "All_missing_predictors_in_extract": [", ".join(all_missing) if all_missing else ""],
            },
            index=[f"Model_{model_number}"],
        )

        # Write human-readable model output
        lines = []
        title = f"Bryson (1996) Table 2 replication (GSS 1993) — Model {model_number}"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_std_j = b_j * SD(X_j) / SD(Y), computed on analytic (listwise) sample (ddof=1).")
        lines.append("Stars: two-tailed p-values from this replication’s unstandardized OLS regression.")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Missing data: listwise deletion on DV + all predictors")
        if all_missing:
            lines.append("")
            lines.append("WARNING: The following predictors are entirely missing in this extract (cannot match Table 2 if present):")
            for p in all_missing:
                lines.append(f"- {p}: {labels.get(p, p)}")
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

        # Save full OLS summary
        with open(f"./output/{model_name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Diagnostics
        diag_lines = []
        diag_lines.append(f"{model_name} diagnostics")
        diag_lines.append("=" * (len(model_name) + 12))
        diag_lines.append(f"N_1993_total: {int(df.shape[0])}")
        diag_lines.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag_lines.append(f"N_analytic_listwise: {int(d.shape[0])}")
        diag_lines.append(f"All_missing_predictors_in_extract: {', '.join(all_missing) if all_missing else '(none)'}")
        diag_lines.append(f"Dropped_no_variation: {', '.join(dropped_no_var) if dropped_no_var else '(none)'}")
        diag_lines.append("")
        for v in predictors:
            diag_lines.append(f"{v}: missing_share_in_1993 = {fmt(df[v].isna().mean(), 3)}; unique_in_analytic = {int(d[v].nunique(dropna=True))}")
        diag_lines.append("")
        # quick counts for key dummies
        for v in ["black", "hispanic", "other_race", "cons_prot", "no_religion", "southern", "female"]:
            vc = d[v].value_counts(dropna=False).sort_index()
            diag_lines.append(f"{v} value counts (analytic):")
            diag_lines.append(vc.to_string())
            diag_lines.append("")
        write_text(f"./output/{model_name}_diagnostics.txt", diag_lines)

        return table, fit_stats, d

    m1_table, m1_fit, m1_frame = fit_table2(dv1_col, "Table2_Model1_MinorityLinked6", 1)
    m2_table, m2_fit, m2_frame = fit_table2(dv2_col, "Table2_Model2_Remaining12", 2)

    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "Model_1_Std_Beta": m1_table["Std_Beta"],
            "Model_1_Sig": m1_table["Sig"],
            "Model_2_Std_Beta": m2_table["Std_Beta"],
            "Model_2_Sig": m2_table["Sig"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # DV descriptives (raw available + analytic)
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

    # Missingness
    miss = df[[dv1_col, dv2_col] + predictors].isna().mean().reset_index()
    miss.columns = ["variable", "share_missing_1993"]
    miss["label"] = miss["variable"].map(lambda v: labels.get(v, v))
    miss = miss.sort_values("share_missing_1993", ascending=False)

    # Write combined summary
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars")
    lines.append("-----------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["Model_1_Std_Beta"] = tmp["Model_1_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["Model_2_Std_Beta"] = tmp["Model_2_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS)")
    lines.append("---------------------------------")
    fs = combined_fit[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation", "All_missing_predictors_in_extract"]].copy()
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

    write_text("./output/combined_summary.txt", lines)

    # Save combined artifacts
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss.to_csv("./output/missingness_1993.csv", index=False)
    dv_desc_df.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc_df,
        "missingness_1993": miss,
        "model1_table": m1_table,
        "model2_table": m2_table,
        "model1_fit": m1_fit,
        "model2_fit": m2_fit,
    }