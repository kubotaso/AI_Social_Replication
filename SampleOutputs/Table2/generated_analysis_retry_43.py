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

    # Restrict to 1993 only (Table 2)
    df = df.loc[df["year"] == 1993].copy()

    # Coerce numerics (keep id)
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
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region"]
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

    def standardized_betas_from_fit(fit, d, ycol, xcols):
        """
        Standardized OLS betas from an unstandardized OLS fit:
          beta_j = b_j * sd(X_j) / sd(Y)
        computed on the analytic sample used for the fit.
        """
        y = d[ycol].astype(float)
        y_sd = y.std(ddof=0)
        betas = {}
        for p in xcols:
            x = d[p].astype(float)
            x_sd = x.std(ddof=0)
            if pd.isna(y_sd) or y_sd == 0 or pd.isna(x_sd) or x_sd == 0:
                betas[p] = np.nan
            else:
                betas[p] = float(fit.params[p] * (x_sd / y_sd))
        return betas

    def value_counts_all(s):
        return pd.Series(s).value_counts(dropna=False)

    # -----------------------------
    # Construct DVs (STRICT counts; listwise on items via strict_sum)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (0-5): STRICT 5/5 components, per mapping
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no edu chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # 0..5

    # -----------------------------
    # Controls (preserve missingness; no "missing=>0" imputation)
    # -----------------------------
    df["education"] = df["educ"]
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]
    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: best-available from ETHNIC; keep simple + deterministic; preserve missingness
    # Rule: if ETHNIC is non-missing and integer-like, treat 20-29 or 30-39 as Hispanic (common coding blocks).
    eth = df["ethnic"]
    df["hispanic"] = np.nan
    m_eth = eth.notna()
    eth_floor = np.floor(eth.where(m_eth))
    is_int = m_eth & (np.abs(eth - eth_floor) < 1e-9)
    e = eth_floor.where(is_int)
    df.loc[is_int, "hispanic"] = (e.between(20, 29) | e.between(30, 39)).astype(float)
    # If ETHNIC non-missing but non-integer-like, code as 0 (keeps deterministic behavior without guessing categories)
    df.loc[m_eth & (~is_int), "hispanic"] = 0.0

    # No religion: RELIG==4, preserve missingness
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: use only provided RELIG+DENOM; preserve missingness (no imputation).
    # Minimal proxy with broad DENOM recode: Protestant (RELIG==1) and (Baptist (1) or Other Protestant (6)).
    df["cons_prot"] = np.nan
    m_rel = df["relig"].notna() & df["denom"].notna()
    df.loc[m_rel, "cons_prot"] = (
        (df.loc[m_rel, "relig"] == 1) & (df.loc[m_rel, "denom"].isin([1, 6]))
    ).astype(float)

    # Southern: REGION==3, preserve missingness
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictor order should match Table 2
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
        dv1: "Dislike of minority-linked genres (count of 6)",
        dv2: "Dislike of remaining genres (count of 12)",
        "racism_score": "Racism score (0–5; sum of 5 dichotomies)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age_years": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (proxy from ETHNIC)",
        "other_race": "Other race (1=Other)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM in {1,6})",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Fit models with strict listwise deletion on DV + all predictors
    # -----------------------------
    def fit_model(dv_col, model_name, file_stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # Sample flow audit (helps debug N mismatches)
        flow_lines = []
        flow_lines.append(f"{model_name} sample flow")
        flow_lines.append("=" * (len(model_name) + 12))
        flow_lines.append(f"N_total_1993: {int(df.shape[0])}")
        flow_lines.append(f"N_nonmissing_{dv_col}: {int(df[dv_col].notna().sum())}")
        for p in predictors:
            flow_lines.append(f"N_nonmissing_{p}: {int(df[p].notna().sum())}")
        flow_lines.append(f"N_listwise_model_frame: {int(d.shape[0])}")
        write_text(f"./output/{file_stub}_sample_flow.txt", "\n".join(flow_lines))

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                f"Missingness shares (1993) for model columns:\n"
                f"{d0.isna().mean().sort_values(ascending=False).to_string()}\n"
            )
            write_text(f"./output/{file_stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop no-variation predictors in analytic sample to avoid singular matrices; record them
        kept, dropped_no_var = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_fit(fit, d, dv_col, kept)

        table = pd.DataFrame(
            [
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": (betas.get(p, np.nan) if p in kept else np.nan),
                    "Sig": (star(fit.pvalues.get(p, np.nan)) if p in kept else ""),
                    "Note": ("dropped (no variation)" if p in dropped_no_var else ""),
                }
                for p in predictors
            ]
        )

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Sig": [star(fit.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable outputs
        title = f"Bryson (1996) Table 2 — {model_name} (replication attempt on provided extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_j = b_j * sd(X_j)/sd(Y) computed on the analytic sample.")
        lines.append("Stars: two-tailed p-values from this run's unstandardized OLS model (not copied from paper).")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Income per capita: REALINC/HOMPOP (HOMPOP must be >0)")
        lines.append("- Missing data: strict listwise deletion on DV + all predictors")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped due to no variation in analytic sample:")
            for p in dropped_no_var:
                lines.append(f"- {p}: {labels.get(p, p)}")
        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp[["Independent Variable", "Std_Beta", "Sig", "Note"]].to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats.copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].to_string(index=False))
        write_text(f"./output/{file_stub}_table2_style.txt", "\n".join(lines))

        with open(f"./output/{file_stub}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Diagnostics: ensure key dummies vary (and show raw value counts)
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
            diag_lines.append(f"\n{v} ({labels.get(v, v)}):")
            diag_lines.append(value_counts_all(d[v]).to_string())
        diag_lines.append("")
        diag_lines.append("Underlying raw distributions (1993, pre-listwise):")
        diag_lines.append("\nRELIG value counts:\n" + value_counts_all(df["relig"]).to_string())
        diag_lines.append("\nDENOM value counts:\n" + value_counts_all(df["denom"]).to_string())
        diag_lines.append("\nREGION value counts:\n" + value_counts_all(df["region"]).to_string())
        diag_lines.append("\nETHNIC value counts (top 50):\n" + value_counts_all(df["ethnic"]).head(50).to_string())
        diag_lines.append("\nRACE value counts:\n" + value_counts_all(df["race"]).to_string())
        diag_lines.append("")
        diag_lines.append("nunique (analytic sample) by predictor:")
        diag_lines.append(d[predictors].nunique(dropna=True).sort_values().to_string())
        write_text(f"./output/{file_stub}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{file_stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{file_stub}_fit.csv", index=False)

        return table, fit_stats, d, fit

    # Correct model naming per discrepancy report:
    # "Model 2A" -> Model 1; "Model 2B" -> Model 2
    m1_table, m1_fit, m1_d, m1_fitobj = fit_model(dv1, "Model 1 (DV1: minority-linked 6)", "Table2_Model1_MinorityLinked6")
    m2_table, m2_fit, m2_d, m2_fitobj = fit_model(dv2, "Model 2 (DV2: remaining 12)", "Table2_Model2_Remaining12")

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

    # DV descriptives (constructed counts before listwise deletion)
    dv_desc = pd.DataFrame(
        {
            "stat": ["N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"],
            "DV1_Minority6": [
                int(df[dv1].notna().sum()),
                df[dv1].mean(),
                df[dv1].std(ddof=0),
                df[dv1].min(),
                df[dv1].quantile(0.25),
                df[dv1].quantile(0.50),
                df[dv1].quantile(0.75),
                df[dv1].max(),
            ],
            "DV2_Remaining12": [
                int(df[dv2].notna().sum()),
                df[dv2].mean(),
                df[dv2].std(ddof=0),
                df[dv2].min(),
                df[dv2].quantile(0.25),
                df[dv2].quantile(0.50),
                df[dv2].quantile(0.75),
                df[dv2].max(),
            ],
        }
    )

    # Write combined summary
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars (from this run)")
    lines.append("--------------------------------------------------------------------------------------")
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
    lines.append("DV descriptives (constructed counts; before listwise deletion)")
    lines.append("-------------------------------------------------------------")
    dv_desc_fmt = dv_desc.copy()
    for c in ["DV1_Minority6", "DV2_Remaining12"]:
        dv_desc_fmt[c] = dv_desc_fmt[c].map(lambda v: f"{int(v)}" if isinstance(v, (int, np.integer)) else fmt(v, 3))
    lines.append(dv_desc_fmt.to_string(index=False))
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("- DVs are strict dislike counts: per-genre dislike = 1 if {4,5}, 0 if {1,2,3}; missing otherwise; DV missing if any component missing.")
    lines.append("- Racism score is a strict 5-item sum (0–5), missing if any component missing.")
    lines.append("- Listwise deletion is applied on DV + all predictors (no missing-to-0 imputation).")
    lines.append("- Hispanic and Conservative Protestant are best-available proxies using only fields in the provided extract; exact published coding may require variables not present here.")
    lines.append("- Stars are computed from this run's p-values and are not copied from the paper.")
    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc,
        "model1_table": m1_table,
        "model2_table": m2_table,
        "model1_analytic_sample": m1_d,
        "model2_analytic_sample": m2_d,
    }