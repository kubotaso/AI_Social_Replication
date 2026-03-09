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

    # Restrict to 1993 as instructed
    df = df.loc[df["year"] == 1993].copy()

    # Coerce numeric for all but id
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
        + minority_genres + remaining_genres + racism_raw
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

    def hispanic_from_ethnic(eth):
        """
        Construct a Hispanic indicator using only ETHNIC (as available in this extract).

        Rules:
        - If ETHNIC missing => missing.
        - Else map a conservative set of common GSS Hispanic/Spanish-origin codes to 1.
        - All other nonmissing codes => 0.

        NOTE: The exact ETHNIC coding can vary by extract; diagnostics are written to ./output
        so the mapping can be refined without changing model code structure.
        """
        x = pd.to_numeric(eth, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        m = x.notna()
        if not m.any():
            return out

        # Default: all non-missing -> 0, then set selected to 1
        out.loc[m] = 0.0

        xi = np.floor(x[m]).astype("Int64")
        valid_int = (x[m] - xi.astype(float)).abs() < 1e-9
        idx = x[m].index[valid_int.values]
        e = xi[valid_int].astype(int)

        # Conservative "likely Hispanic" sets/ranges seen in some GSS recodes
        # (kept narrow to avoid misclassifying other ancestries).
        exact = set([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
        # Some extracts use 200-series for Hispanic/Spanish origin; only apply if present in data.
        ranges = [(200, 299), (700, 799)]

        present_vals = set(e.unique().tolist())
        is_hisp = e.isin(list(exact))

        for lo, hi in ranges:
            if any((v >= lo and v <= hi) for v in present_vals):
                is_hisp |= e.between(lo, hi)

        out.loc[idx] = is_hisp.astype(float).values
        # For non-integer-coded nonmissing ETHNIC, keep 0 (already set)
        return out

    # -----------------------------
    # Construct DVs (STRICT counts; missing if any component missing)
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
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])  # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])  # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])  # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])  # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])  # 1=yes willpower -> 1
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # 0..5

    # -----------------------------
    # Controls (preserve missingness; DO NOT impute missing to 0)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies (reference: White non-Hispanic in ideal data; here Hispanic comes from ETHNIC)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic (constructed from ETHNIC; missing preserved)
    df["hispanic"] = hispanic_from_ethnic(df["ethnic"])

    # No religion (RELIG==4), preserve missingness
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: cannot exactly reproduce Bryson with this extract; keep minimal, deterministic proxy.
    # Preserve missingness to maintain listwise deletion behavior.
    df["cons_prot"] = np.nan
    m_rel = df["relig"].notna() & df["denom"].notna()
    df.loc[m_rel, "cons_prot"] = ((df.loc[m_rel, "relig"] == 1) & (df.loc[m_rel, "denom"] == 1)).astype(float)

    # Southern (REGION==3 per provided mapping), preserve missingness
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
        "racism_score": "Racism score (0–5; sum of 5 dichotomies)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age_years": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (constructed from ETHNIC; mapping documented in diagnostics)",
        "other_race": "Other race (1=Other)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1; missing preserved)",
        "no_religion": "No religion (RELIG==4; missing preserved)",
        "southern": "Southern (REGION==3; missing preserved)",
        "const": "Constant",
    }

    # -----------------------------
    # Fit models (strict listwise deletion on DV + all predictors)
    # -----------------------------
    def fit_model(dv_col, model_name, file_stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # Sample flow + raw distributions for debugging / reconciliation
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

        # Drop no-variation predictors (prevents singular matrix / runtime errors)
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
                    "Note": ("dropped (no variation in analytic sample)" if p in dropped_no_var else ""),
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

        # Human-readable "Table 2 style" output
        title = f"Bryson (1996) Table 2 — {model_name} (computed from provided GSS 1993 extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_j = b_j * sd(X_j)/sd(Y), computed on this model's analytic sample.")
        lines.append("Stars: two-tailed p-values from this run's unstandardized OLS model (replication stars).")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Income per capita: REALINC/HOMPOP (HOMPOP must be >0)")
        lines.append("- Missing data: strict listwise deletion on DV + all predictors")
        lines.append("")
        if dropped_no_var:
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

        # Save full OLS summary for debugging
        with open(f"./output/{file_stub}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Diagnostics to pinpoint N collapse / dummy variation issues
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

        return table, fit_stats, d

    m1_table, m1_fit, m1_d = fit_model(dv1, "Model A (Minority-linked genres: 6)", "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_d = fit_model(dv2, "Model B (Remaining genres: 12)", "Table2_ModelB_Remaining12")

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

    # DV descriptives (before listwise deletion)
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

    # Combined summary (human-readable)
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
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
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("DV descriptives (constructed counts; before listwise deletion)")
    lines.append("-------------------------------------------------------------")
    dv_desc_fmt = dv_desc.copy()
    for c in ["DV1_Minority6", "DV2_Remaining12"]:
        dv_desc_fmt[c] = dv_desc_fmt[c].map(lambda v: f"{int(v)}" if isinstance(v, (int, np.integer)) else fmt(v, 3))
    lines.append(dv_desc_fmt.to_string(index=False))
    lines.append("")
    lines.append("Implementation notes (computed, not copied from paper)")
    lines.append("------------------------------------------------------")
    lines.append("- DVs are strict dislike counts: per-genre dislike = 1 if {4,5}, 0 if {1,2,3}; otherwise missing; DV missing if any component missing.")
    lines.append("- Racism score is a strict 5-item sum (0–5), missing if any component missing.")
    lines.append("- Missing handling: strict listwise deletion on DV + all predictors (no missing-to-0 imputation).")
    lines.append("- Hispanic is constructed from ETHNIC using a conservative code mapping; see *_diagnostics.txt for ETHNIC distributions.")
    lines.append("- Conservative Protestant uses a deterministic proxy from RELIG and broad DENOM recode; exact Bryson coding may require finer denomination detail not present here.")
    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_analytic_sample": m1_d,
        "modelB_analytic_sample": m2_d,
    }