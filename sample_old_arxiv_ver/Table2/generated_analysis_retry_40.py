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

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # Coerce numeric (keep id)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Required columns per mapping
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]
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
        else missing.
        """
        x = pd.to_numeric(series, errors="coerce")
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
        # Missing if ANY component missing
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

    def value_counts_all(s):
        return pd.Series(s).value_counts(dropna=False)

    def standardized_betas_from_fit(fit, d, ycol, xcols):
        """
        Conventional standardized OLS betas:
        beta_j = b_j * sd(X_j) / sd(Y), computed on analytic sample used for fit.
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

    def hispanic_from_ethnic(series):
        """
        Create a Hispanic indicator from ETHNIC using a robust, data-driven heuristic.

        Tries common GSS-style ETHNIC codings:
          - 20-29 => Hispanic
          - 30-39 => Hispanic
          - 10-19 => Hispanic

        If none of these ranges exist in the data, fall back to a very conservative rule:
          - if a single small code dominates (e.g., 1) and other codes exist, we DO NOT guess.
          - return missing for all (so the model will be explicit about non-replicability).

        Rationale:
        - Prior bug: forcing constant 0 made the variable drop for no variation.
        - We must not silently misclassify if coding is unknown.
        """
        eth = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=eth.index, dtype="float64")
        if eth.dropna().empty:
            return out

        # treat integer-like codes
        eth_floor = np.floor(eth)
        is_int = eth.notna() & (np.abs(eth - eth_floor) < 1e-9)
        if is_int.sum() == 0:
            return out

        e = eth_floor.loc[is_int].astype(int)
        unique = set(e.unique().tolist())

        def apply_range(lo, hi):
            out.loc[is_int] = ((e >= lo) & (e <= hi)).astype(float)
            return out

        # Try common ranges in descending likelihood.
        for lo, hi in [(20, 29), (30, 39), (10, 19)]:
            if any((u >= lo and u <= hi) for u in unique):
                return apply_range(lo, hi)

        return out

    # -----------------------------
    # Dependent variables (strict dislike counts)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])     # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])    # 0..12

    # -----------------------------
    # Racism score (0–5): strict 5/5 items (per mapping)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no edu chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # 0..5

    # -----------------------------
    # Controls / dummies (preserve missing; strict listwise in model)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    # female: 1 if sex==2, 0 if sex==1, else missing
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    df["age_years"] = df["age"]

    # race dummies: reference=white; preserve missing if race not in {1,2,3}
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # hispanic from ETHNIC (do NOT force to 0 if unknown scheme)
    df["hispanic"] = hispanic_from_ethnic(df["ethnic"])

    # Conservative Protestant: cannot exactly reproduce without detailed tradition coding.
    # Use the available RELIG + DENOM as a minimal proxy; preserve missing (do NOT fill with 0).
    df["cons_prot"] = np.nan
    m = df["relig"].notna() & df["denom"].notna()
    df.loc[m, "cons_prot"] = ((df.loc[m, "relig"] == 1) & (df.loc[m, "denom"] == 1)).astype(float)

    # No religion: preserve missing (fixes prior "dropped for no variation" caused by over-filtering/filling)
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern: preserve missing
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
        "racism_score": "Racism score (0–5)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age_years": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (from ETHNIC; coding-dependent)",
        "other_race": "Other race (1=Other)",
        "cons_prot": "Conservative Protestant (RELIG==1 & DENOM==1 proxy)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting: strict listwise deletion on DV + all predictors
    # -----------------------------
    def fit_model(dv_col, model_name, file_stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # Save a flow summary to help diagnose sample-size discrepancies
        flow_lines = []
        flow_lines.append(f"{model_name} sample flow")
        flow_lines.append("=" * (len(model_name) + 12))
        flow_lines.append(f"N_total_1993: {int(df.shape[0])}")
        flow_lines.append(f"N_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        for p in predictors:
            flow_lines.append(f"N_nonmissing_{p}: {int(df[p].notna().sum())}")
        flow_lines.append(f"N_listwise_model_frame: {int(d.shape[0])}")
        write_text(f"./output/{file_stub}_sample_flow.txt", "\n".join(flow_lines))

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after strict listwise deletion.\n\n"
                f"Missingness shares (1993) for model columns:\n"
                f"{d0.isna().mean().sort_values(ascending=False).to_string()}\n"
            )
            write_text(f"./output/{file_stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop no-variation predictors AFTER listwise selection (prevents singular matrices)
        kept, dropped_no_var = [], []
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

        betas = standardized_betas_from_fit(fit, d, dv_col, kept)

        rows = []
        for p in predictors:
            rows.append(
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": betas.get(p, np.nan) if p in kept else np.nan,
                    "Sig": star(fit.pvalues.get(p, np.nan)) if p in kept else "",
                    "Note": ("dropped (no variation)" if p in dropped_no_var else ""),
                }
            )
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
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable table
        title = f"Bryson (1996) Table 2 — {model_name} (replication attempt)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: beta_j = b_j * sd(X_j)/sd(Y) on the analytic (listwise) sample.")
        lines.append("Significance stars: two-tailed p-values from the unstandardized OLS regression (this run).")
        lines.append("")
        lines.append("Construction rules used:")
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

        lines.append("Standardized coefficients (Table 2 style)")
        lines.append("---------------------------------------")
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

        # Diagnostics to confirm variation (especially no_religion + hispanic)
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
        diag_lines.append("Value counts in analytic sample (key dummies):")
        for v in ["female", "black", "hispanic", "other_race", "cons_prot", "no_religion", "southern"]:
            diag_lines.append(f"\n{v} ({labels.get(v, v)}):")
            diag_lines.append(value_counts_all(d[v]).to_string())

        diag_lines.append("")
        diag_lines.append("nunique (analytic sample) by predictor:")
        nun = d[predictors].nunique(dropna=True).sort_values()
        diag_lines.append(nun.to_string())

        write_text(f"./output/{file_stub}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{file_stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{file_stub}_fit.csv", index=False)

        return table, fit_stats, d

    m1_table, m1_fit, m1_d = fit_model(dv1, "Model 1 (DV1: minority-linked 6)", "Table2_Model1_MinorityLinked6")
    m2_table, m2_fit, m2_d = fit_model(dv2, "Model 2 (DV2: remaining 12)", "Table2_Model2_Remaining12")

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

    # Quickcheck distributions (pre-listwise) for debugging coding/scope issues
    qc = []
    qc.append("Quickcheck distributions (1993 sample, pre-listwise)")
    qc.append("====================================================")
    qc.append("")
    qc.append("RELIG value counts:")
    qc.append(value_counts_all(df["relig"]).to_string())
    qc.append("")
    qc.append("DENOM value counts:")
    qc.append(value_counts_all(df["denom"]).to_string())
    qc.append("")
    qc.append("REGION value counts:")
    qc.append(value_counts_all(df["region"]).to_string())
    qc.append("")
    qc.append("ETHNIC value counts (top 50):")
    qc.append(value_counts_all(df["ethnic"]).head(50).to_string())
    qc.append("")
    qc.append("Derived dummies value counts (pre-listwise):")
    for v in ["female", "black", "hispanic", "other_race", "cons_prot", "no_religion", "southern"]:
        qc.append(f"\n{v} ({labels.get(v, v)}):")
        qc.append(value_counts_all(df[v]).to_string())

    qc.append("")
    qc.append("DV1 distribution:")
    qc.append(value_counts_all(df[dv1]).sort_index().to_string())
    qc.append("")
    qc.append("DV2 distribution:")
    qc.append(value_counts_all(df[dv2]).sort_index().to_string())
    qc.append("")
    qc.append("Racism score distribution:")
    qc.append(value_counts_all(df["racism_score"]).sort_index().to_string())
    write_text("./output/quickcheck_distributions.txt", "\n".join(qc))

    # Combined human-readable summary
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (GSS 1993 extract provided)"
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
    lines.append("Notes:")
    lines.append("- Missing values are preserved; models use strict listwise deletion on DV + all predictors (closer to published-table practice).")
    lines.append("- Hispanic is derived from ETHNIC only when the coding scheme is detectable in this extract; otherwise left missing (and will reduce N).")
    lines.append("- Conservative Protestant uses the best-available proxy from the extract (RELIG + DENOM); if your extract lacks detailed tradition coding, exact replication may be impossible.")
    lines.append("- Stars reflect this run's p-values (not copied from the paper).")
    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "model1_table": m1_table,
        "model2_table": m2_table,
        "model1_analytic_sample": m1_d,
        "model2_analytic_sample": m2_d,
    }