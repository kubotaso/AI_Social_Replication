def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # I/O helpers
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")

    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return ""

    # -----------------------------
    # Core coding helpers
    # -----------------------------
    def dislike_indicator(s):
        # 1 if 4/5; 0 if 1/2/3; else missing
        x = pd.to_numeric(s, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(s, ones, zeros):
        # Map to {0,1}; anything else missing
        x = pd.to_numeric(s, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        # sum; missing if ANY component missing
        return dfin[cols].sum(axis=1, skipna=False)

    def zscore(s):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        m = s.mean()
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - m) / sd

    # -----------------------------
    # Load + normalize columns
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column.")
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    # numeric coercion (except id)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Required fields
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["year", "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # -----------------------------
    # Dependent variables (strict)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dislike_minority_linked_6"
    dv2 = "dislike_remaining_12"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])      # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])     # 0..12

    # -----------------------------
    # Racism score (0..5), strict
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])        # object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])        # oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])      # not discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])      # not education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])      # willpower -> 1
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls
    # -----------------------------
    df["education"] = df["educ"]

    # income per capita: REALINC / HOMPOP, strict presence of both and HOMPOP>0
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic: dataset lacks canonical field; derive best-effort from ETHNIC.
    # Keep it simple: flag as Hispanic if ETHNIC in [20..29]; observed ETHNIC but not in that range -> 0.
    eth = pd.to_numeric(df["ethnic"], errors="coerce")
    df["hispanic"] = np.where(eth.notna(), 0.0, np.nan)
    df.loc[eth.between(20, 29, inclusive="both"), "hispanic"] = 1.0
    df.loc[~race_known, "hispanic"] = np.nan  # keep race/ethnicity block consistent

    # Conservative Protestant proxy: RELIG==1 & DENOM==1 (Baptist); if RELIG observed but DENOM missing, treat as 0
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.where(rel.notna(), 0.0, np.nan)
    prot = rel.notna() & (rel == 1)
    df.loc[prot, "cons_prot"] = np.where(den.loc[prot].notna(), (den.loc[prot] == 1).astype(float), 0.0)

    df["no_religion"] = np.where(rel.notna(), (rel == 4).astype(float), np.nan)
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

    # -----------------------------
    # Fit model and produce standardized coefficients (beta weights)
    # IMPORTANT: Do not print stars; Table 2 stars are from the publication and
    # cannot be assumed identical to re-estimated stars from this extract.
    # -----------------------------
    def fit_model(dv_col, model_name, stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(axis=0, how="any").copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                "Missingness shares (YEAR==1993) for model columns:\n"
                + d0.isna().mean().sort_values(ascending=False).to_string()
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop no-variation predictors (safety)
        kept = []
        dropped_no_var = []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        # Unstandardized OLS on raw DV and raw X (for intercept)
        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas via re-fitting on z-scored y and x
        yz = zscore(d[dv_col])
        Xz = pd.DataFrame({p: zscore(d[p]) for p in kept})
        Xzc = sm.add_constant(Xz, has_constant="add")
        fit_std = sm.OLS(yz, Xzc).fit()

        # Build table (Table-2-like: coefficients only)
        var_order = [
            ("racism_score", "Racism score"),
            ("education", "Education"),
            ("income_pc", "Household income per capita"),
            ("occ_prestige", "Occupational prestige"),
            ("female", "Female"),
            ("age_years", "Age"),
            ("black", "Black"),
            ("hispanic", "Hispanic"),
            ("other_race", "Other race"),
            ("cons_prot", "Conservative Protestant"),
            ("no_religion", "No religion"),
            ("southern", "Southern"),
        ]

        rows = []
        for key, lab in var_order:
            if key in kept:
                rows.append({"Independent Variable": lab, "Standardized OLS coefficient (beta)": float(fit_std.params.get(key, np.nan))})
            else:
                rows.append({"Independent Variable": lab, "Standardized OLS coefficient (beta)": np.nan})

        # Intercept (unstandardized, comparable to paper's constant row)
        rows.append({"Independent Variable": "Constant", "Standardized OLS coefficient (beta)": float(fit_unstd.params.get("const", np.nan))})

        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [dv_col],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable summary
        lines = []
        lines.append(f"{model_name}")
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append("Data: provided GSS extract, YEAR==1993")
        lines.append(f"Dependent variable: {dv_col}")
        lines.append("Estimator: OLS (unweighted).")
        lines.append("Displayed coefficients:")
        lines.append("- Predictor rows: standardized OLS coefficients (beta weights), computed by OLS on z-scored DV and predictors.")
        lines.append("- Constant: unstandardized intercept from OLS on raw DV and raw predictors.")
        lines.append("Note: No significance stars are printed here; publication stars cannot be assumed to match re-estimated results.")
        lines.append("")
        lines.append(f"N (listwise on DV + predictors): {int(round(fit_unstd.nobs))}")
        lines.append(f"R2: {fmt(fit_unstd.rsquared, 3)}")
        lines.append(f"Adj R2: {fmt(fit_unstd.rsquared_adj, 3)}")
        if dropped_no_var:
            lines.append(f"Dropped for no variation: {', '.join(dropped_no_var)}")
        lines.append("")
        disp = table.copy()
        disp["Value"] = disp["Standardized OLS coefficient (beta)"].map(lambda v: fmt(v, 3))
        lines.append("Regression table")
        lines.append("----------------")
        lines.append(disp[["Independent Variable", "Value"]].to_string(index=False))
        lines.append("")
        lines.append("Unstandardized OLS summary (for reference)")
        lines.append("----------------------------------------")
        lines.append(fit_unstd.summary().as_text())

        write_text(f"./output/{stub}_summary.txt", "\n".join(lines))

        # Diagnostics to explain N
        miss = d0.isna().mean().sort_values(ascending=False)
        diag = []
        diag.append(f"{model_name} diagnostics")
        diag.append("=" * (len(model_name) + 12))
        diag.append(f"N_total_YEAR1993: {int(df.shape[0])}")
        diag.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag.append(f"N_listwise_model: {int(d.shape[0])}")
        diag.append("")
        diag.append("Missingness share by model column (descending):")
        diag.append(miss.map(lambda v: fmt(v, 3)).to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag))

        table.to_csv(f"./output/{stub}_table.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d

    # Run both models
    t1, f1, d1 = fit_model(
        dv1,
        "Model A: Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music",
        "table2_modelA_minority_linked_6",
    )
    t2, f2, d2 = fit_model(
        dv2,
        "Model B: Dislike of the 12 Remaining Genres",
        "table2_modelB_remaining_12",
    )

    # Combined output
    combined = t1[["Independent Variable"]].copy()
    combined = combined.merge(
        t1.rename(columns={"Standardized OLS coefficient (beta)": "ModelA"}),
        on="Independent Variable",
        how="left",
    ).merge(
        t2.rename(columns={"Standardized OLS coefficient (beta)": "ModelB"}),
        on="Independent Variable",
        how="left",
    )

    combined_fit = pd.concat([f1, f2], axis=0, ignore_index=True)

    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (computed from provided 1993 extract)")
    lines.append("============================================================================")
    lines.append("")
    lines.append("Combined coefficients (predictors are standardized betas; Constant is unstandardized intercept)")
    lines.append("-------------------------------------------------------------------------------------------")
    disp = combined.copy()
    disp["ModelA"] = disp["ModelA"].map(lambda v: fmt(v, 3))
    disp["ModelB"] = disp["ModelB"].map(lambda v: fmt(v, 3))
    lines.append(disp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics")
    lines.append("-------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("Important notes")
    lines.append("--------------")
    lines.append("- This function computes coefficients from the provided extract; it does not (and cannot) reproduce publication stars without the original author’s exact setup.")
    lines.append("- Hispanic is approximated from ETHNIC (20..29) because a canonical Hispanic flag is not present in the provided fields.")
    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)

    return {
        "modelA_table": t1,
        "modelB_table": t2,
        "combined_table": combined,
        "fit": combined_fit,
        "modelA_analytic": d1,
        "modelB_analytic": d2,
    }