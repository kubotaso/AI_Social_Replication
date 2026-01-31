def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    # Conservative missing sentinels often seen in GSS extracts
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(series, valid):
        s = mask_missing(series)
        return s.where(s.isin(list(valid)), np.nan)

    def zscore(series):
        s = series.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    # Political intolerance coding per mapping
    def intolerance_indicator(col, s):
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            if col == "colcom":
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    def build_polintol_strict_complete(df, tol_items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        pol = intoler.sum(axis=1, min_count=len(tol_items))  # NaN unless all answered
        return pol.astype(float), intoler

    def build_display_table(coef_long, fitstats, model_names, row_order, label_map, dv_label):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_names).fillna("—")
        wide.index = [label_map.get(t, t) for t in wide.index]

        fit = fitstats.set_index("model").reindex(model_names)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        return pd.concat([header, wide, extra], axis=0)

    def fit_table1_model(d, y, xvars, model_name):
        """
        - Listwise deletion within model.
        - Raw OLS for fit stats + p-values.
        - Standardized betas via re-fitting on z-scored Y and X within estimation sample.
        """
        cols = [y] + xvars
        dd = d.loc[:, cols].dropna(how="any").copy()
        if dd.empty:
            coef = pd.DataFrame({"model": model_name, "term": xvars, "beta_std": np.nan, "p_raw": np.nan, "cell": "—"})
            fit = pd.DataFrame([{"model": model_name, "N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan}])
            return None, coef, fit, dd

        X = sm.add_constant(dd[xvars].astype(float), has_constant="add")
        y_raw = dd[y].astype(float)
        res_raw = sm.OLS(y_raw, X).fit()

        dz = dd.copy()
        dz[y] = zscore(dz[y])
        for v in xvars:
            dz[v] = zscore(dz[v])
        dz = dz.dropna(subset=[y] + xvars)

        Xz = sm.add_constant(dz[xvars].astype(float), has_constant="add")
        yz = dz[y].astype(float)
        res_z = sm.OLS(yz, Xz).fit()

        rows = []
        for v in xvars:
            b_std = float(res_z.params.get(v, np.nan))
            p = float(res_raw.pvalues.get(v, np.nan))
            cell = "—" if not np.isfinite(b_std) else f"{b_std:.3f}{stars(p)}"
            rows.append({"model": model_name, "term": v, "beta_std": b_std, "p_raw": p, "cell": cell})
        coef = pd.DataFrame(rows)

        fit = pd.DataFrame([{
            "model": model_name,
            "N": int(dd.shape[0]),
            "R2": float(res_raw.rsquared),
            "Adj_R2": float(res_raw.rsquared_adj),
            "const_raw": float(res_raw.params.get("const", np.nan))
        }])
        return res_raw, coef, fit, dd

    def missingness_table(d, vars_):
        rows = []
        for v in vars_:
            rows.append({"var": v, "nonmissing": int(d[v].notna().sum()), "missing": int(d[v].isna().sum())})
        return pd.DataFrame(rows).sort_values(["missing", "var"], ascending=[False, True])

    # -------------------------
    # Load + restrict to 1993
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower().strip() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Required column missing: year")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Variable lists (per mapping)
    # -------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    tol_items = [
        "spkath", "colath", "libath",
        "spkrac", "colrac", "librac",
        "spkcom", "colcom", "libcom",
        "spkmil", "colmil", "libmil",
        "spkhomo", "colhomo", "libhomo"
    ]
    core_needed = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "relig", "denom", "region"
    ]
    needed = core_needed + music_items + tol_items
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music ratings: keep 1..5
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # Predictors / components
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    df["sex"] = keep_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_valid(df["race"], {1, 2, 3})
    df["relig"] = keep_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = keep_valid(df["region"], {1, 2, 3, 4})

    # Tolerance items: keep only valid substantive codes by item type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:
            # col* expected 4/5 in this extract
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count disliked, complete-case on all 18 items
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs (faithful to mapping; avoid guessing Hispanic from ETHNIC)
    # -------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference omitted)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: not constructible from provided mapping; keep as all-missing so it won't affect Ns if excluded
    # (Model 2/3 in the paper include it, but we cannot reproduce without a proper Hispanic-origin variable.)
    d["hispanic"] = np.nan

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 (Protestant) & DENOM in {1,6,7}
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across 15 items, sum 0..15
    d["political_intolerance"], intoler_df = build_polintol_strict_complete(d, tol_items)

    # -------------------------
    # Models (exact variable sets per summary)
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res1, tab1, fit1, dd1 = fit_table1_model(d, y, x_m1, model_names[0])
    res2, tab2, fit2, dd2 = fit_table1_model(d, y, x_m2, model_names[1])
    res3, tab3, fit3, dd3 = fit_table1_model(d, y, x_m3, model_names[2])

    coef_long = pd.concat([tab1, tab2, tab3], ignore_index=True)
    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Table 1-style output (standardized betas only; labeled rows; em-dash for not in model)
    # -------------------------
    label_map = {
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "conservative_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "political_intolerance": "Political intolerance",
    }
    row_order = [
        "educ", "income_pc", "prestg80",
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern",
        "political_intolerance",
    ]
    dv_label = "Number of music genres disliked"
    table1 = build_display_table(coef_long, fitstats, model_names, row_order, label_map, dv_label)

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1_listwise": int(dd1.shape[0]),
        "N_model2_listwise": int(dd2.shape[0]),
        "N_model3_listwise": int(dd3.shape[0]),
        "note_hispanic": "Hispanic not constructible from provided mapping; set to NaN (will reduce Model 2/3 N if included).",
        "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
        "income_pc_nonmissing_in_dv_complete": int(d["income_pc"].notna().sum()),
        "prestg80_nonmissing_in_dv_complete": int(d["prestg80"].notna().sum()),
    }])

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs
    # -------------------------
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    summary_lines.append("")
    summary_lines.append(f"Dependent variable (DV): {dv_label}")
    summary_lines.append("DV construction: 18 music items; disliked = 4 or 5; requires complete responses on all 18 items.")
    summary_lines.append("")
    summary_lines.append("Models: OLS on raw variables; standardized betas computed by re-fitting OLS on z-scored DV and predictors")
    summary_lines.append("within each model estimation sample (after listwise deletion).")
    summary_lines.append("Table output: standardized betas only (no standard errors).")
    summary_lines.append("Stars: from raw-model two-tailed p-values (* p<.05, ** p<.01, *** p<.001).")
    summary_lines.append("")
    summary_lines.append("Important limitation for exact paper replication:")
    summary_lines.append("  Hispanic origin variable is not available in the provided fields; 'hispanic' is set to missing.")
    summary_lines.append("  This will reduce Model 2/3 estimation Ns relative to the paper if Hispanic is included as in Table 1.")
    summary_lines.append("")
    summary_lines.append("Political intolerance:")
    summary_lines.append("  Strict complete-case across 15 tolerance items; sum of intolerant responses (0–15).")
    summary_lines.append("")
    summary_lines.append("Table 1-style standardized coefficients:")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Fit statistics (raw OLS):")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Diagnostics:")
    summary_lines.append(diag.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness (within DV-complete sample):")
    summary_lines.append("\nModel 1 vars:\n" + miss_m1.to_string(index=False))
    summary_lines.append("\nModel 2 vars:\n" + miss_m2.to_string(index=False))
    summary_lines.append("\nModel 3 vars:\n" + miss_m3.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Raw OLS summaries (debug):")
    summary_lines.append("\n==== Model 1 (SES) ====\n" + (res1.summary().as_text() if res1 is not None else "No model fit."))
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + (res2.summary().as_text() if res2 is not None else "No model fit."))
    summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + (res3.summary().as_text() if res3 is not None else "No model fit."))

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw-model p-values.\n")
        f.write("— indicates predictor not included or missing.\n\n")
        f.write(table1.to_string())
        f.write("\n")

    # Machine-readable outputs
    table1.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)
    miss_m1.to_csv("./output/missingness_m1.tsv", sep="\t", index=False)
    miss_m2.to_csv("./output/missingness_m2.tsv", sep="\t", index=False)
    miss_m3.to_csv("./output/missingness_m3.tsv", sep="\t", index=False)

    return {
        "table1_style": table1,
        "fit_stats": fitstats,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "missingness_m1": miss_m1,
        "missingness_m2": miss_m2,
        "missingness_m3": miss_m3,
        "estimation_samples": {"m1": dd1, "m2": dd2, "m3": dd3},
    }