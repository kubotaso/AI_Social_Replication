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

    # Conservative: treat common GSS sentinel codes as missing (do not treat 0 as missing)
    MISSING_CODES = {
        8, 9, 98, 99, 998, 999, 9998, 9999,
        -1, -2, -3, -4, -5, -6, -7, -8, -9
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(series, valid_set):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_set)), np.nan)

    def stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def intolerance_indicator(col, s):
        """
        Mapping instruction:
          SPK*: 1 allowed, 2 not allowed -> intolerant=1 if 2
          LIB*: 1 remove, 2 not remove -> intolerant=1 if 1
          COL*: 4 allowed, 5 not allowed -> intolerant=1 if 5
                COLCOM special: 4 fired, 5 not fired -> intolerant=1 if 4
        """
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)
        elif col.startswith("col"):
            m = s.isin([4, 5])
            if col == "colcom":
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    def fit_ols_standardized_betas(data, y, xvars, model_name):
        """
        Listwise deletion on y + xvars.
        OLS on raw variables; standardized betas computed on estimation sample using ddof=1:
            beta_j = b_j * SD(X_j) / SD(Y)
        """
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()
        if dd.shape[0] == 0:
            coef = pd.DataFrame({"model": model_name, "term": xvars, "cell": ["—"] * len(xvars),
                                 "beta_std": np.nan, "b_raw": np.nan, "p_raw": np.nan, "dropped_reason": "empty_sample"})
            fit = pd.DataFrame([{"model": model_name, "N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan,
                                 "note": "No complete cases for this model after listwise deletion."}])
            return None, coef, fit, dd

        # Drop zero-variance predictors to avoid singular matrices
        x_keep = [v for v in xvars if dd[v].nunique(dropna=True) > 1]
        dropped = [v for v in xvars if v not in x_keep]

        yy = dd[y].astype(float)

        if len(x_keep) == 0:
            X = np.ones((dd.shape[0], 1))
            res = sm.OLS(yy, X).fit()
            coef = pd.DataFrame({"model": model_name, "term": xvars, "cell": ["—"] * len(xvars),
                                 "beta_std": np.nan, "b_raw": np.nan, "p_raw": np.nan,
                                 "dropped_reason": "zero_variance_or_all_missing"})
            fit = pd.DataFrame([{"model": model_name, "N": int(dd.shape[0]), "R2": float(res.rsquared),
                                 "Adj_R2": float(res.rsquared_adj), "const_raw": float(res.params[0]),
                                 "note": "Intercept-only fit; all predictors dropped."}])
            return res, coef, fit, dd

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        res = sm.OLS(yy, X).fit()

        sd_y = float(yy.std(ddof=1)) if yy.notna().sum() >= 2 else np.nan

        rows = []
        for v in xvars:
            if v in x_keep:
                sx = float(dd[v].astype(float).std(ddof=1)) if dd[v].notna().sum() >= 2 else np.nan
                b = float(res.params.get(v, np.nan))
                p = float(res.pvalues.get(v, np.nan))
                beta = np.nan
                if np.isfinite(sx) and np.isfinite(sd_y) and sx != 0 and sd_y != 0:
                    beta = b * (sx / sd_y)
                cell = "—" if not np.isfinite(beta) else f"{beta:.3f}{stars_from_p(p)}"
                rows.append({"model": model_name, "term": v, "cell": cell,
                             "beta_std": beta, "b_raw": b, "p_raw": p, "dropped_reason": ""})
            else:
                rows.append({"model": model_name, "term": v, "cell": "—",
                             "beta_std": np.nan, "b_raw": np.nan, "p_raw": np.nan,
                             "dropped_reason": "zero_variance_or_all_missing"})

        fit = pd.DataFrame([{
            "model": model_name,
            "N": int(dd.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
            "note": ("Dropped (no variance): " + ", ".join(dropped)) if dropped else ""
        }])
        return res, pd.DataFrame(rows), fit, dd

    def build_table(coef_long, fitstats, model_names, row_order, pretty_row, dv_label):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_names).fillna("—")
        wide.index = [pretty_row.get(t, t) for t in wide.index]

        fit = fitstats.set_index("model").reindex(model_names)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        return pd.concat([header, wide, extra], axis=0)

    def missingness_table(data, vars_):
        out = []
        for v in vars_:
            out.append({"var": v, "nonmissing": int(data[v].notna().sum()), "missing": int(data[v].isna().sum())})
        return pd.DataFrame(out).sort_values(["missing", "var"], ascending=[False, True])

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
    # Variables per mapping
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
    core_needed = ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
    needed = core_needed + music_items + tol_items
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music: keep 1..5 only
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # Core
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])
    df["sex"] = keep_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_valid(df["race"], {1, 2, 3})
    df["ethnic"] = mask_missing(df["ethnic"])
    df["relig"] = keep_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = keep_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: keep only valid substantive codes by item type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            # treat extract as 4/5; if other codes exist, they become missing
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: count of genres disliked (complete cases on 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita: realinc / hompop, require hompop > 0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white is reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: construct from ETHNIC when available.
    # In this extract ETHNIC appears numeric; implement a minimal, robust rule:
    # treat ETHNIC==1 as Hispanic; all other non-missing values as non-Hispanic.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # Conservative Protestant proxy (as implementable with RELIG + coarse DENOM)
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across all 15 items, sum of intolerance indicators
    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    d["political_intolerance"] = intoler.sum(axis=1, min_count=len(tol_items)).astype(float)

    # -------------------------
    # Models
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res1, tab1, fit1, dd1 = fit_ols_standardized_betas(d, y, x_m1, model_names[0])
    res2, tab2, fit2, dd2 = fit_ols_standardized_betas(d, y, x_m2, model_names[1])
    res3, tab3, fit3, dd3 = fit_ols_standardized_betas(d, y, x_m3, model_names[2])

    coef_long = pd.concat([tab1, tab2, tab3], ignore_index=True)
    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Table output
    # -------------------------
    pretty_row = {
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
    table1 = build_table(coef_long, fitstats, model_names, row_order, pretty_row, dv_label)

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1": int(dd1.shape[0]),
        "N_model2": int(dd2.shape[0]),
        "N_model3": int(dd3.shape[0]),
        "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
        "hispanic_1_count": int((d["hispanic"] == 1).sum(skipna=True)),
        "political_intolerance_nonmissing": int(d["political_intolerance"].notna().sum()),
        "standardization_rule": "beta_j = b_j * SD(X_j)/SD(Y), SD ddof=1, computed on each model estimation sample.",
        "notes": "Stars from raw unweighted OLS p-values; paper may differ if weighted/design-based."
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
    summary_lines.append(f"Dependent variable: {dv_label}")
    summary_lines.append("DV construction: 18 music items; disliked = response 4 or 5; listwise complete on all 18 items.")
    summary_lines.append("")
    summary_lines.append("Model estimation: model-specific listwise deletion (drop missing only on variables in that model).")
    summary_lines.append("Displayed coefficients: standardized betas only (no standard errors).")
    summary_lines.append("Standardization: computed post-estimation on the estimation sample (ddof=1).")
    summary_lines.append("Stars: from raw unweighted OLS p-values (* p<.05, ** p<.01, *** p<.001).")
    summary_lines.append("")
    summary_lines.append("Political intolerance: strict complete-case across 15 tolerance items; sum of intolerant responses (0-15).")
    summary_lines.append("Hispanic: derived from ETHNIC==1 (all other non-missing ETHNIC treated as non-Hispanic).")
    summary_lines.append("")
    summary_lines.append("Table 1-style standardized coefficients:")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Model fit stats:")
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
    summary_lines.append("\n==== Model 1 (SES) ====\n" + (res1.summary().as_text() if res1 is not None else "No model fit (empty sample)."))
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + (res2.summary().as_text() if res2 is not None else "No model fit (empty sample)."))
    summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + (res3.summary().as_text() if res3 is not None else "No model fit (empty sample)."))

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw unweighted OLS p-values.\n")
        f.write("— indicates predictor not included / unavailable / dropped due to zero variance.\n\n")
        f.write(table1.to_string())
        f.write("\n")

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