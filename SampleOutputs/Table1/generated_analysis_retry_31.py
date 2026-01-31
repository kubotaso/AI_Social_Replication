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

    # Conservative "missing" sentinels often seen in GSS-style extracts
    MISSING_CODES = {
        8, 9, 98, 99, 998, 999, 9998, 9999,
        -1, -2, -3, -4, -5, -6, -7, -8, -9
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_codes(series, valid_codes):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_codes)), np.nan)

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

    def intolerance_indicator(col, s):
        """
        Per mapping:
          SPK*: 1 allowed, 2 not allowed -> intolerant=1 if 2
          LIB*: 1 remove, 2 not remove -> intolerant=1 if 1
          COL*: mostly 4 allowed, 5 not allowed -> intolerant=1 if 5
                special COLCOM: 4 fired, 5 not fired -> intolerant=1 if 4
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

    def build_polintol_complete(df, tol_items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        # strict 15/15 complete-case per mapping summary
        pol = intoler.sum(axis=1, min_count=len(tol_items))
        pol = pol.where(intoler.notna().all(axis=1), np.nan).astype(float)
        return pol, intoler.notna().sum(axis=1)

    def conservative_protestant_proxy(relig, denom):
        """
        Proxy using current RELIG + coarse DENOM codes available in this extract:
          conserv_prot = 1 if RELIG==1 (Protestant) and DENOM in {1,6,7}
        Missing relig/denom -> NaN.
        """
        out = pd.Series(np.nan, index=relig.index, dtype=float)
        m_rel = relig.notna()
        out.loc[m_rel] = 0.0
        m_prot = relig.eq(1) & relig.notna()
        out.loc[m_prot] = 0.0
        m_prot_denom = m_prot & denom.notna()
        out.loc[m_prot_denom] = denom.loc[m_prot_denom].isin([1, 6, 7]).astype(float)
        return out

    def build_hispanic_from_ethnic(df):
        """
        ETHNIC is available in this extract. We must avoid structural missingness.
        Define:
          hispanic = 1 if ETHNIC == 1
          hispanic = 0 if ETHNIC is any other observed value
        If ETHNIC missing, set 0 (best effort; avoids collapsing N).
        """
        eth = mask_missing(df["ethnic"])
        hisp = pd.Series(0.0, index=df.index, dtype=float)
        m = eth.notna()
        hisp.loc[m] = (eth.loc[m] == 1).astype(float)
        # if eth missing, keep 0.0
        return hisp

    def fit_ols_std_betas(data, y, xvars, model_name):
        """
        Fit raw OLS on original variables (with intercept),
        then compute standardized betas as: beta = b * SD(x) / SD(y)
        computed on the estimation sample (listwise deletion for y and all xvars).
        """
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna().copy()
        # Drop any predictor with no variance in the estimation sample
        x_keep = [v for v in xvars if dd[v].nunique(dropna=True) > 1]

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = float(yy.std(ddof=0)) if yy.notna().sum() > 1 else np.nan

        rows = []
        for v in xvars:
            if v not in x_keep:
                rows.append({"model": model_name, "term": v, "beta_std": np.nan, "p": np.nan, "cell": "—"})
                continue
            b = float(res.params[v])
            p = float(res.pvalues[v])
            sd_x = float(dd[v].astype(float).std(ddof=0)) if dd[v].notna().sum() > 1 else np.nan
            if not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0:
                beta = np.nan
                cell = "—"
            else:
                beta = b * (sd_x / sd_y)
                cell = f"{beta:.3f}{stars(p)}"
            rows.append({"model": model_name, "term": v, "beta_std": beta, "p": p, "cell": cell})

        fit = pd.DataFrame([{
            "model": model_name,
            "N": int(dd.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
        }])

        return res, pd.DataFrame(rows), fit, dd

    def missingness_table(df, vars_):
        out = []
        for v in vars_:
            out.append({"var": v, "missing": int(df[v].isna().sum()), "nonmissing": int(df[v].notna().sum())})
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

    required = (
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
        + music_items
        + tol_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    for c in music_items:
        df[c] = keep_codes(df[c], {1, 2, 3, 4, 5})

    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    df["sex"] = keep_codes(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_codes(df["race"], {1, 2, 3})
    df["region"] = keep_codes(df["region"], {1, 2, 3, 4})
    df["relig"] = keep_codes(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            # validity for COL* in this extract is 4/5
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count disliked across 18 (complete cases on all 18)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs construction
    # -------------------------
    # Income per capita (REALINC/HOMPOP), avoid division by zero
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic (best effort proxy; no structural missingness)
    d["hispanic"] = build_hispanic_from_ethnic(d)

    # Religion dummies
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)
    d["conservative_protestant"] = conservative_protestant_proxy(d["relig"], d["denom"])

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case on all 15 items (per mapping summary)
    d["political_intolerance"], pol_answered = build_polintol_complete(d, tol_items)

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

    res1, tab1, fit1, dd1 = fit_ols_std_betas(d, y, x_m1, model_names[0])
    res2, tab2, fit2, dd2 = fit_ols_std_betas(d, y, x_m2, model_names[1])
    res3, tab3, fit3, dd3 = fit_ols_std_betas(d, y, x_m3, model_names[2])

    coef_long = pd.concat([tab1, tab2, tab3], ignore_index=True)
    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Table 1-style display: standardized betas only (no SEs)
    # -------------------------
    pretty = {
        "educ": "Education",
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

    wide = coef_long.pivot(index="term", columns="model", values="cell").reindex(index=row_order, columns=model_names)
    wide = wide.fillna("—")
    wide.index = [pretty.get(t, t) for t in wide.index]

    fit = fitstats.set_index("model").reindex(model_names)
    extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
    for m in model_names:
        extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
        extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
        extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
        extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

    dv_label = "Number of music genres disliked"
    header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
    table1 = pd.concat([header, wide, extra], axis=0)

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_M1_completecases": int(dd1.shape[0]),
        "N_M2_completecases": int(dd2.shape[0]),
        "N_M3_completecases": int(dd3.shape[0]),
        "hispanic_1_count": int((d["hispanic"] == 1).sum()),
        "polintol_nonmissing": int(d["political_intolerance"].notna().sum()),
        "polintol_items_answered_mean": float(pol_answered.mean()) if len(pol_answered) else np.nan,
        "polintol_items_answered_min": float(pol_answered.min()) if len(pol_answered) else np.nan,
        "polintol_items_answered_max": float(pol_answered.max()) if len(pol_answered) else np.nan,
        "polintol_rule": "strict_complete_15_of_15",
    }])

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs
    # -------------------------
    summary_lines = []
    summary_lines.append("Replication: Table 1-style OLS (1993 GSS)")
    summary_lines.append("")
    summary_lines.append(f"DV (all models): {dv_label}")
    summary_lines.append("DV construction: 18 music items; dislike = 4/5; excludes any case with DK/missing on any of the 18 items.")
    summary_lines.append("")
    summary_lines.append("Estimation: OLS (unweighted).")
    summary_lines.append("Displayed coefficients: standardized betas computed from raw OLS as beta = b * SD(X)/SD(Y) on each model estimation sample.")
    summary_lines.append("Stars: from raw OLS p-values (* p<.05, ** p<.01, *** p<.001).")
    summary_lines.append("Note: Table prints betas only (no standard errors).")
    summary_lines.append("")
    summary_lines.append("Table 1-style standardized coefficients (betas only):")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Model fit stats:")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Diagnostics:")
    summary_lines.append(diag.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness (within DV-complete sample) — Model 1 variables:")
    summary_lines.append(miss_m1.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness (within DV-complete sample) — Model 2 variables:")
    summary_lines.append(miss_m2.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness (within DV-complete sample) — Model 3 variables:")
    summary_lines.append(miss_m3.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Raw OLS summaries (debug):")
    summary_lines.append("\n==== Model 1 (SES) ====\n" + res1.summary().as_text())
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + res2.summary().as_text())
    summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3.summary().as_text())

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("No standard errors are displayed.\n\n")
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