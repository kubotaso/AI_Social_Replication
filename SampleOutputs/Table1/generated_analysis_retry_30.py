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

    # Common GSS-style missing sentinels; keep this conservative.
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

    def zscore(series):
        s = series.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=series.index, dtype=float)
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

    def build_polintol_allow_missing(df, tol_items, min_answered=10):
        """
        Many replications of GSS tolerance batteries allow some missingness.
        We implement a simple, documented rule:
          - compute intolerant indicator for each item
          - require at least `min_answered` non-missing items
          - scale = sum across answered items (missing contribute 0)
        This avoids severe N collapse versus strict 15/15.
        """
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        answered = intoler.notna().sum(axis=1)
        scale = pd.Series(np.nan, index=df.index, dtype=float)
        m = answered >= int(min_answered)
        scale.loc[m] = intoler.loc[m].fillna(0.0).sum(axis=1).astype(float)
        return scale, answered

    def conservative_protestant_proxy(relig, denom):
        """
        Proxy using current RELIG + coarse DENOM codes available in this extract:
          conserv_prot = 1 if RELIG==1 (Protestant) and DENOM in {1,6,7}
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
        Use ETHNIC as best available proxy in this dataset.
        IMPORTANT: do NOT override race dummies; that can flip signs depending on how ETHNIC is coded.
        We define:
          hispanic = 1 if ETHNIC == 1, else 0 when ETHNIC observed; missing stays missing.
        """
        eth = mask_missing(df["ethnic"])
        hisp = pd.Series(np.nan, index=df.index, dtype=float)
        m = eth.notna()
        hisp.loc[m] = (eth.loc[m] == 1).astype(float)
        return hisp

    def fit_table1_style(data, y, xvars, model_name):
        """
        Table 1 reports standardized coefficients (betas). A common approach consistent with nonzero intercept:
          - z-score predictors only (within model estimation sample)
          - regress raw DV on standardized predictors
          - betas are the coefficients on z-scored predictors
        """
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna().copy()

        # Standardize predictors only
        Xz = pd.DataFrame(index=dd.index)
        dropped = []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
                continue
            Xz[v] = zscore(dd[v])

        X = sm.add_constant(Xz, has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        # Build output rows in the original xvars order, using em dash for excluded/not estimable
        rows = []
        for v in xvars:
            if v not in Xz.columns or v not in res.params.index:
                rows.append({"model": model_name, "term": v, "beta": np.nan, "p": np.nan, "cell": "—"})
            else:
                b = float(res.params[v])
                p = float(res.pvalues[v])
                rows.append({"model": model_name, "term": v, "beta": b, "p": p, "cell": f"{b:.3f}{stars(p)}"})

        fit = {
            "model": model_name,
            "N": int(dd.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
            "dropped_no_variance": ", ".join(dropped) if dropped else "",
        }
        return res, pd.DataFrame(rows), pd.DataFrame([fit]), dd

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

    # Race dummies (White reference): do NOT force mutual exclusivity with Hispanic proxy
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic (proxy from ETHNIC)
    d["hispanic"] = build_hispanic_from_ethnic(d)

    # Religion dummies
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)
    d["conservative_protestant"] = conservative_protestant_proxy(d["relig"], d["denom"])

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: allow partial completion to reduce N collapse (paper's Model 3 N is larger than strict-15 in many extracts)
    d["political_intolerance"], pol_answered = build_polintol_allow_missing(d, tol_items, min_answered=10)

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

    res1, tab1, fit1, dd1 = fit_table1_style(d, y, x_m1, model_names[0])
    res2, tab2, fit2, dd2 = fit_table1_style(d, y, x_m2, model_names[1])
    res3, tab3, fit3, dd3 = fit_table1_style(d, y, x_m3, model_names[2])

    coef_long = pd.concat([tab1, tab2, tab3], ignore_index=True)
    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Table 1-style display: betas only (no SE rows)
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
    extra = pd.DataFrame(index=["Constant", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
    for m in model_names:
        extra.loc["Constant", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
        extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
        extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
        extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

    dv_label = "Number of music genres disliked"
    header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
    table1 = pd.concat([header, wide, extra], axis=0).replace({np.nan: ""})

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_M1_completecases": int(dd1.shape[0]),
        "N_M2_completecases": int(dd2.shape[0]),
        "N_M3_completecases": int(dd3.shape[0]),
        "polintol_nonmissing": int(d["political_intolerance"].notna().sum()),
        "polintol_items_answered_mean": float(pol_answered.mean()) if len(pol_answered) else np.nan,
        "polintol_items_answered_min": float(pol_answered.min()) if len(pol_answered) else np.nan,
        "polintol_items_answered_max": float(pol_answered.max()) if len(pol_answered) else np.nan,
        "polintol_min_answered_rule": 10,
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
    summary_lines.append("Standardized coefficients (betas): predictors are z-scored within each model estimation sample; DV is left in original units.")
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