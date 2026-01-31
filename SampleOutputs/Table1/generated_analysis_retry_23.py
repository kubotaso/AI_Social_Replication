def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    # Conservative NA sentinels; do not treat 0 as missing.
    MISSING = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING), np.nan)

    def keep_codes(s, codes):
        s = mask_missing(s)
        return s.where(s.isin(list(codes)), np.nan)

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

    def pop_sd(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

    # Political intolerance item -> intolerant(1/0), NaN if missing
    def intolerance_indicator(col, s):
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            m = s.isin([4, 5])
            if col == "colcom":
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired
            else:
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    def fit_raw_and_betas(d, y, xvars, model_name):
        """
        Fit raw OLS on complete cases; compute standardized betas as:
            beta_j = b_j * SD(x_j) / SD(y)
        SDs computed on the same estimation sample.
        Stars come from raw OLS p-values (paper reports betas + stars, not SEs).
        """
        cols = [y] + xvars
        dd = d.loc[:, cols].dropna(how="any").copy()
        n = int(dd.shape[0])

        # Drop no-variance predictors in this estimation sample (avoid singular fits)
        x_keep = []
        dropped = []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = pop_sd(yy.values)
        betas = {}
        for v in x_keep:
            sd_x = pop_sd(dd[v].values)
            b = float(res.params.get(v, np.nan))
            if np.isfinite(sd_x) and np.isfinite(sd_y) and sd_x > 0 and sd_y > 0:
                betas[v] = b * (sd_x / sd_y)
            else:
                betas[v] = np.nan

        rows = []
        for v in xvars:
            if v in x_keep:
                beta = betas.get(v, np.nan)
                p = float(res.pvalues.get(v, np.nan))
                cell = "—" if not np.isfinite(beta) else f"{beta:.3f}{stars_from_p(p)}"
                rows.append({"model": model_name, "term": v, "beta_std": beta, "p_raw": p, "cell": cell, "included": True})
            else:
                rows.append({"model": model_name, "term": v, "beta_std": np.nan, "p_raw": np.nan, "cell": "—", "included": False})

        coef_long = pd.DataFrame(rows)
        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": n,
                "R2": float(res.rsquared) if n > 0 else np.nan,
                "Adj_R2": float(res.rsquared_adj) if n > 0 else np.nan,
                "const_raw": float(res.params.get("const", np.nan)) if n > 0 else np.nan,
                "dropped_no_variance": ", ".join(dropped) if dropped else "",
            }]
        )
        return res, coef_long, fit, dd

    def build_table(coef_long, fitstats, row_order, model_order, pretty_map):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_order).fillna("—")
        wide.index = [pretty_map.get(t, t) for t in wide.index]

        fit = fitstats.set_index("model").reindex(model_order)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))
        return pd.concat([wide, extra], axis=0)

    def missingness_table(df, vars_):
        rows = []
        for v in vars_:
            rows.append({"var": v, "missing": int(df[v].isna().sum()), "nonmissing": int(df[v].notna().sum())})
        return pd.DataFrame(rows).sort_values(["missing", "var"], ascending=[False, True])

    # -------------------------
    # Load + year filter
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower().strip() for c in df.columns]
    if "year" not in df.columns:
        raise ValueError("Required column missing: year")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Variable lists
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
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ballot"]
        + music_items + tol_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean music items (1..5)
    # -------------------------
    for c in music_items:
        df[c] = keep_codes(df[c], {1, 2, 3, 4, 5})

    # -------------------------
    # Clean core predictors
    # -------------------------
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    df["sex"] = keep_codes(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_codes(df["race"], {1, 2, 3})
    df["relig"] = keep_codes(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = keep_codes(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # NOTE: Hispanic is not reliably constructible from provided variable list.
    # To avoid collapsing N (and avoid arbitrary miscoding), set Hispanic=0 for all.
    df["hispanic"] = 0.0

    # -------------------------
    # Clean tolerance items: keep only valid substantive codes
    # -------------------------
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: number of genres disliked (listwise complete across 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs construction
    # -------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female indicator
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (coarse, based on provided DENOM recode)
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: STRICT complete-case across all 15 items, then sum (0..15)
    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    d["political_intolerance"] = intoler.sum(axis=1, min_count=len(tol_items))

    # -------------------------
    # Model specifications (Table 1)
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = [
        "Model 1 (SES)",
        "Model 2 (Demographic)",
        "Model 3 (Adds political intolerance)"
    ]

    # -------------------------
    # Fit models
    # -------------------------
    m1_res, c1, f1, s1 = fit_raw_and_betas(d, y, x_m1, model_names[0])
    m2_res, c2, f2, s2 = fit_raw_and_betas(d, y, x_m2, model_names[1])
    m3_res, c3, f3, s3 = fit_raw_and_betas(d, y, x_m3, model_names[2])

    coef_long = pd.concat([c1, c2, c3], ignore_index=True)
    fitstats = pd.concat([f1, f2, f3], ignore_index=True)

    # -------------------------
    # Build Table 1-style (betas only; no SEs)
    # -------------------------
    pretty = {
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
    table1 = build_table(coef_long, fitstats, row_order, model_names, pretty)

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
        "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
        "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
        "political_intolerance_nonmissing_strict15": int(d["political_intolerance"].notna().sum()),
        "hispanic_all_zero": bool((d["hispanic"] == 0).all()),
        "note": "Hispanic not constructible from provided fields; set to 0 to avoid N collapse."
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
    summary_lines.append("Dependent variable (all models): Number of music genres disliked (0–18).")
    summary_lines.append("DV construction: 18 genre ratings; disliked = 4 or 5; requires complete responses on all 18 items.")
    summary_lines.append("")
    summary_lines.append("Models: OLS on raw DV; reported coefficients are standardized betas computed as beta = b * SD(x)/SD(y) on each model's estimation sample.")
    summary_lines.append("Stars: two-tailed p-values from raw OLS coefficients: * p<.05, ** p<.01, *** p<.001.")
    summary_lines.append("")
    summary_lines.append("Important note on Hispanic: Not reliably constructible from provided variable list; set to 0.0 for all cases to prevent listwise deletion collapse.")
    summary_lines.append("")
    summary_lines.append("Table 1-style coefficients (standardized betas only) + fit statistics:")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Fit stats:")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Diagnostics:")
    summary_lines.append(diag.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness within DV-complete sample (Model 1 vars):")
    summary_lines.append(miss_m1.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness within DV-complete sample (Model 2 vars):")
    summary_lines.append(miss_m2.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness within DV-complete sample (Model 3 vars):")
    summary_lines.append(miss_m3.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Raw OLS summaries (not in published table; included for debugging):")
    summary_lines.append("\n==== Model 1 (SES) ====\n" + m1_res.summary().as_text())
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + m2_res.summary().as_text())
    summary_lines.append("\n==== Model 3 (Adds political intolerance) ====\n" + m3_res.summary().as_text())

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("DV: Number of music genres disliked\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n\n")
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
        "estimation_samples": {"m1": s1, "m2": s2, "m3": s3},
    }