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

    # Conservative GSS-style missing sentinels frequently used across variables.
    # (Do NOT treat 0 as missing.)
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

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

    def zscore(series):
        s = series.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)  # population SD, consistent with many beta conversions
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

    # Political intolerance coding per mapping instruction
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

    def build_polintol_partial(df, tol_items, min_answered=1):
        """
        Paper describes a count scale across 15 dichotomous items.
        To avoid excessive N loss (common with GSS split ballots / DK),
        compute the *count of intolerant responses among answered items*,
        requiring at least `min_answered` answered items.

        NOTE: Set min_answered=15 for strict complete-case scale if desired.
        """
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        answered = intoler.notna().sum(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        m = answered >= int(min_answered)
        pol.loc[m] = intoler.loc[m].fillna(0.0).sum(axis=1).astype(float)
        return pol, answered, intoler

    def fit_standardized_ols(dd, y, xvars):
        """
        Standardized betas via: z-score y and each x within the estimation sample,
        then OLS. Coefficients on z-scored X are standardized betas.
        """
        # drop no-variance predictors within this estimation sample
        x_keep = []
        dropped = []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        # Standardize within sample
        y_z = zscore(dd[y])
        Xz = pd.DataFrame({v: zscore(dd[v]) for v in x_keep})
        X = sm.add_constant(Xz, has_constant="add")
        res_std = sm.OLS(y_z, X).fit()

        # Raw model for constant and R2 using same sample (unstandardized)
        X_raw = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        res_raw = sm.OLS(dd[y].astype(float), X_raw).fit()

        return res_std, res_raw, x_keep, dropped

    def build_table_cells(model_name, xvars, x_keep, res_std):
        rows = []
        for v in xvars:
            if v in x_keep:
                beta = float(res_std.params.get(v, np.nan))
                p = float(res_std.pvalues.get(v, np.nan))
                cell = "—" if not np.isfinite(beta) else f"{beta:.3f}{stars(p)}"
                rows.append({"model": model_name, "term": v, "cell": cell, "beta": beta, "p": p})
            else:
                rows.append({"model": model_name, "term": v, "cell": "—", "beta": np.nan, "p": np.nan})
        return pd.DataFrame(rows)

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
    # Music ratings: 1..5 valid; DK/NA/refused -> NaN
    for c in music_items:
        df[c] = keep_codes(df[c], {1, 2, 3, 4, 5})

    # Core covariates
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

    # Tolerance items validity
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: musical exclusiveness = count disliked across 18; require complete on all 18
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita = REALINC / HOMPOP; require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic indicator:
    # IMPORTANT FIX: do NOT assume missing ETHNIC => non-Hispanic.
    # Keep missing as missing so listwise deletion behaves correctly.
    # (Coding: ETHNIC==1 is treated as Hispanic in this extract.)
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant (proxy from coarse DENOM): Protestant AND denom in {1,6,7}
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance:
    # IMPORTANT FIX: allow partial completion to better match expected N (paper's N is larger than strict-15 in many extracts).
    # Use a conservative threshold: at least 10 of 15 answered.
    d["political_intolerance"], pol_answered, intoler_df = build_polintol_partial(d, tol_items, min_answered=10)

    # -------------------------
    # Models: listwise deletion per model
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    # Estimation samples
    dd1 = d[[y] + x_m1].dropna().copy()
    dd2 = d[[y] + x_m2].dropna().copy()
    dd3 = d[[y] + x_m3].dropna().copy()

    res1_std, res1_raw, x1_keep, x1_drop = fit_standardized_ols(dd1, y, x_m1) if len(dd1) else (None, None, [], x_m1)
    res2_std, res2_raw, x2_keep, x2_drop = fit_standardized_ols(dd2, y, x_m2) if len(dd2) else (None, None, [], x_m2)
    res3_std, res3_raw, x3_keep, x3_drop = fit_standardized_ols(dd3, y, x_m3) if len(dd3) else (None, None, [], x_m3)

    # Coefficient cells (betas only; no SE rows)
    c1 = build_table_cells(model_names[0], x_m1, x1_keep, res1_std) if res1_std is not None else build_table_cells(model_names[0], x_m1, [], None)
    c2 = build_table_cells(model_names[1], x_m2, x2_keep, res2_std) if res2_std is not None else build_table_cells(model_names[1], x_m2, [], None)
    c3 = build_table_cells(model_names[2], x_m3, x3_keep, res3_std) if res3_std is not None else build_table_cells(model_names[2], x_m3, [], None)

    coef_long = pd.concat([c1, c2, c3], ignore_index=True)

    def fit_row(name, dd, res_raw, dropped):
        return {
            "model": name,
            "N": int(dd.shape[0]),
            "R2": (float(res_raw.rsquared) if res_raw is not None else np.nan),
            "Adj_R2": (float(res_raw.rsquared_adj) if res_raw is not None else np.nan),
            "const_raw": (float(res_raw.params.get("const", np.nan)) if res_raw is not None else np.nan),
            "dropped_no_variance": ", ".join(dropped) if dropped else "",
        }

    fitstats = pd.DataFrame([
        fit_row(model_names[0], dd1, res1_raw, x1_drop),
        fit_row(model_names[1], dd2, res2_raw, x2_drop),
        fit_row(model_names[2], dd3, res3_raw, x3_drop),
    ])

    # -------------------------
    # Build Table 1-style output (betas only)
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

    dv_label = "Number of music genres disliked (0–18)"
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
        "hispanic_missing_count": int(d["hispanic"].isna().sum()),
        "hispanic_1_count": int((d["hispanic"] == 1).sum()),
        "polintol_nonmissing_min10": int(d["political_intolerance"].notna().sum()),
        "polintol_items_answered_mean": float(pol_answered.mean()) if len(pol_answered) else np.nan,
        "polintol_items_answered_min": float(pol_answered.min()) if len(pol_answered) else np.nan,
        "polintol_items_answered_max": float(pol_answered.max()) if len(pol_answered) else np.nan,
        "polintol_min_answered_rule": 10,
    }])

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    lines = []
    lines.append("Replication: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV (all models): {dv_label}")
    lines.append("DV construction: 18 music items; dislike=4/5; requires complete responses on all 18 items.")
    lines.append("")
    lines.append("Displayed coefficients: standardized betas from OLS on z-scored variables (within each model estimation sample).")
    lines.append("Stars: from OLS p-values (* p<.05, ** p<.01, *** p<.001).")
    lines.append("Note: Table prints betas only (no standard errors).")
    lines.append("")
    lines.append("Table 1-style standardized coefficients (betas only):")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Model fit stats (raw-scale regressions on same estimation samples):")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 1 variables:")
    lines.append(miss_m1.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 2 variables:")
    lines.append(miss_m2.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 3 variables:")
    lines.append(miss_m3.to_string(index=False))
    lines.append("")
    lines.append("Raw OLS summaries (debug):")
    if res1_raw is not None:
        lines.append("\n==== Model 1 (SES), raw DV ====\n" + res1_raw.summary().as_text())
    if res2_raw is not None:
        lines.append("\n==== Model 2 (Demographic), raw DV ====\n" + res2_raw.summary().as_text())
    if res3_raw is not None:
        lines.append("\n==== Model 3 (Political intolerance), raw DV ====\n" + res3_raw.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from OLS p-values.\n")
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