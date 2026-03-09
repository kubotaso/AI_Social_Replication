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

    # Common GSS missing sentinels. Keep 0 as valid unless a variable's logic forbids it.
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(s, valid_set):
        s = mask_missing(s)
        return s.where(s.isin(list(valid_set)), np.nan)

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

    def zscore(s):
        s = s.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if (not np.isfinite(sd)) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

    # Political intolerance item coding per mapping
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

    def fit_standardized(dd, y, xvars, model_name):
        """
        Standardize y and all x within the estimation sample, then run OLS.
        Coefficients on standardized variables are standardized betas.
        Robustness: if no complete cases (or only intercept), return empty but non-crashing objects.
        """
        cols = [y] + xvars
        d = dd.loc[:, cols].dropna(how="any").copy()

        # If there are no rows after listwise deletion, avoid statsmodels "zero-size array" error.
        if d.shape[0] == 0:
            coef_rows = [{"model": model_name, "term": v, "cell": "—", "beta": np.nan, "p": np.nan, "included": False} for v in xvars]
            fit = pd.DataFrame([{"model": model_name, "N": 0, "R2": np.nan, "Adj_R2": np.nan}])
            return None, pd.DataFrame(coef_rows), fit, d

        yz = zscore(d[y])

        # Standardize predictors; drop those that fail z-scoring (e.g., zero variance) within the sample.
        Xz = pd.DataFrame(index=d.index)
        keep = []
        for v in xvars:
            zv = zscore(d[v])
            if zv.notna().sum() == len(zv):
                Xz[v] = zv
                keep.append(v)

        # If no predictors remain, still fit intercept-only model to get N/R2.
        # (R2 will be 0 by definition if only intercept and yz has variance.)
        if len(keep) == 0:
            X = np.ones((len(yz), 1), dtype=float)
            res = sm.OLS(yz.values, X).fit()
            coef_rows = [{"model": model_name, "term": v, "cell": "—", "beta": np.nan, "p": np.nan, "included": False} for v in xvars]
            fit = pd.DataFrame([{"model": model_name, "N": int(len(d)), "R2": float(res.rsquared), "Adj_R2": float(res.rsquared_adj)}])
            return res, pd.DataFrame(coef_rows), fit, d

        X = sm.add_constant(Xz[keep].astype(float), has_constant="add")
        res = sm.OLS(yz.astype(float), X).fit()

        rows = []
        for v in xvars:
            if v in keep:
                b = float(res.params.get(v, np.nan))
                p = float(res.pvalues.get(v, np.nan))
                cell = f"{b:.3f}{stars(p)}" if np.isfinite(b) else "—"
                rows.append({"model": model_name, "term": v, "cell": cell, "beta": b, "p": p, "included": True})
            else:
                rows.append({"model": model_name, "term": v, "cell": "—", "beta": np.nan, "p": np.nan, "included": False})

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": int(len(d)),
                "R2": float(res.rsquared),
                "Adj_R2": float(res.rsquared_adj),
            }]
        )
        return res, pd.DataFrame(rows), fit, d

    # -------------------------
    # Load data + restrict to 1993
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

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
    core_needed = ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ballot"]
    required = core_needed + music_items + tol_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean / coerce variables
    # -------------------------
    # Music items: 1..5 only
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # SES covariates
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    # Demographics
    df["sex"] = coerce_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = coerce_valid(df["race"], {1, 2, 3})
    df["relig"] = coerce_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = coerce_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: keep only substantive codes per type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: number of music genres disliked (requires complete responses on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs construction
    # -------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP > 0 and both present
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Age already numeric cleaned
    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: available variables do not support construction (per instruction)
    # Keep as missing so it does not silently miscode; shown as "—" in tables.
    d["hispanic"] = np.nan

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy using RELIG + DENOM (coarse)
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case sum across all 15 items (0–15)
    intoler_mat = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    d["political_intolerance"] = intoler_mat.sum(axis=1, min_count=len(tol_items))  # NaN unless all 15 present

    # -------------------------
    # Model specifications
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    # -------------------------
    # Fit models (model-wise listwise deletion inside fit_standardized)
    # -------------------------
    m1, t1, f1, s1 = fit_standardized(d, y, x_m1, model_names[0])
    m2, t2, f2, s2 = fit_standardized(d, y, x_m2, model_names[1])
    m3, t3, f3, s3 = fit_standardized(d, y, x_m3, model_names[2])

    coef_long = pd.concat([t1, t2, t3], ignore_index=True)
    fitstats = pd.concat([f1, f2, f3], ignore_index=True)

    # -------------------------
    # Build Table 1-style display (standardized betas only; no SE lines)
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
    row_order = x_m3
    col_order = model_names

    wide = coef_long.pivot(index="term", columns="model", values="cell").reindex(row_order)
    wide = wide.reindex(columns=col_order).fillna("—")
    wide.index = [pretty.get(i, i) for i in wide.index]

    fit_wide = fitstats.set_index("model").reindex(col_order)
    extra = pd.DataFrame(index=["R²", "Adj. R²", "N"], columns=col_order, dtype=object)
    for m in col_order:
        r2 = fit_wide.loc[m, "R2"] if m in fit_wide.index else np.nan
        ar2 = fit_wide.loc[m, "Adj_R2"] if m in fit_wide.index else np.nan
        n = fit_wide.loc[m, "N"] if m in fit_wide.index else np.nan
        extra.loc["R²", m] = f"{float(r2):.3f}" if pd.notna(r2) else ""
        extra.loc["Adj. R²", m] = f"{float(ar2):.3f}" if pd.notna(ar2) else ""
        extra.loc["N", m] = str(int(n)) if pd.notna(n) else ""

    table1_style = pd.concat([wide, extra], axis=0)

    # -------------------------
    # Diagnostics
    # -------------------------
    diag = pd.DataFrame(
        [{
            "N_year_1993": int(df.shape[0]),
            "N_complete_music_18": int(d.shape[0]),
            "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
            "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
            "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
            "polintol_nonmissing_strict15": int(d["political_intolerance"].notna().sum()),
        }]
    )

    # -------------------------
    # Save outputs
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append("Dependent variable: Number of music genres disliked (0–18).")
    lines.append("DV construction: 18 genre ratings; disliked = 4 or 5; requires complete responses on all 18 items.")
    lines.append("")
    lines.append("Models: OLS; displayed cells are standardized coefficients (betas).")
    lines.append("Standardization: DV and predictors are z-scored within each model estimation sample; OLS run on standardized variables.")
    lines.append("Stars: two-tailed p-values from that standardized regression: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Hispanic indicator is not constructible from the provided fields; it appears as — and is excluded by listwise deletion in models that include it.")
    lines.append("- Political intolerance is a strict complete-case sum across 15 tolerance items (0–15).")
    lines.append("")
    lines.append("Table 1-style coefficients (standardized betas only):")
    lines.append(table1_style.to_string())
    lines.append("")
    lines.append("Model fit statistics:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Raw model summaries (standardized DV and predictors):")
    lines.append("\n==== Model 1 (SES) ====\n" + (m1.summary().as_text() if m1 is not None else "Model 1 could not be estimated (no complete cases)."))
    lines.append("\n==== Model 2 (Demographic) ====\n" + (m2.summary().as_text() if m2 is not None else "Model 2 could not be estimated (no complete cases)."))
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + (m3.summary().as_text() if m3 is not None else "Model 3 could not be estimated (no complete cases)."))

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("DV: Number of music genres disliked\n")
        f.write("Cells: standardized betas with stars from standardized regression p-values.\n")
        f.write("— indicates not in model or not constructible.\n\n")
        f.write(table1_style.to_string())
        f.write("\n")

    table1_style.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)

    return {
        "table1_style": table1_style,
        "fit_stats": fitstats,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "estimation_samples": {"m1": s1, "m2": s2, "m3": s3},
    }