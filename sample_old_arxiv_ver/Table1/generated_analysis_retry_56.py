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

    # Common GSS-style missing sentinels (avoid treating 0 as missing)
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(s, valid):
        s = mask_missing(s)
        return s.where(s.isin(list(valid)), np.nan)

    def zscore(x):
        x = np.asarray(x, dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return np.full_like(x, np.nan, dtype=float)
        return (x - mu) / sd

    def stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    # Intolerance item coding per mapping
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

    def build_polintol(df, tol_items, min_answered=15):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        answered = intoler.notna().sum(axis=1)
        scale = pd.Series(np.nan, index=df.index, dtype=float)
        m = answered >= int(min_answered)
        # Sum across answered items only (missing contribute 0) AFTER minimum threshold
        scale.loc[m] = intoler.loc[m].fillna(0.0).sum(axis=1).astype(float)
        return scale, intoler, answered

    def fit_standardized_ols(dd, y, xvars):
        """
        Standardized coefficients via z-scoring Y and all X on the estimation sample
        then OLS with no intercept. Coefficients are standardized betas.
        Stars come from p-values of standardized regression.
        """
        use = dd[[y] + xvars].dropna().copy()
        if use.shape[0] == 0:
            return None, use, pd.DataFrame({"term": xvars, "beta": [np.nan] * len(xvars), "p": [np.nan] * len(xvars)}), {
                "N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan
            }

        # z-score
        yz = zscore(use[y].to_numpy(dtype=float))
        Xz = {}
        for v in xvars:
            Xz[v] = zscore(use[v].to_numpy(dtype=float))
        Z = pd.DataFrame(Xz, index=use.index)

        # Drop any predictors that became all-NaN due to zero variance after z-scoring
        keep = [v for v in xvars if np.isfinite(Z[v]).sum() == Z.shape[0] and np.nanstd(use[v].to_numpy(dtype=float), ddof=0) > 0]
        Z = Z[keep]

        # If nothing left, return empty
        if Z.shape[1] == 0:
            coef = pd.DataFrame({"term": xvars, "beta": [np.nan] * len(xvars), "p": [np.nan] * len(xvars)})
            fit = {"N": int(use.shape[0]), "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan}
            return None, use, coef, fit

        res_std = sm.OLS(yz, Z).fit()

        # Also fit raw model for fit rows (with intercept)
        X_raw = sm.add_constant(use[keep].astype(float), has_constant="add")
        res_raw = sm.OLS(use[y].astype(float), X_raw).fit()

        # Map back to full xvars order and format
        rows = []
        for v in xvars:
            if v in keep:
                b = float(res_std.params.get(v, np.nan))
                p = float(res_std.pvalues.get(v, np.nan))
                rows.append({"term": v, "beta": b, "p": p})
            else:
                rows.append({"term": v, "beta": np.nan, "p": np.nan})
        coef = pd.DataFrame(rows)

        fit = {
            "N": int(use.shape[0]),
            "R2": float(res_raw.rsquared),
            "Adj_R2": float(res_raw.rsquared_adj),
            "const_raw": float(res_raw.params.get("const", np.nan)),
        }
        return {"std": res_std, "raw": res_raw}, use, coef, fit

    def build_table(coefs_by_model, fits_by_model, model_names, row_order, label_map, dv_label):
        # cells: beta with stars, or em dash if not in model
        table = pd.DataFrame(index=row_order, columns=model_names, dtype=object)
        for m in model_names:
            cdf = coefs_by_model[m].set_index("term")
            for term in row_order:
                if term not in cdf.index:
                    table.loc[term, m] = "—"
                else:
                    b = cdf.loc[term, "beta"]
                    p = cdf.loc[term, "p"]
                    if not np.isfinite(b):
                        table.loc[term, m] = "—"
                    else:
                        table.loc[term, m] = f"{b:.3f}{stars(p)}"

        table.index = [label_map.get(t, t) for t in table.index]

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            fit = fits_by_model[m]
            extra.loc["Constant (raw)", m] = "" if not np.isfinite(fit["const_raw"]) else f"{fit['const_raw']:.3f}"
            extra.loc["R²", m] = "" if not np.isfinite(fit["R2"]) else f"{fit['R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if not np.isfinite(fit["Adj_R2"]) else f"{fit['Adj_R2']:.3f}"
            extra.loc["N", m] = str(int(fit["N"])) if np.isfinite(fit["N"]) else ""

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        return pd.concat([header, table, extra], axis=0)

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

    required = ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region"] + music_items + tol_items
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music 1..5 only
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # SES / demos
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

    # ethnicity present in file; use as "Hispanic" indicator if ethnic==1, else 0, and missing->0
    if "ethnic" in df.columns:
        df["ethnic"] = mask_missing(df["ethnic"])

    # Tolerance items: validate and mask missing
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: strict complete-case on all 18 music items
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: from ETHNIC==1 if available; missing->0 to avoid collapsing sample
    if "ethnic" in d.columns:
        hisp = (d["ethnic"] == 1).astype(float)
        hisp = hisp.where(d["ethnic"].notna(), 0.0)
        d["hispanic"] = hisp
    else:
        d["hispanic"] = 0.0

    # Make race dummies mutually exclusive with Hispanic (paper-style group identities often do this)
    m_h = d["hispanic"].eq(1.0)
    d.loc[m_h & d["black"].notna(), "black"] = 0.0
    d.loc[m_h & d["other_race"].notna(), "other_race"] = 0.0

    # Conservative Protestant proxy (coarse, based on RELIG + DENOM)
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

    # Political intolerance scale:
    # Use partial completion to avoid N collapse; require at least 10/15 answered to mirror typical scale practice.
    # (Strict 15/15 can be too aggressive in this extract.)
    d["political_intolerance"], intoler_df, pol_answered = build_polintol(d, tol_items, min_answered=10)

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

    # Fit (standardized betas via standardized regression on each model sample)
    results = {}
    coefs_by_model = {}
    fits_by_model = {}
    samples = {}

    for name, xvars in zip(model_names, [x_m1, x_m2, x_m3]):
        res, use, coef, fit = fit_standardized_ols(d, y, xvars)
        results[name] = res
        coefs_by_model[name] = coef
        fits_by_model[name] = fit
        samples[name] = use

    # -------------------------
    # Build Table 1-style output
    # -------------------------
    dv_label = "Number of music genres disliked"
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

    table1 = build_table(
        coefs_by_model=coefs_by_model,
        fits_by_model=fits_by_model,
        model_names=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label
    )

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1_listwise": int(samples[model_names[0]].shape[0]),
        "N_model2_listwise": int(samples[model_names[1]].shape[0]),
        "N_model3_listwise": int(samples[model_names[2]].shape[0]),
        "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
        "polintol_answered_mean": float(pol_answered.mean()) if len(pol_answered) else np.nan,
        "polintol_answered_min": float(pol_answered.min()) if len(pol_answered) else np.nan,
        "polintol_answered_max": float(pol_answered.max()) if len(pol_answered) else np.nan,
        "polintol_min_answered_rule": 10,
        "hispanic_1_count_in_dv_complete": int((d["hispanic"] == 1).sum()),
    }])

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    fitstats = pd.DataFrame([{
        "model": m,
        "N": fits_by_model[m]["N"],
        "R2": fits_by_model[m]["R2"],
        "Adj_R2": fits_by_model[m]["Adj_R2"],
        "const_raw": fits_by_model[m]["const_raw"],
    } for m in model_names])

    # Long coefficients (machine readable)
    coef_long = []
    for m in model_names:
        tmp = coefs_by_model[m].copy()
        tmp["model"] = m
        coef_long.append(tmp)
    coef_long = pd.concat(coef_long, ignore_index=True)

    # -------------------------
    # Save outputs (human-readable text)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"Dependent variable (DV): {dv_label}")
    lines.append("DV construction: 18 music items; disliked = 4 or 5; strict complete-case on all 18 items.")
    lines.append("")
    lines.append("Estimation:")
    lines.append("  - Standardized coefficients are obtained by z-scoring Y and all included X variables on the model estimation sample,")
    lines.append("    then fitting OLS with no intercept. Reported betas are the standardized coefficients.")
    lines.append("  - Fit rows (R², Adj. R², Constant) are from the corresponding raw-variable OLS with an intercept on the same estimation sample.")
    lines.append("  - Stars use two-tailed p-values from the standardized regression: * p<.05, ** p<.01, *** p<.001.")
    lines.append("")
    lines.append("Table 1-style standardized coefficients:")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample):")
    lines.append("\nModel 1 vars:\n" + miss_m1.to_string(index=False))
    lines.append("\nModel 2 vars:\n" + miss_m2.to_string(index=False))
    lines.append("\nModel 3 vars:\n" + miss_m3.to_string(index=False))
    lines.append("")
    lines.append("Raw model summaries (debug):")
    for m in model_names:
        lines.append(f"\n==== {m} ====")
        if results[m] is None:
            lines.append("(No estimable model.)")
        else:
            lines.append(results[m]["raw"].summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from standardized-regression p-values.\n")
        f.write("— indicates predictor not included or not estimable.\n\n")
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
        "estimation_samples": {m: samples[m] for m in model_names},
    }