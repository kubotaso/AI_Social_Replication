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

    # Common GSS-style missing sentinels; keep 0 as valid.
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid):
        s = mask_missing(series)
        return s.where(s.isin(valid), np.nan)

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

    def wmean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

    def wvar_pop(x, w):
        # population weighted variance
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        mu = np.sum(w[m] * x[m]) / np.sum(w[m])
        return float(np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m]))

    def wsd_pop(x, w):
        v = wvar_pop(x, w)
        return np.nan if not np.isfinite(v) else float(np.sqrt(v))

    def fit_wls_and_std_betas(df, y, xvars, wcol=None):
        cols = [y] + xvars + ([wcol] if wcol else [])
        use = df[cols].dropna().copy()
        if use.shape[0] == 0:
            coef = pd.DataFrame({"term": xvars, "beta_std": np.nan, "p_raw": np.nan, "sig": ""})
            fit = {"N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan}
            return coef, fit, use, None

        X = sm.add_constant(use[xvars].astype(float), has_constant="add")
        yy = use[y].astype(float)

        if wcol:
            ww = use[wcol].astype(float).clip(lower=0)
            res = sm.WLS(yy, X, weights=ww).fit()
            sd_y = wsd_pop(yy.values, ww.values)
            rows = []
            for v in xvars:
                sd_x = wsd_pop(use[v].astype(float).values, ww.values)
                b = float(res.params.get(v, np.nan))
                p = float(res.pvalues.get(v, np.nan))
                beta = np.nan
                if np.isfinite(b) and np.isfinite(sd_x) and np.isfinite(sd_y) and sd_x != 0 and sd_y != 0:
                    beta = b * (sd_x / sd_y)
                rows.append({"term": v, "beta_std": beta, "p_raw": p, "sig": stars(p)})
            coef = pd.DataFrame(rows)
        else:
            res = sm.OLS(yy, X).fit()
            sd_y = float(yy.std(ddof=0))
            rows = []
            for v in xvars:
                sd_x = float(use[v].astype(float).std(ddof=0))
                b = float(res.params.get(v, np.nan))
                p = float(res.pvalues.get(v, np.nan))
                beta = np.nan
                if np.isfinite(b) and np.isfinite(sd_x) and np.isfinite(sd_y) and sd_x != 0 and sd_y != 0:
                    beta = b * (sd_x / sd_y)
                rows.append({"term": v, "beta_std": beta, "p_raw": p, "sig": stars(p)})
            coef = pd.DataFrame(rows)

        fit = {
            "N": int(use.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
        }
        return coef, fit, use, res

    def build_table1(coefs_by_model, fits_by_model, model_names, row_order, label_map, dv_label):
        table = pd.DataFrame(index=row_order, columns=model_names, dtype=object)

        for m in model_names:
            c = coefs_by_model[m].set_index("term")
            for term in row_order:
                if term not in c.index:
                    table.loc[term, m] = "—"
                else:
                    b = c.loc[term, "beta_std"]
                    sig = c.loc[term, "sig"]
                    table.loc[term, m] = "—" if pd.isna(b) else f"{b:.3f}{sig}"

        # relabel
        table.index = [label_map.get(t, t) for t in table.index]

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            fit = fits_by_model[m]
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit["const_raw"]) else f"{fit['const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit["R2"]) else f"{fit['R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit["Adj_R2"]) else f"{fit['Adj_R2']:.3f}"
            extra.loc["N", m] = "" if fit["N"] is None else str(int(fit["N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        out = pd.concat([header, table, extra], axis=0)
        return out

    def missingness_table(df, vars_):
        rows = []
        for v in vars_:
            rows.append({"var": v, "nonmissing": int(df[v].notna().sum()), "missing": int(df[v].isna().sum())})
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

    required = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "relig", "denom", "region", "ballot",
        "ethnic"
    ] + music_items + tol_items
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music ratings: keep 1..5 only
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # SES / demos
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    df["sex"] = coerce_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = coerce_valid(df["race"], {1, 2, 3})
    df["relig"] = coerce_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = coerce_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])
    df["ethnic"] = mask_missing(df["ethnic"])

    # Tolerance items: enforce valid substantive codes
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: number of genres disliked, strict complete-case on all 18 items
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    d["num_genres_disliked"] = d[[f"dislike_{c}" for c in music_items]].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita: REALINC / HOMPOP; require both nonmissing and HOMPOP > 0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: fix N collapse by coding non-Hispanic as 0 (not NaN) when ETHNIC present.
    # If ETHNIC missing, set NA (unknown).
    # (This matches the discrepancy feedback: previously too many NA because 0 was treated as missing.)
    d["hispanic"] = np.nan
    m_eth = d["ethnic"].notna()
    d.loc[m_eth, "hispanic"] = (d.loc[m_eth, "ethnic"] == 1).astype(float)

    # Conservative Protestant proxy (coarse): Protestant (RELIG==1) and DENOM in {1,6,7}
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

    # Political intolerance: FIX to match paper rule more closely:
    # - sum across 15 dichotomies
    # - require complete responses on all 15 items (strict)
    # (This directly addresses the prior "min answered=10" mismatch feedback.)
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

    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    pol_ok = intoler.notna().all(axis=1)
    d["political_intolerance"] = np.nan
    d.loc[pol_ok, "political_intolerance"] = intoler.loc[pol_ok].sum(axis=1).astype(float)

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

    coefs_by_model, fits_by_model, samples_by_model, raw_results = {}, {}, {}, {}
    for name, xvars in zip(model_names, [x_m1, x_m2, x_m3]):
        coef, fit, use, res = fit_wls_and_std_betas(d, y, xvars, wcol=None)
        coefs_by_model[name] = coef
        fits_by_model[name] = fit
        samples_by_model[name] = use
        raw_results[name] = res

    # -------------------------
    # Table output (standardized betas only; labeled; em-dash for excluded)
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

    table1 = build_table1(
        coefs_by_model=coefs_by_model,
        fits_by_model=fits_by_model,
        model_names=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label,
    )

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1_listwise": int(samples_by_model[model_names[0]].shape[0]),
        "N_model2_listwise": int(samples_by_model[model_names[1]].shape[0]),
        "N_model3_listwise": int(samples_by_model[model_names[2]].shape[0]),
        "hispanic_nonmissing_in_dv_complete": int(d["hispanic"].notna().sum()),
        "hispanic_1_count_in_dv_complete": int((d["hispanic"] == 1).sum(skipna=True)),
        "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
        "polintol_complete_case_rule": "15/15 items required",
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

    coef_long = []
    for m in model_names:
        tmp = coefs_by_model[m].copy()
        tmp["model"] = m
        coef_long.append(tmp)
    coef_long = pd.concat(coef_long, ignore_index=True)

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"Dependent variable (DV): {dv_label}")
    lines.append("DV construction: 18 music items; disliked = 4 or 5; strict complete-case across all 18 items.")
    lines.append("")
    lines.append("Estimation:")
    lines.append("  - Raw OLS with intercept; constants and R²/Adj. R² come from raw model.")
    lines.append("  - Standardized betas computed on each model estimation sample as: beta = b * SD(x) / SD(y).")
    lines.append("  - Stars from two-tailed p-values of raw OLS coefficients: * p<.05, ** p<.01, *** p<.001.")
    lines.append("")
    lines.append("Construct notes:")
    lines.append("  - Hispanic is constructed from ETHNIC: ETHNIC==1 -> Hispanic; ETHNIC!=1 -> not Hispanic (0); missing ETHNIC -> NA.")
    lines.append("  - Political intolerance is the strict sum across 15 tolerance items (15/15 complete-case).")
    lines.append("")
    lines.append("Table 1-style standardized coefficients (no standard errors printed):")
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
    lines.append("Raw model summaries (debugging):")
    for m in model_names:
        lines.append(f"\n==== {m} ====")
        res = raw_results[m]
        lines.append("(No estimable model.)" if res is None else res.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("— indicates predictor not included or not estimable.\n\n")
        f.write(table1.to_string())
        f.write("\n")

    # Machine-readable extracts
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
        "estimation_samples": {m: samples_by_model[m] for m in model_names},
    }