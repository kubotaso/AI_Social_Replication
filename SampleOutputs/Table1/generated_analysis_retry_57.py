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

    # Broader GSS-style missing sentinels (do NOT treat 0 as missing)
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid_set):
        s = mask_missing(series)
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

    def zscore_series(s):
        s = s.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    # Intolerance coding per mapping
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

    def build_polintol_complete(df, tol_items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        # Strict complete-case on all 15 items, as described
        ok = intoler.notna().all(axis=1)
        scale = pd.Series(np.nan, index=df.index, dtype=float)
        scale.loc[ok] = intoler.loc[ok].sum(axis=1).astype(float)
        return scale, intoler, ok

    def fit_model_with_std_betas(df, y, xvars):
        """
        Fit:
          - raw OLS on original units with intercept (for constant, R2, AdjR2, p-values)
          - standardized betas computed as:
              beta_j = b_j * sd(x_j) / sd(y)
            where sds computed on the model estimation sample (listwise-complete for y and all xvars).
        Return:
          - coef table for predictors (standardized betas + stars from raw p-values)
          - fit dict (N, R2, AdjR2, raw constant)
          - estimation sample (dataframe)
          - raw results object
        """
        cols = [y] + xvars
        use = df[cols].dropna().copy()
        if use.shape[0] == 0:
            coef = pd.DataFrame({"term": xvars, "beta_std": [np.nan]*len(xvars), "p_raw": [np.nan]*len(xvars), "sig": [""]*len(xvars)})
            fit = {"N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan}
            return coef, fit, use, None

        # raw OLS with intercept
        X = sm.add_constant(use[xvars].astype(float), has_constant="add")
        yy = use[y].astype(float)
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
        # predictor cells: beta with stars; "—" if not in that model
        table = pd.DataFrame(index=row_order, columns=model_names, dtype=object)
        for m in model_names:
            c = coefs_by_model[m].set_index("term")
            for term in row_order:
                if term not in c.index:
                    table.loc[term, m] = "—"
                else:
                    b = c.loc[term, "beta_std"]
                    sig = c.loc[term, "sig"]
                    if pd.isna(b):
                        table.loc[term, m] = "—"
                    else:
                        table.loc[term, m] = f"{b:.3f}{sig}"

        # relabel rows to match paper labels
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

    required = ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ballot"] + music_items + tol_items
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music: keep 1..5 only (DK/missing => NaN)
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

    # ETHNIC exists in this file, but per mapping instruction, "Hispanic cannot be constructed"
    # We therefore DO NOT use ETHNIC to create Hispanic, and we do not include it in models.

    # Tolerance items: mask missing; keep only valid codes
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness (count genres disliked); strict complete-case on all 18 items
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita: REALINC / HOMPOP; require both present and HOMPOP > 0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Age already cleaned
    # Race indicators (White reference; do not make them mutually exclusive with "Hispanic" since Hispanic is not available here)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic unavailable -> include a placeholder all-NaN column so it prints "—" in all models
    d["hispanic"] = np.nan

    # Conservative Protestant proxy (coarse, based on RELIG + DENOM as described)
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

    # Political intolerance scale (0-15), strict complete-case on all 15 items
    d["political_intolerance"], intoler_df, pol_complete = build_polintol_complete(d, tol_items)

    # -------------------------
    # Models (Table 1)
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    coefs_by_model = {}
    fits_by_model = {}
    samples_by_model = {}
    raw_results = {}

    for name, xvars in zip(model_names, [x_m1, x_m2, x_m3]):
        coef, fit, use, res_raw = fit_model_with_std_betas(d, y, xvars)
        coefs_by_model[name] = coef
        fits_by_model[name] = fit
        samples_by_model[name] = use
        raw_results[name] = res_raw

    # -------------------------
    # Output table with correct labeling and dashes (no SE rows)
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
        "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
        "polintol_complete_case_rule": "15/15 items required",
        "hispanic_available": False,
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
    lines.append("  - OLS on raw variables with intercept; R²/Adj. R²/Constant are from this raw model.")
    lines.append("  - Standardized betas are computed on each model estimation sample as: beta = b * SD(x) / SD(y).")
    lines.append("  - Stars are from two-tailed p-values of the raw OLS coefficients: * p<.05, ** p<.01, *** p<.001.")
    lines.append("")
    lines.append("Notes on construct availability:")
    lines.append("  - Hispanic indicator is not constructible from the provided mapping; it is left as missing and shown as '—'.")
    lines.append("  - Political intolerance scale requires complete responses on all 15 tolerance items.")
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
    lines.append("Raw model summaries (for debugging; not part of Table 1):")
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
        f.write("— indicates predictor not included or not constructible/estimable.\n\n")
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
        "estimation_samples": {m: samples_by_model[m] for m in model_names},
    }