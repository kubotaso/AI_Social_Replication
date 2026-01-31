def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    # Broad set of common GSS "special missing" codes (varies by item)
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        7, 8, 9,
        77, 78, 79, 88, 89, 98, 99,
        997, 998, 999,
        9997, 9998, 9999,
        99997, 99998, 99999,
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(s, valid):
        s = mask_missing(s)
        return s.where(s.isin(set(valid)), np.nan)

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

    def pop_sd(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

    # -----------------------------
    # Political intolerance coding per mapping
    # -----------------------------
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

    def build_polintol_complete(df, items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        complete = intoler.notna().all(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        pol.loc[complete] = intoler.loc[complete].sum(axis=1).astype(float)
        return pol, intoler, complete

    # -----------------------------
    # OLS + standardized betas for display (paper style):
    # - Fit UNSTANDARDIZED OLS (for constant, R2, adj R2, p-values)
    # - Compute standardized beta_j = b_j * SD(x_j)/SD(y) using the same estimation sample
    # -----------------------------
    def fit_ols_with_betas(df, y, xvars, model_name):
        cols = [y] + xvars
        use = df.loc[:, cols].dropna(how="any").copy()
        if use.shape[0] == 0:
            raise ValueError(f"{model_name}: empty estimation sample after listwise deletion.")

        yv = use[y].astype(float).values
        X = sm.add_constant(use[xvars].astype(float), has_constant="add")
        res = sm.OLS(yv, X).fit()

        sd_y = pop_sd(yv)
        betas = {}
        for v in xvars:
            sd_x = pop_sd(use[v].astype(float).values)
            b = res.params.get(v, np.nan)
            if not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0:
                betas[v] = np.nan
            else:
                betas[v] = float(b) * (sd_x / sd_y)

        coef_rows = []
        for v in xvars:
            p = float(res.pvalues.get(v, np.nan))
            coef_rows.append(
                {
                    "model": model_name,
                    "term": v,
                    "beta_std": float(betas.get(v, np.nan)) if np.isfinite(betas.get(v, np.nan)) else np.nan,
                    "b_raw": float(res.params.get(v, np.nan)),
                    "p_raw": p,
                    "sig": stars(p),
                }
            )

        fit = {
            "model": model_name,
            "N": int(use.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
        }

        return res, pd.DataFrame(coef_rows), pd.DataFrame([fit]), use

    def build_table_table1_style(coef_list, fit_list, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coef_list, ignore_index=True)
        fit_long = pd.concat(fit_list, ignore_index=True).set_index("model").reindex(model_order)

        # One row per predictor; em dash if not in model / not estimable
        wide = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            ctab = coef_long[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in ctab.index:
                    wide.loc[t, m] = "—"
                    continue
                r = ctab.loc[t]
                if pd.isna(r["beta_std"]):
                    wide.loc[t, m] = "—"
                else:
                    wide.loc[t, m] = f"{float(r['beta_std']):.3f}{r['sig']}"

        wide.index = [label_map.get(t, t) for t in wide.index]

        extra = pd.DataFrame(index=["Constant", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant", m] = "" if pd.isna(fit_long.loc[m, "const_raw"]) else f"{float(fit_long.loc[m, 'const_raw']):.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit_long.loc[m, "R2"]) else f"{float(fit_long.loc[m, 'R2']):.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit_long.loc[m, "Adj_R2"]) else f"{float(fit_long.loc[m, 'Adj_R2']):.3f}"
            extra.loc["N", m] = "" if pd.isna(fit_long.loc[m, "N"]) else str(int(fit_long.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        table = pd.concat([header, wide, extra], axis=0)
        return table, coef_long, fit_long.reset_index()

    def missingness_table(data, vars_):
        rows = []
        for v in vars_:
            rows.append({"var": v, "missing": int(data[v].isna().sum()), "nonmissing": int(data[v].notna().sum())})
        return pd.DataFrame(rows).sort_values(["missing", "var"], ascending=[False, True])

    # -----------------------------
    # Load + restrict to 1993
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower().strip() for c in df.columns]
    if "year" not in df.columns:
        raise ValueError("Required column missing: year")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Variables per mapping
    # -----------------------------
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
        "sex", "age", "race", "relig", "denom", "region", "ballot"
    ] + music_items + tol_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -----------------------------
    # Clean / coerce
    # -----------------------------
    # Music items: keep 1..5 only (DK/missing -> NaN)
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # Core covariates
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
    df["ballot"] = mask_missing(df["ballot"])

    # If ETHNIC exists, keep it; but per mapping instruction, "Hispanic cannot be constructed"
    if "ethnic" in df.columns:
        df["ethnic"] = mask_missing(df["ethnic"])

    # Tolerance items: validate by family
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -----------------------------
    # DV: Musical exclusiveness = count of genres disliked; strict complete-case on all 18
    # -----------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -----------------------------
    # IVs per mapping
    # -----------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: NOT constructible from provided mapping -> omit from models (paper includes it, but we can't)
    # We keep a placeholder column for diagnostics/table but do not use it in estimation.
    d["hispanic_unavailable"] = np.nan

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}; if Protestant and denom missing -> missing
    d["conservative_protestant"] = np.nan
    m_relig = d["relig"].notna()
    d.loc[m_relig, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = np.nan
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across 15 items; sum 0..15
    d["political_intolerance"], intoler_df, pol_complete = build_polintol_complete(d, tol_items)

    # -----------------------------
    # Models (OLS); standardized betas computed from unstandardized model
    # NOTE: Hispanic excluded due to unavailable mapping.
    # -----------------------------
    y = "num_genres_disliked"

    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res1, coef1, fit1, use1 = fit_ols_with_betas(d, y, x_m1, model_names[0])
    res2, coef2, fit2, use2 = fit_ols_with_betas(d, y, x_m2, model_names[1])
    res3, coef3, fit3, use3 = fit_ols_with_betas(d, y, x_m3, model_names[2])

    # -----------------------------
    # Table formatting: standardized betas only; labeled; — for excluded vars
    # -----------------------------
    dv_label = "Number of music genres disliked"
    label_map = {
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "other_race": "Other race",
        "conservative_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "political_intolerance": "Political intolerance",
    }
    row_order = [
        "educ", "income_pc", "prestg80",
        "female", "age", "black", "other_race",
        "conservative_protestant", "no_religion", "southern",
        "political_intolerance",
    ]

    table1, coef_long, fit_long = build_table_table1_style(
        coef_list=[coef1, coef2, coef3],
        fit_list=[fit1, fit2, fit3],
        model_order=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label,
    )

    # -----------------------------
    # Diagnostics / missingness
    # -----------------------------
    diag = pd.DataFrame(
        [{
            "N_year_1993": int(df.shape[0]),
            "N_complete_music_18": int(d.shape[0]),
            "N_model1_listwise": int(fit1.loc[0, "N"]),
            "N_model2_listwise": int(fit2.loc[0, "N"]),
            "N_model3_listwise": int(fit3.loc[0, "N"]),
            "dv_rule": "strict complete-case across 18 music items; dislike=4/5; sum to 0–18",
            "beta_rule": "standardized beta = b_raw * SD(X)/SD(Y) computed on each model estimation sample",
            "polintol_rule": "strict complete-case across 15 tolerance items; sum of 15 intolerance indicators (0..15)",
            "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
            "hispanic_note": "Hispanic not included: no constructible field per provided mapping/instructions",
        }]
    )

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -----------------------------
    # Save outputs (human-readable)
    # -----------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("Cells: standardized coefficients (beta) only; stars from unstandardized OLS p-values.")
    lines.append("Standard errors are not shown.")
    lines.append("— indicates predictor not in model / not estimable.")
    lines.append("")
    lines.append("Table 1-style coefficients (standardized betas):")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fit_long.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 1 vars):")
    lines.append(miss_m1.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 2 vars):")
    lines.append(miss_m2.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 3 vars):")
    lines.append(miss_m3.to_string(index=False))
    lines.append("")
    lines.append("Raw OLS summaries (debug):")
    lines.append("\n==== Model 1 (SES) ====\n" + res1.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + res2.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas (b * SD(X)/SD(Y)) with significance stars from unstandardized OLS p-values.\n")
        f.write("Standard errors not shown.\n")
        f.write("— indicates predictor not included / not estimable.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n")

    # Machine-readable outputs
    table1.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fit_long.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)
    miss_m1.to_csv("./output/missingness_m1.tsv", sep="\t", index=False)
    miss_m2.to_csv("./output/missingness_m2.tsv", sep="\t", index=False)
    miss_m3.to_csv("./output/missingness_m3.tsv", sep="\t", index=False)

    return {
        "table1_style": table1,
        "fit_stats": fit_long,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "missingness_m1": miss_m1,
        "missingness_m2": miss_m2,
        "missingness_m3": miss_m3,
        "estimation_samples": {
            model_names[0]: use1,
            model_names[1]: use2,
            model_names[2]: use3,
        },
    }