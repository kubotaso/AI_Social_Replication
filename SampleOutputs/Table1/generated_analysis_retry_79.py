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

    # Conservative but broad GSS missing code set (do NOT treat 0 as missing)
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        7, 8, 9,
        77, 78, 79, 88, 89, 97, 98, 99,
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

    def pop_sd(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

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
                # In provided extract, COL* are coded 4/5; intolerance corresponds to "not allowed" coded 5
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    def build_polintol_strict(df, items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        answered = intoler.notna().sum(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        ok = answered == len(items)  # strict complete-case as described
        pol.loc[ok] = intoler.loc[ok].sum(axis=1).astype(float)
        return pol, intoler, answered

    def fit_ols_with_standardized_betas(dd, y, xvars, model_name):
        use = dd[[y] + xvars].dropna(how="any").copy()
        if use.shape[0] == 0:
            raise ValueError(f"{model_name}: empty estimation sample after listwise deletion.")

        yy = use[y].astype(float)
        X = sm.add_constant(use[xvars].astype(float), has_constant="add")
        res = sm.OLS(yy.values, X).fit()

        sd_y = pop_sd(yy.values)
        beta = {}
        for v in xvars:
            sd_x = pop_sd(use[v].astype(float).values)
            b = float(res.params.get(v, np.nan))
            if not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0 or not np.isfinite(b):
                beta[v] = np.nan
            else:
                beta[v] = b * (sd_x / sd_y)

        coef = pd.DataFrame(
            {
                "model": model_name,
                "term": xvars,
                "beta_std": [beta.get(v, np.nan) for v in xvars],
                "b_raw": [float(res.params.get(v, np.nan)) for v in xvars],
                "p_raw": [float(res.pvalues.get(v, np.nan)) for v in xvars],
            }
        )
        coef["sig"] = coef["p_raw"].map(stars)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "N": int(use.shape[0]),
                    "R2": float(res.rsquared),
                    "Adj_R2": float(res.rsquared_adj),
                    "const_raw": float(res.params.get("const", np.nan)),
                }
            ]
        )
        return res, coef, fit, use

    def build_table1_style(coef_list, fit_list, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coef_list, ignore_index=True)
        fit_long = pd.concat(fit_list, ignore_index=True).set_index("model").reindex(model_order)

        table = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            cm = coef_long.loc[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in cm.index:
                    table.loc[t, m] = "—"
                else:
                    r = cm.loc[t]
                    if pd.isna(r["beta_std"]):
                        table.loc[t, m] = "—"
                    else:
                        table.loc[t, m] = f"{float(r['beta_std']):.3f}{r['sig']}"

        table.index = [label_map.get(t, t) for t in table.index]

        extra = pd.DataFrame(index=["Constant", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant", m] = "" if pd.isna(fit_long.loc[m, "const_raw"]) else f"{float(fit_long.loc[m, 'const_raw']):.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit_long.loc[m, "R2"]) else f"{float(fit_long.loc[m, 'R2']):.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit_long.loc[m, "Adj_R2"]) else f"{float(fit_long.loc[m, 'Adj_R2']):.3f}"
            extra.loc["N", m] = "" if pd.isna(fit_long.loc[m, "N"]) else str(int(fit_long.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        out = pd.concat([header, table, extra], axis=0)
        return out, coef_long, fit_long.reset_index()

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
    # Variable lists per mapping instruction
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

    base_required = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "relig", "denom", "region", "ballot"
    ]
    required = base_required + music_items + tol_items
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -----------------------------
    # Clean / coerce
    # -----------------------------
    # Music: must be 1..5; DK/missing => NaN; DV uses strict complete-cases on all 18
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # Numeric / categorical
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

    # Hispanic: not constructible from provided documentation; if ETHNIC exists in file, keep it for diagnostics only.
    # Do NOT include Hispanic in models to avoid incorrect sign/coding.
    if "ethnic" in df.columns:
        df["ethnic"] = mask_missing(df["ethnic"])
    else:
        df["ethnic"] = np.nan

    # Tolerance items: validate codes by item type (strict)
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            # Accept 4/5 only in this extract
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -----------------------------
    # DV: musical exclusiveness = count of genres disliked (0-18)
    # Strict rule: exclude if any DK/missing among 18 items
    # -----------------------------
    dv_base = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        dv_base[dc] = dv_base[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    dv_base["num_genres_disliked"] = dv_base[dislike_cols].sum(axis=1).astype(float)

    # -----------------------------
    # IVs
    # -----------------------------
    # Income per capita = REALINC / HOMPOP (require HOMPOP > 0)
    dv_base["income_pc"] = np.nan
    m_inc = dv_base["realinc"].notna() & dv_base["hompop"].notna() & (dv_base["hompop"] > 0)
    dv_base.loc[m_inc, "income_pc"] = (dv_base.loc[m_inc, "realinc"] / dv_base.loc[m_inc, "hompop"]).astype(float)
    dv_base["income_pc"] = dv_base["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female indicator
    dv_base["female"] = np.where(dv_base["sex"].notna(), (dv_base["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    dv_base["black"] = np.where(dv_base["race"].notna(), (dv_base["race"] == 2).astype(float), np.nan)
    dv_base["other_race"] = np.where(dv_base["race"].notna(), (dv_base["race"] == 3).astype(float), np.nan)

    # No religion
    dv_base["no_religion"] = np.where(dv_base["relig"].notna(), (dv_base["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (as implementable with coarse denom):
    # RELIG==1 and DENOM in {1,6,7}. If Protestant but denom missing => missing.
    dv_base["conservative_protestant"] = np.nan
    m_rel = dv_base["relig"].notna()
    dv_base.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = dv_base["relig"].eq(1) & dv_base["relig"].notna()
    dv_base.loc[m_prot, "conservative_protestant"] = np.nan
    m_prot_d = m_prot & dv_base["denom"].notna()
    dv_base.loc[m_prot_d, "conservative_protestant"] = dv_base.loc[m_prot_d, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    dv_base["southern"] = np.where(dv_base["region"].notna(), (dv_base["region"] == 3).astype(float), np.nan)

    # Political intolerance (strict complete across 15 items)
    pol, intoler_df, answered_tol = build_polintol_strict(dv_base, tol_items)
    dv_base["political_intolerance"] = pol

    # -----------------------------
    # Models (simple, faithful, no ad-hoc thresholds)
    # NOTE: Hispanic omitted because not constructible reliably from provided mapping/documentation.
    # -----------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "other_race", "conservative_protestant", "no_religion", "southern"]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res1, coef1, fit1, use1 = fit_ols_with_standardized_betas(dv_base, y, x_m1, model_names[0])
    res2, coef2, fit2, use2 = fit_ols_with_standardized_betas(dv_base, y, x_m2, model_names[1])
    res3, coef3, fit3, use3 = fit_ols_with_standardized_betas(dv_base, y, x_m3, model_names[2])

    # -----------------------------
    # Table formatting: standardized betas only (no SE rows), labeled, em-dash for excluded vars
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

    table1, coef_long, fit_long = build_table1_style(
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
            "N_complete_music_18": int(dv_base.shape[0]),
            "N_model1_listwise": int(fit1.loc[0, "N"]),
            "N_model2_listwise": int(fit2.loc[0, "N"]),
            "N_model3_listwise": int(fit3.loc[0, "N"]),
            "polintol_nonmissing_strict15": int(dv_base["political_intolerance"].notna().sum()),
            "tol_items_answered_mean": float(answered_tol.mean()) if len(answered_tol) else np.nan,
            "tol_items_answered_min": float(answered_tol.min()) if len(answered_tol) else np.nan,
            "tol_items_answered_max": float(answered_tol.max()) if len(answered_tol) else np.nan,
            "note_hispanic": "Hispanic omitted (not reliably constructible from provided mapping).",
        }]
    )

    miss_m1 = missingness_table(dv_base, [y] + x_m1)
    miss_m2 = missingness_table(dv_base, [y] + x_m2)
    miss_m3 = missingness_table(dv_base, [y] + x_m3)

    # -----------------------------
    # Save outputs (human-readable)
    # -----------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("Cells: standardized coefficients (beta) only; stars from unstandardized OLS p-values.")
    lines.append("Standard errors are not shown. — indicates predictor not included.")
    lines.append("")
    lines.append("DV construction:")
    lines.append("- 18 genre ratings; dislike=4/5; strict complete-case across all 18 (drop any DK/missing).")
    lines.append("")
    lines.append("Political intolerance scale:")
    lines.append("- Sum of 15 intolerance indicators; strict complete-case across all 15 items.")
    lines.append("")
    lines.append("Table 1-style standardized betas:")
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
        f.write("Cells: standardized betas with stars from unstandardized OLS p-values.\n")
        f.write("Standard errors not shown.\n")
        f.write("— indicates predictor not included.\n")
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