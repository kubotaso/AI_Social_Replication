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

    # Conservative set of missing sentinels commonly found in GSS extracts.
    # Important: do NOT treat 0 as missing globally.
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(s, valid):
        s = mask_missing(s)
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

    def pop_sd(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

    def fit_ols_and_std_betas(data, y, xvars, model_name):
        cols = [y] + xvars
        use = data.loc[:, cols].dropna(how="any").copy()

        # Drop predictors with no variance in this estimation sample
        x_keep, dropped = [], []
        for v in xvars:
            if use[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        if use.shape[0] == 0 or len(x_keep) == 0:
            coef = pd.DataFrame(
                {
                    "model": model_name,
                    "term": xvars,
                    "included": [False] * len(xvars),
                    "beta_std": [np.nan] * len(xvars),
                    "b_raw": [np.nan] * len(xvars),
                    "p_raw": [np.nan] * len(xvars),
                    "sig": [""] * len(xvars),
                }
            )
            fit = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "N": int(use.shape[0]),
                        "R2": np.nan,
                        "Adj_R2": np.nan,
                        "const_raw": np.nan,
                        "dropped_no_variance": ", ".join(dropped) if dropped else "",
                    }
                ]
            )
            return None, coef, fit, use

        X = sm.add_constant(use[x_keep].astype(float), has_constant="add")
        yy = use[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = pop_sd(yy.values)
        beta = {}
        for v in x_keep:
            sd_x = pop_sd(use[v].values)
            b = res.params.get(v, np.nan)
            if np.isfinite(b) and np.isfinite(sd_x) and np.isfinite(sd_y) and sd_x != 0 and sd_y != 0:
                beta[v] = float(b) * (sd_x / sd_y)
            else:
                beta[v] = np.nan

        coef = pd.DataFrame(
            {
                "model": model_name,
                "term": xvars,
                "included": [v in x_keep for v in xvars],
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
                    "dropped_no_variance": ", ".join(dropped) if dropped else "",
                }
            ]
        )
        return res, coef, fit, use

    def build_table1(coef_tabs, fit_tabs, model_order, row_order, label_map, dv_label):
        long = pd.concat(coef_tabs, ignore_index=True)

        def cell(row):
            if not bool(row["included"]):
                return "—"
            if pd.isna(row["beta_std"]):
                return "—"
            return f"{row['beta_std']:.3f}{row['sig']}"

        long["cell"] = long.apply(cell, axis=1)
        wide = long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(row_order)
        wide = wide.reindex(columns=model_order)
        wide.index = [label_map.get(t, t) for t in wide.index]

        fit = pd.concat(fit_tabs, ignore_index=True).set_index("model").reindex(model_order)

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        out = pd.concat([header, wide, extra], axis=0)
        return out, long, fit.reset_index()

    def missingness_table(df, vars_):
        rows = []
        for v in vars_:
            rows.append(
                {"var": v, "nonmissing": int(df[v].notna().sum()), "missing": int(df[v].isna().sum())}
            )
        return pd.DataFrame(rows).sort_values(["missing", "var"], ascending=[False, True])

    # Political intolerance coding
    def intolerance_indicator(col, s):
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            # Most COL* in this extract appear as 4/5; COLCOM special
            if col == "colcom":
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

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
        "sex", "age", "race", "relig", "denom", "region", "ballot"
    ] + music_items + tol_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music ratings: keep 1..5 only (DK/missing -> NaN)
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

    # Tolerance items: mask missing; then restrict to valid codes by item family
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            # Keep 4/5 only
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: number of genres disliked (strict complete-case on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

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

    # Hispanic: not available in provided variable list -> omit from models to avoid incorrect construction
    # (The paper includes it; this extract does not reliably support it.)

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

    # Political intolerance scale:
    # Use strict complete-case across all 15 items to match the mapping summary.
    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    pol_ok = intoler.notna().all(axis=1)
    d["political_intolerance"] = np.nan
    d.loc[pol_ok, "political_intolerance"] = intoler.loc[pol_ok].sum(axis=1).astype(float)

    # -------------------------
    # Models (simple, faithful, listwise deletion per model)
    # NOTE: Hispanic omitted due to unavailable/ambiguous construction in this extract.
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    m1, tab1, fit1, use1 = fit_ols_and_std_betas(d, y, x_m1, model_names[0])
    m2, tab2, fit2, use2 = fit_ols_and_std_betas(d, y, x_m2, model_names[1])
    m3, tab3, fit3, use3 = fit_ols_and_std_betas(d, y, x_m3, model_names[2])

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

    table1, coef_long, fit_for_display = build_table1(
        coef_tabs=[tab1, tab2, tab3],
        fit_tabs=[fit1, fit2, fit3],
        model_order=model_names,
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
        "N_model1_listwise": int(use1.shape[0]),
        "N_model2_listwise": int(use2.shape[0]),
        "N_model3_listwise": int(use3.shape[0]),
        "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
        "polintol_complete_case_rule": "15/15 items required",
        "note_hispanic": "Hispanic omitted: not reliably constructible from provided fields.",
    }])

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

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
    lines.append("Important note:")
    lines.append("  - This dataset extract does not include a clear Hispanic/ethnicity variable suitable for Table 1; Hispanic is omitted.")
    lines.append("  - Political intolerance is computed as a strict 15/15 complete-case sum across the tolerance battery.")
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
    for name, res in [(model_names[0], m1), (model_names[1], m2), (model_names[2], m3)]:
        lines.append(f"\n==== {name} ====")
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
        "estimation_samples": {
            model_names[0]: use1,
            model_names[1]: use2,
            model_names[2]: use3,
        },
    }