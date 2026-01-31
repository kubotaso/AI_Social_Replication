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

    # Broad set of common GSS missing sentinels; do not treat 0 as missing.
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        7, 8, 9,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999,
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

    def zscore(series):
        s = series.astype(float)
        m = s.mean()
        sd = s.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - m) / sd

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

    def build_polintol_prorated_sum(df, items, min_answered=10):
        """
        Paper describes a count 0..15; to avoid excessive N loss while keeping scale comparable,
        compute a prorated sum: (mean across answered items) * 15, requiring >= min_answered.
        """
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        answered = intoler.notna().sum(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        m = answered >= int(min_answered)
        pol.loc[m] = intoler.loc[m].mean(axis=1) * len(items)
        return pol, intoler, answered

    def fit_table1_style(data, y, xvars, model_name):
        """
        Table 1 reports standardized coefficients (betas).
        We estimate OLS on z-scored y and z-scored Xs on the model's estimation sample.
        Then coefficients are directly standardized betas.
        """
        cols = [y] + xvars
        use = data.loc[:, cols].dropna(how="any").copy()
        if use.shape[0] == 0:
            raise ValueError(f"{model_name}: empty estimation sample after listwise deletion.")

        # z-score within estimation sample
        yz = zscore(use[y])
        Xz = pd.DataFrame({v: zscore(use[v]) for v in xvars})
        # drop any columns that became all-NaN due to zero variance
        keep = [v for v in xvars if Xz[v].notna().any()]
        Xz = Xz[keep]

        X = sm.add_constant(Xz, has_constant="add")
        res = sm.OLS(yz, X).fit()

        # Build coefficient table with em-dash for not-in-model or not-estimable
        coef_rows = []
        for v in xvars:
            if v not in keep:
                coef_rows.append(
                    {"model": model_name, "term": v, "beta_std": np.nan, "p_raw": np.nan, "sig": ""}
                )
            else:
                coef_rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "beta_std": float(res.params.get(v, np.nan)),
                        "p_raw": float(res.pvalues.get(v, np.nan)),
                        "sig": stars(float(res.pvalues.get(v, np.nan))),
                    }
                )
        coef = pd.DataFrame(coef_rows)

        # Fit statistics: compute R2/AdjR2 on standardized regression (same as unstandardized R2)
        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "N": int(use.shape[0]),
                    "R2": float(res.rsquared),
                    "Adj_R2": float(res.rsquared_adj),
                }
            ]
        )
        return res, coef, fit, use

    def build_display_table(coefs, fits, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coefs, ignore_index=True)
        fit_long = pd.concat(fits, ignore_index=True)

        wide = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            ctab = coef_long.loc[coef_long["model"] == m].set_index("term")
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

        fits_ix = fit_long.set_index("model").reindex(model_order)
        extra = pd.DataFrame(index=["R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["R²", m] = "" if pd.isna(fits_ix.loc[m, "R2"]) else f"{float(fits_ix.loc[m, 'R2']):.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fits_ix.loc[m, "Adj_R2"]) else f"{float(fits_ix.loc[m, 'Adj_R2']):.3f}"
            extra.loc["N", m] = "" if pd.isna(fits_ix.loc[m, "N"]) else str(int(fits_ix.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        table = pd.concat([header, wide, extra], axis=0)
        return table, coef_long, fit_long

    def missingness_table(data, vars_):
        rows = []
        for v in vars_:
            rows.append(
                {"var": v, "missing": int(data[v].isna().sum()), "nonmissing": int(data[v].notna().sum())}
            )
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
    # Variable lists per mapping
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
        "id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race",
        "relig", "denom", "region", "ballot"
    ] + music_items + tol_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean / coerce
    # -------------------------
    # Music items: 1..5 only; DK/missing -> NaN
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

    # Tolerance items: validate values by family (keep only substantive codes)
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count of genres disliked (STRICT complete-case on all 18)
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
    # Income per capita: REALINC / HOMPOP (HOMPOP>0)
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female indicator
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race indicators (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: not constructible from provided variables; keep as 0 with variance=0 (will be non-estimable)
    # We keep it for table structure but it will show as — once detected as no-variance after z-scoring.
    d["hispanic"] = 0.0

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}; missing if Protestant and DENOM missing
    d["conservative_protestant"] = np.nan
    m_relig = d["relig"].notna()
    d.loc[m_relig, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = np.nan
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: prorated sum with minimum answered to reduce N loss (aiming for split-ballot sample size)
    d["political_intolerance"], intoler_df, pol_answered = build_polintol_prorated_sum(d, tol_items, min_answered=10)

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

    res1, coef1, fit1, use1 = fit_table1_style(d, y, x_m1, model_names[0])
    res2, coef2, fit2, use2 = fit_table1_style(d, y, x_m2, model_names[1])
    res3, coef3, fit3, use3 = fit_table1_style(d, y, x_m3, model_names[2])

    # -------------------------
    # Table formatting: BETAS ONLY; labeled; em-dash for excluded/non-estimable
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

    table1, coef_long, fit_long = build_display_table(
        coefs=[coef1, coef2, coef3],
        fits=[fit1, fit2, fit3],
        model_order=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label,
    )

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "N_year_1993": int(df.shape[0]),
                "N_complete_music_18": int(d.shape[0]),
                "N_model1_listwise": int(fit1.loc[0, "N"]),
                "N_model2_listwise": int(fit2.loc[0, "N"]),
                "N_model3_listwise": int(fit3.loc[0, "N"]),
                "dv_rule": "strict complete-case across all 18 music items; dislike=4/5; sum to 0–18",
                "income_pc_rule": "REALINC/HOMPOP with HOMPOP>0; missing if REALINC or HOMPOP missing",
                "hispanic_rule": "not constructible from provided fields; set to 0 (non-estimable; shown as —)",
                "conservative_protestant_rule": "RELIG==1 and DENOM in {1,6,7}; missing if Protestant and DENOM missing",
                "polintol_rule": "prorated sum across 15 intolerance indicators; require >=10 answered; scale approximately 0–15",
                "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
                "polintol_items_answered_min": float(pol_answered.min()) if len(pol_answered) else np.nan,
                "polintol_items_answered_mean": float(pol_answered.mean()) if len(pol_answered) else np.nan,
                "polintol_items_answered_max": float(pol_answered.max()) if len(pol_answered) else np.nan,
                "betas_rule": "OLS on z-scored DV and z-scored predictors within each model estimation sample; coefficients are standardized betas",
                "stars_rule": "stars from OLS p-values (* p<.05, ** p<.01, *** p<.001)",
            }
        ]
    )

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("Table cells show standardized coefficients (betas) only; standard errors are not printed.")
    lines.append("— indicates predictor not included or not estimable (e.g., no variance).")
    lines.append("")
    lines.append("Standardization method: z-score DV and all predictors within each model estimation sample; run OLS.")
    lines.append("Significance stars: from OLS p-values on the same estimation sample (* p<.05, ** p<.01, *** p<.001).")
    lines.append("")
    lines.append("Table 1-style standardized coefficients (betas):")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics (standardized OLS):")
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
    lines.append("Raw model summaries (debug; standardized regressions):")
    lines.append("\n==== Model 1 (SES) ====\n" + res1.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + res2.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with significance stars.\n")
        f.write("— indicates predictor not included / not estimable.\n")
        f.write("Note: Standard errors are not printed (Table 1 style).\n\n")
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