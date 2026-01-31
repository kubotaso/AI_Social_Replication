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

    # Conservative set of GSS-like missing sentinels; keep 0 as valid.
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

    def zscore(s):
        s = s.astype(float)
        m = np.nanmean(s.values)
        sd = np.nanstd(s.values, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return s * np.nan
        return (s - m) / sd

    def ols_betas_and_fit(dd, y, xvars):
        """
        Returns:
          - standardized betas by estimating OLS on z-scored y and X
          - p-values/stars from raw OLS (unstandardized), for table stars
          - raw OLS fit stats (R2, adjR2, constant, N)
        """
        dd = dd[[y] + xvars].dropna().copy()

        # raw
        X_raw = sm.add_constant(dd[xvars].astype(float), has_constant="add")
        y_raw = dd[y].astype(float)
        res_raw = sm.OLS(y_raw, X_raw).fit()

        # standardized (z-scored within estimation sample)
        zX = dd[xvars].apply(zscore)
        zy = zscore(dd[y])
        X_z = sm.add_constant(zX.astype(float), has_constant="add")
        res_z = sm.OLS(zy.astype(float), X_z).fit()

        coef = pd.DataFrame({
            "term": xvars,
            "beta_std": [float(res_z.params.get(v, np.nan)) for v in xvars],
            "b_raw": [float(res_raw.params.get(v, np.nan)) for v in xvars],
            "p_raw": [float(res_raw.pvalues.get(v, np.nan)) for v in xvars],
        })
        coef["sig"] = coef["p_raw"].map(stars)

        fit = {
            "N": int(dd.shape[0]),
            "R2": float(res_raw.rsquared),
            "Adj_R2": float(res_raw.rsquared_adj),
            "const_raw": float(res_raw.params.get("const", np.nan)),
        }
        return res_raw, res_z, coef, fit, dd

    def build_table(coef_by_model, fit_by_model, model_names, row_order, label_map, dv_label):
        # one row per variable; em dash for not in model; no SE rows
        wide = pd.DataFrame(index=row_order, columns=model_names, dtype=object)
        for m in model_names:
            ctab = coef_by_model[m].set_index("term")
            for t in row_order:
                if t in ctab.index:
                    b = ctab.loc[t, "beta_std"]
                    sg = ctab.loc[t, "sig"]
                    wide.loc[t, m] = "—" if pd.isna(b) else f"{b:.3f}{sg}"
                else:
                    wide.loc[t, m] = "—"

        wide.index = [label_map.get(t, t) for t in wide.index]

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            fit = fit_by_model[m]
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit["const_raw"]) else f"{fit['const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit["R2"]) else f"{fit['R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit["Adj_R2"]) else f"{fit['Adj_R2']:.3f}"
            extra.loc["N", m] = str(int(fit["N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        out = pd.concat([header, wide, extra], axis=0)
        return out

    # Political intolerance coding (strict complete-case across 15 items)
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

    def build_polintol_strict(df, tol_items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        # strict: require all 15 present
        ok = intoler.notna().all(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        pol.loc[ok] = intoler.loc[ok].sum(axis=1).astype(float)
        return pol, intoler.notna().sum(axis=1)

    # -------------------------
    # Load + year restriction
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
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "relig", "denom", "region", "ballot"
    ] + music_items + tol_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean/coerce core fields
    # -------------------------
    # Music: 1..5 only; DK/missing -> NaN
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # SES variables
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

    # Tolerance items: mask + keep valid codes only
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count of genres disliked (strict complete on all 18)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IV construction
    # -------------------------
    # Income per capita = REALINC / HOMPOP; require HOMPOP > 0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: not available in this extract -> do not construct / do not include in models
    # (kept out of model 2/3 to avoid incorrect sign/coding)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
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

    # Political intolerance: strict complete-case across 15 items
    d["political_intolerance"], pol_answered = build_polintol_strict(d, tol_items)

    # -------------------------
    # Models (simple + faithful to mapping; omit Hispanic due to unavailability)
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    # Fit (listwise per model)
    res_raw_1, res_z_1, coef1, fit1, use1 = ols_betas_and_fit(d, y, x_m1)
    res_raw_2, res_z_2, coef2, fit2, use2 = ols_betas_and_fit(d, y, x_m2)
    res_raw_3, res_z_3, coef3, fit3, use3 = ols_betas_and_fit(d, y, x_m3)

    coef_by_model = {
        model_names[0]: coef1,
        model_names[1]: coef2,
        model_names[2]: coef3,
    }
    fit_by_model = {
        model_names[0]: fit1,
        model_names[1]: fit2,
        model_names[2]: fit3,
    }

    # -------------------------
    # Table formatting (labeled rows; em dash for excluded; no SE lines)
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

    table1 = build_table(
        coef_by_model=coef_by_model,
        fit_by_model=fit_by_model,
        model_names=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label,
    )

    # -------------------------
    # Diagnostics
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1_listwise": int(use1.shape[0]),
        "N_model2_listwise": int(use2.shape[0]),
        "N_model3_listwise": int(use3.shape[0]),
        "polintol_answered_mean_in_dv_complete": float(np.nanmean(pol_answered.values)) if len(pol_answered) else np.nan,
        "polintol_answered_min_in_dv_complete": float(np.nanmin(pol_answered.values)) if len(pol_answered) else np.nan,
        "polintol_answered_max_in_dv_complete": float(np.nanmax(pol_answered.values)) if len(pol_answered) else np.nan,
        "note_hispanic": "Hispanic not modeled: no unambiguous Hispanic/ethnicity field in provided extract; avoiding incorrect coding.",
        "note_polintol": "Political intolerance is strict complete-case sum across 15 items (0–15).",
        "note_standardization": "Standardized betas estimated by OLS on z-scored y and z-scored X within each model estimation sample.",
    }])

    fitstats = pd.DataFrame([
        {"model": model_names[0], **fit1},
        {"model": model_names[1], **fit2},
        {"model": model_names[2], **fit3},
    ])

    coef_long = pd.concat([
        coef1.assign(model=model_names[0]),
        coef2.assign(model=model_names[1]),
        coef3.assign(model=model_names[2]),
    ], ignore_index=True)

    # -------------------------
    # Save outputs (human-readable text)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("DV construction: 18 music items; dislike = 4 or 5; strict complete-case across all 18 items.")
    lines.append("")
    lines.append("Models: OLS with intercept. Table cells are standardized coefficients (betas).")
    lines.append("Betas computed by OLS on z-scored variables within each model estimation sample.")
    lines.append("Stars: from two-tailed p-values of raw (unstandardized) OLS coefficients: * p<.05, ** p<.01, *** p<.001.")
    lines.append("")
    lines.append("Important: Hispanic covariate omitted because no unambiguous Hispanic/ethnicity variable is available in the provided extract.")
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
    lines.append("Raw OLS summaries (debug):")
    lines.append("\n==== Model 1 (SES) ====\n" + res_raw_1.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + res_raw_2.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + res_raw_3.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("— indicates predictor not included.\n\n")
        f.write(table1.to_string())
        f.write("\n")

    # TSVs for easy checking
    table1.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)

    return {
        "table1_style": table1,
        "fit_stats": fitstats,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "estimation_samples": {
            model_names[0]: use1,
            model_names[1]: use2,
            model_names[2]: use3,
        },
    }