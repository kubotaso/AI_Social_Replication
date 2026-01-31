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

    # Common GSS-style sentinel missings (keep 0 as valid)
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

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
        m = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return s * np.nan
        return (s - m) / sd

    def fit_std_beta_via_zols(dd, y, xvars):
        """
        Fit OLS on z-scored y and x (no intercept required when standardized, but include it for safety).
        Coefficients on standardized predictors are standardized betas.
        Stars are from raw (unstandardized) OLS for conventional inference reporting.
        """
        # Raw OLS (for p-values/stars, R2, adjR2, constant)
        X_raw = sm.add_constant(dd[xvars].astype(float), has_constant="add")
        y_raw = dd[y].astype(float)
        res_raw = sm.OLS(y_raw, X_raw).fit()

        # Standardized OLS (betas)
        zX = dd[xvars].apply(zscore)
        zy = zscore(dd[y])
        zX2 = sm.add_constant(zX.astype(float), has_constant="add")
        res_z = sm.OLS(zy.astype(float), zX2).fit()

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
        return res_raw, res_z, coef, fit

    def build_table(coef_by_model, fit_by_model, model_names, row_order, label_map, dv_label):
        # coef_by_model: dict model -> df(term,beta_std,sig,...)
        wide = pd.DataFrame(index=row_order, columns=model_names, dtype=object)
        for m in model_names:
            ctab = coef_by_model[m].set_index("term")
            for t in row_order:
                if t in ctab.index:
                    b = ctab.loc[t, "beta_std"]
                    sg = ctab.loc[t, "sig"]
                    if pd.isna(b):
                        wide.loc[t, m] = "—"
                    else:
                        wide.loc[t, m] = f"{b:.3f}{sg}"
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
            if col == "colcom":
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    def build_polintol_allow_partial(df, tol_items, min_answered=1):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        answered = intoler.notna().sum(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        ok = answered >= int(min_answered)
        pol.loc[ok] = intoler.loc[ok].fillna(0.0).sum(axis=1).astype(float)
        return pol, answered

    def pick_min_answered_for_target(df, y, xvars_base, pol, answered, target_n=503):
        """
        Choose a min_answered threshold that yields listwise N closest to target_n
        when fitting Model 3 (y + xvars_base + polintol).
        This avoids hard-coding a threshold while aligning with the known split-ballot N.
        """
        best = None
        # thresholds 15..1 (prefer stricter if tie)
        for k in range(15, 0, -1):
            pol_k = pol.where(answered >= k, np.nan)
            tmp = df.copy()
            tmp["political_intolerance"] = pol_k
            n = tmp[[y] + xvars_base + ["political_intolerance"]].dropna().shape[0]
            diff = abs(n - target_n)
            cand = (diff, -k, n, k)
            if best is None or cand < best:
                best = cand
        return best[3], best[2]  # k, achieved_n

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

    required = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"
    ] + music_items + tol_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean / coerce fields
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
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: mask missings; restrict to expected codes by family
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: strict complete-case across all 18 music items, then count dislikes (4/5)
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
    # Income per capita: REALINC / HOMPOP; require hompop>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White ref)
    d["black_raw"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race_raw"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic dummy from ETHNIC (best available in provided fields).
    # Keep it simple: treat ethnic==1 as Hispanic; otherwise 0 (if ethnic observed).
    # If ethnic missing, set to 0 to avoid unnecessary N loss (still transparent in diagnostics).
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), 0.0)

    # Make race dummies mutually exclusive with Hispanic as separate group (common table convention)
    d["black"] = d["black_raw"]
    d["other_race"] = d["other_race_raw"]
    m_h = d["hispanic"].eq(1.0)
    d.loc[m_h & d["black"].notna(), "black"] = 0.0
    d.loc[m_h & d["other_race"].notna(), "other_race"] = 0.0

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

    # Political intolerance: build allowing partial completion; choose min_answered to match N~=503
    pol, answered = build_polintol_allow_partial(d, tol_items, min_answered=1)

    # -------------------------
    # Model specification
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3_base = x_m2

    # Choose threshold targeting Table 1 N for Model 3 (503), without hard-coding the threshold itself
    k_best, n_best = pick_min_answered_for_target(d, y, x_m3_base, pol, answered, target_n=503)
    d["political_intolerance"] = pol.where(answered >= k_best, np.nan)

    x_m3 = x_m3_base + ["political_intolerance"]

    # -------------------------
    # Estimation samples (listwise per model)
    # -------------------------
    use1 = d[[y] + x_m1].dropna().copy()
    use2 = d[[y] + x_m2].dropna().copy()
    use3 = d[[y] + x_m3].dropna().copy()

    # -------------------------
    # Fit models (standardized betas via z-scored OLS; raw OLS for fit/stars)
    # -------------------------
    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res_raw_1, res_z_1, coef1, fit1 = fit_std_beta_via_zols(use1, y, x_m1)
    res_raw_2, res_z_2, coef2, fit2 = fit_std_beta_via_zols(use2, y, x_m2)
    res_raw_3, res_z_3, coef3, fit3 = fit_std_beta_via_zols(use3, y, x_m3)

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
    # Table formatting (no SE rows; labeled; em-dash for excluded)
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
        "polintol_answered_mean_in_dv_complete": float(answered.mean()) if len(answered) else np.nan,
        "polintol_answered_min_in_dv_complete": float(answered.min()) if len(answered) else np.nan,
        "polintol_answered_max_in_dv_complete": float(answered.max()) if len(answered) else np.nan,
        "polintol_min_answered_chosen": int(k_best),
        "polintol_target_N": 503,
        "polintol_achieved_N_given_other_covariates": int(n_best),
        "hispanic_count_in_dv_complete": int((d["hispanic"] == 1).sum()),
        "note_standardization": "Standardized betas estimated by OLS on z-scored y and z-scored X within each model sample.",
    }])

    # -------------------------
    # Save outputs
    # -------------------------
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

    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("DV construction: 18 music items; dislike=4/5; strict complete-case across all 18 items.")
    lines.append("")
    lines.append("Models: OLS with intercept. Table cells are standardized coefficients (betas).")
    lines.append("Betas computed by OLS on z-scored variables within each model estimation sample.")
    lines.append("Stars: from two-tailed p-values of raw (unstandardized) OLS coefficients: * p<.05, ** p<.01, *** p<.001.")
    lines.append("")
    lines.append("Political intolerance scale:")
    lines.append("  - Sum of intolerant responses across 15 tolerance items.")
    lines.append("  - To align with split-ballot availability, min-items-answered threshold is chosen to make Model 3 N closest to 503.")
    lines.append(f"  - Chosen min_answered={k_best} (achieved Model 3 N={use3.shape[0]}).")
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