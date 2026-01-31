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

    # Conservative set of common GSS sentinel missings; do not treat 0 as missing
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid_values):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_values)), np.nan)

    def sd0(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(np.std(a, ddof=0))

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
        ok = intoler.notna().all(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        pol.loc[ok] = intoler.loc[ok].sum(axis=1).astype(float)
        return pol, intoler.notna().sum(axis=1)

    def standardized_betas_from_raw_ols(data, y, xvars):
        """
        Raw OLS on unstandardized y and X. Then compute standardized betas:
            beta_j = b_j * SD(X_j) / SD(Y)
        SDs computed on the estimation sample after listwise deletion.
        """
        dd = data[[y] + xvars].dropna().copy()

        # If empty, return placeholders (prevents statsmodels zero-size errors)
        if dd.shape[0] == 0:
            coef = pd.DataFrame({
                "term": xvars,
                "included": [False] * len(xvars),
                "beta_std": [np.nan] * len(xvars),
                "b_raw": [np.nan] * len(xvars),
                "p_raw": [np.nan] * len(xvars),
            })
            coef["sig"] = ""
            fit = {"N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan, "dropped_no_variance": ""}
            return None, coef, fit, dd

        # Drop no-variance predictors within estimation sample
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        # If no predictors left, fit intercept-only model (still valid)
        yy = dd[y].astype(float)
        if len(x_keep) == 0:
            X = np.ones((dd.shape[0], 1))
            res = sm.OLS(yy, X).fit()
            coef = pd.DataFrame({
                "term": xvars,
                "included": [False] * len(xvars),
                "beta_std": [np.nan] * len(xvars),
                "b_raw": [np.nan] * len(xvars),
                "p_raw": [np.nan] * len(xvars),
            })
            coef["sig"] = ""
            fit = {
                "N": int(dd.shape[0]),
                "R2": float(res.rsquared),
                "Adj_R2": float(res.rsquared_adj),
                "const_raw": float(res.params[0]),
                "dropped_no_variance": ", ".join(dropped) if dropped else "",
            }
            return res, coef, fit, dd

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")

        # Extra safety: replace inf, ensure finite
        X = X.replace([np.inf, -np.inf], np.nan)
        ok = X.notna().all(axis=1) & yy.notna()
        X = X.loc[ok]
        yy = yy.loc[ok]

        if X.shape[0] == 0:
            coef = pd.DataFrame({
                "term": xvars,
                "included": [False] * len(xvars),
                "beta_std": [np.nan] * len(xvars),
                "b_raw": [np.nan] * len(xvars),
                "p_raw": [np.nan] * len(xvars),
            })
            coef["sig"] = ""
            fit = {"N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan, "dropped_no_variance": ", ".join(dropped) if dropped else ""}
            return None, coef, fit, dd.iloc[0:0].copy()

        res = sm.OLS(yy, X).fit()

        sdy = sd0(yy.values)
        betas = {}
        for v in x_keep:
            sdx = sd0(dd.loc[ok, v].values)
            b = res.params.get(v, np.nan)
            if not np.isfinite(sdx) or not np.isfinite(sdy) or sdx == 0 or sdy == 0:
                betas[v] = np.nan
            else:
                betas[v] = float(b) * (sdx / sdy)

        coef = pd.DataFrame({
            "term": xvars,
            "included": [v in x_keep for v in xvars],
            "beta_std": [betas.get(v, np.nan) for v in xvars],
            "b_raw": [float(res.params.get(v, np.nan)) for v in xvars],
            "p_raw": [float(res.pvalues.get(v, np.nan)) for v in xvars],
        })
        coef["sig"] = coef["p_raw"].map(stars)

        fit = {
            "N": int(X.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
            "dropped_no_variance": ", ".join(dropped) if dropped else "",
        }
        return res, coef, fit, dd.loc[ok.index[ok]].copy() if isinstance(ok, pd.Series) else dd.copy()

    def build_table(coefs_by_model, fits_by_model, model_names, row_order, label_map, dv_label):
        wide = pd.DataFrame(index=row_order, columns=model_names, dtype=object)
        for m in model_names:
            ctab = coefs_by_model[m].set_index("term")
            for t in row_order:
                if t not in ctab.index:
                    wide.loc[t, m] = "—"
                    continue
                r = ctab.loc[t]
                if not bool(r["included"]):
                    wide.loc[t, m] = "—"
                else:
                    b = r["beta_std"]
                    wide.loc[t, m] = "—" if pd.isna(b) else f"{b:.3f}{r['sig']}"
        wide.index = [label_map.get(t, t) for t in wide.index]

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            fit = fits_by_model[m]
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit["const_raw"]) else f"{fit['const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit["R2"]) else f"{fit['R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit["Adj_R2"]) else f"{fit['Adj_R2']:.3f}"
            extra.loc["N", m] = str(int(fit["N"])) if fit["N"] is not None else "0"

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        return pd.concat([header, wide, extra], axis=0)

    # -------------------------
    # Load + filter year 1993
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
    core_required = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "relig", "denom", "region", "ballot", "ethnic"
    ]
    required = core_required + music_items + tol_items
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean / coerce
    # -------------------------
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

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

    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count of genres disliked (strict complete on all 18 items)
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
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: only if ETHNIC is binary {1,2} in this extract; otherwise leave missing
    d["hispanic"] = np.nan
    eth_vals = set(pd.unique(d["ethnic"].dropna().astype(int).values)) if d["ethnic"].notna().any() else set()
    if len(eth_vals) > 0 and eth_vals.issubset({1, 2}):
        d["hispanic"] = (d["ethnic"] == 1).astype(float)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    d["political_intolerance"], pol_answered = build_polintol_strict(d, tol_items)

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

    res1, coef1, fit1, use1 = standardized_betas_from_raw_ols(d, y, x_m1)
    res2, coef2, fit2, use2 = standardized_betas_from_raw_ols(d, y, x_m2)
    res3, coef3, fit3, use3 = standardized_betas_from_raw_ols(d, y, x_m3)

    coefs_by_model = {model_names[0]: coef1, model_names[1]: coef2, model_names[2]: coef3}
    fits_by_model = {model_names[0]: fit1, model_names[1]: fit2, model_names[2]: fit3}

    # -------------------------
    # Table formatting
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
        dv_label=dv_label,
    )

    # -------------------------
    # Diagnostics
    # -------------------------
    hispanic_mode = "not coded (ETHNIC not binary {1,2} in this extract)"
    if d["hispanic"].notna().any():
        hispanic_mode = "from ETHNIC (assumed 1=Hispanic, 2=Not Hispanic)"

    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1_listwise": int(fit1["N"]),
        "N_model2_listwise": int(fit2["N"]),
        "N_model3_listwise": int(fit3["N"]),
        "hispanic_coding": hispanic_mode,
        "hispanic_1_count_in_dv_complete": int((d["hispanic"] == 1).sum()) if d["hispanic"].notna().any() else 0,
        "polintol_answered_mean_in_dv_complete": float(np.nanmean(pol_answered.values)) if len(pol_answered) else np.nan,
        "polintol_answered_min_in_dv_complete": float(np.nanmin(pol_answered.values)) if len(pol_answered) else np.nan,
        "polintol_answered_max_in_dv_complete": float(np.nanmax(pol_answered.values)) if len(pol_answered) else np.nan,
        "note_standardization": "Standardized betas computed from raw OLS: beta=b*SD(x)/SD(y) on each model estimation sample.",
        "note_table": "Table prints standardized betas only (no SE rows). Stars are from raw OLS p-values.",
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
    # Save outputs
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("DV construction: 18 music items; dislike = 4 or 5; strict complete-case across all 18 items (drop any DK/missing on any genre).")
    lines.append("")
    lines.append("Models: OLS with intercept.")
    lines.append("Displayed coefficients: standardized betas computed from raw OLS as beta=b*SD(x)/SD(y) within each model estimation sample.")
    lines.append("Stars: two-tailed p-values from raw (unstandardized) OLS coefficients: * p<.05, ** p<.01, *** p<.001.")
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
    if res1 is not None:
        lines.append("\n==== Model 1 (SES) ====\n" + res1.summary().as_text())
    else:
        lines.append("\n==== Model 1 (SES) ====\nModel could not be estimated (empty estimation sample).")
    if res2 is not None:
        lines.append("\n==== Model 2 (Demographic) ====\n" + res2.summary().as_text())
    else:
        lines.append("\n==== Model 2 (Demographic) ====\nModel could not be estimated (empty estimation sample).")
    if res3 is not None:
        lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3.summary().as_text())
    else:
        lines.append("\n==== Model 3 (Political intolerance) ====\nModel could not be estimated (empty estimation sample).")

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("— indicates predictor not included / not estimated.\n\n")
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