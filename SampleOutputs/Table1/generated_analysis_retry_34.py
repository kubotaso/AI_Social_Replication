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

    # Common GSS-style "missing" sentinels across extracts (keep conservative; do not treat 0 as missing)
    MISSING_CODES = {
        8, 9, 98, 99, 998, 999, 9998, 9999,
        -1, -2, -3, -4, -5, -6, -7, -8, -9
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(s, valid):
        s = mask_missing(s)
        return s.where(s.isin(list(valid)), np.nan)

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
        """
        Coding per mapping instruction:
          SPK*: 1 allowed, 2 not allowed -> intolerant=1 if 2
          LIB*: 1 remove, 2 not remove -> intolerant=1 if 1
          COL*: generally 4 allowed, 5 not allowed -> intolerant=1 if 5
                COLCOM special: 4 fired, 5 not fired -> intolerant=1 if 4
        """
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)
        elif col.startswith("col"):
            if col == "colcom":
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    def fit_ols_std_betas(data, y, xvars, model_name):
        """
        OLS on raw scales; standardized betas computed on the estimation sample:
            beta_j = b_j * SD(X_j) / SD(Y)
        SD uses population SD (ddof=0), matching common published-beta practice.
        """
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()

        # Avoid singularity if any predictor has no variance
        x_keep = [v for v in xvars if dd[v].nunique(dropna=True) > 1]
        dropped = [v for v in xvars if v not in x_keep]

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = float(yy.std(ddof=0)) if yy.notna().sum() >= 2 else np.nan

        rows = []
        for v in xvars:
            if v in x_keep:
                sx = float(dd[v].astype(float).std(ddof=0)) if dd[v].notna().sum() >= 2 else np.nan
                b = float(res.params.get(v, np.nan))
                beta = np.nan
                if np.isfinite(sx) and np.isfinite(sd_y) and sx != 0 and sd_y != 0:
                    beta = b * (sx / sd_y)
                p = float(res.pvalues.get(v, np.nan))
                cell = "—" if not np.isfinite(beta) else f"{beta:.3f}{stars(p)}"
            else:
                beta = np.nan
                p = np.nan
                cell = "—"
            rows.append(
                {
                    "model": model_name,
                    "term": v,
                    "beta_std": beta,
                    "p_raw": p,
                    "b_raw": float(res.params.get(v, np.nan)),
                    "cell": cell,
                    "dropped_zero_var": v in dropped,
                }
            )

        fit = {
            "model": model_name,
            "N": int(dd.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
            "dropped_zero_variance": ", ".join(dropped) if dropped else "",
        }
        return res, pd.DataFrame(rows), pd.DataFrame([fit]), dd

    def make_table(coef_long, fitstats, model_names, row_order, pretty_row, dv_label):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_names).fillna("—")
        wide.index = [pretty_row.get(t, t) for t in wide.index]

        fit = fitstats.set_index("model").reindex(model_names)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        table = pd.concat([header, wide, extra], axis=0)
        return table

    # -------------------------
    # Load + restrict year
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
    core_needed = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"
    ]
    needed = core_needed + music_items + tol_items
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    df["sex"] = keep_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_valid(df["race"], {1, 2, 3})
    df["ethnic"] = mask_missing(df["ethnic"])  # will be mapped to hispanic below
    df["relig"] = keep_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = keep_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # tolerance items: keep only substantive codes by item type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: musical exclusiveness = count of genres disliked (4/5), listwise complete on all 18
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
    # Income per capita: REALINC / HOMPOP, require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic (required): best-available field in this extract is ETHNIC.
    # Use ETHNIC==1 as Hispanic (common GSS coding in some extracts).
    # If ETHNIC is missing, keep hispanic missing (do not assume 0) to avoid misclassification.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # Religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy
    # 1 if RELIG==1 (Protestant) and DENOM in {1,6,7}; else 0; missing if RELIG missing
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance scale:
    # To address N mismatch from strict listwise deletion, allow partial completion while still
    # adhering to a "count across items" concept:
    #   - compute intolerance indicators with missing for DK/NA
    #   - require at least MIN_TOL_ANSWERED items answered
    #   - sum across answered items (missing contribute 0 by omission)
    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    answered = intoler.notna().sum(axis=1)

    # Choose threshold close to "battery asked to ~2/3" without collapsing N due to scattered DK.
    # This is intentionally conservative but not strictly listwise across 15.
    MIN_TOL_ANSWERED = 12
    d["political_intolerance"] = np.where(
        answered >= MIN_TOL_ANSWERED,
        intoler.fillna(0.0).sum(axis=1).astype(float),
        np.nan
    )

    # -------------------------
    # Models (match Table 1 variable sets)
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res1, tab1, fit1, dd1 = fit_ols_std_betas(d, y, x_m1, model_names[0])
    res2, tab2, fit2, dd2 = fit_ols_std_betas(d, y, x_m2, model_names[1])
    res3, tab3, fit3, dd3 = fit_ols_std_betas(d, y, x_m3, model_names[2])

    coef_long = pd.concat([tab1, tab2, tab3], ignore_index=True)
    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Table 1-style display (standardized betas only)
    # -------------------------
    pretty_row = {
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
        "political_intolerance"
    ]

    dv_label = "Number of music genres disliked"
    table1 = make_table(coef_long, fitstats, model_names, row_order, pretty_row, dv_label)

    # -------------------------
    # Diagnostics (focused on prior issues)
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1": int(dd1.shape[0]),
        "N_model2": int(dd2.shape[0]),
        "N_model3": int(dd3.shape[0]),
        "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
        "hispanic_1_count": int((d["hispanic"] == 1).sum(skipna=True)),
        "political_intolerance_nonmissing": int(d["political_intolerance"].notna().sum()),
        "tol_items_answered_mean": float(answered.mean()) if len(answered) else np.nan,
        "tol_items_answered_min": float(answered.min()) if len(answered) else np.nan,
        "tol_items_answered_max": float(answered.max()) if len(answered) else np.nan,
        "tol_min_answered_rule": int(MIN_TOL_ANSWERED),
        "note": "Table shows standardized betas only; stars from raw OLS p-values (SE method may differ from paper)."
    }])

    # -------------------------
    # Save outputs
    # -------------------------
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    summary_lines.append("")
    summary_lines.append(f"Dependent variable: {dv_label}")
    summary_lines.append("DV construction: 18 music items; disliked = response 4 or 5; listwise complete across all 18 items.")
    summary_lines.append("")
    summary_lines.append("Models: OLS (unweighted).")
    summary_lines.append("Displayed coefficients: standardized betas computed as b * SD(X) / SD(Y) on each model estimation sample.")
    summary_lines.append("Stars: from raw OLS p-values (* p<.05, ** p<.01, *** p<.001).")
    summary_lines.append("")
    summary_lines.append("Political intolerance scale: sum of 15 intolerance indicators; requires at least "
                         f"{MIN_TOL_ANSWERED}/15 items answered; missing items treated as 0 in the sum.")
    summary_lines.append("")
    summary_lines.append("Table 1-style standardized coefficients:")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Model fit stats:")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Diagnostics:")
    summary_lines.append(diag.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Raw OLS summaries (debug):")
    summary_lines.append("\n==== Model 1 (SES) ====\n" + res1.summary().as_text())
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + res2.summary().as_text())
    summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3.summary().as_text())

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("— indicates predictor not included in that model (or dropped for zero variance).\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
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
        "estimation_samples": {"m1": dd1, "m2": dd2, "m3": dd3},
    }