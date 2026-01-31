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

    # Conservative sentinel missing values common in GSS extracts.
    # (Do NOT treat 0 as missing globally.)
    MISSING_CODES = {
        8, 9, 98, 99, 998, 999, 9998, 9999,
        -1, -2, -3, -4, -5, -6, -7, -8, -9
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(series, valid_codes):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_codes)), np.nan)

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

    def weighted_mean(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

    def weighted_sd_pop(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() < 2:
            return np.nan
        mu = np.sum(w[m] * x[m]) / np.sum(w[m])
        var = np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])
        return float(np.sqrt(var))

    def build_table_cells(coef_long, model_order, row_order, pretty_row):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_order)
        wide = wide.fillna("—")
        wide.index = [pretty_row.get(t, t) for t in wide.index]
        return wide

    # Political intolerance coding per provided mapping
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

    def fit_ols_and_standardized_betas(data, y, xvars, model_name):
        """
        OLS on raw scale; compute standardized betas as:
            beta_j = b_j * SD(X_j) / SD(Y)
        using population SD (ddof=0) on the *estimation sample*.
        """
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()

        # Drop any predictor with zero variance in estimation sample (avoid singular matrix)
        x_keep = [v for v in xvars if dd[v].nunique(dropna=True) > 1]
        dropped = [v for v in xvars if v not in x_keep]

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = float(yy.std(ddof=0)) if yy.notna().sum() >= 2 else np.nan

        rows = []
        for v in xvars:
            if v in x_keep:
                sd_x = float(dd[v].astype(float).std(ddof=0)) if dd[v].notna().sum() >= 2 else np.nan
                b = float(res.params.get(v, np.nan))
                beta = np.nan
                if np.isfinite(sd_x) and np.isfinite(sd_y) and sd_x != 0 and sd_y != 0:
                    beta = b * (sd_x / sd_y)
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

    # -------------------------
    # Load + basic preparation
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
        "sex", "age", "race", "relig", "denom", "region", "ballot"
    ]
    needed = core_needed + music_items + tol_items
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music items: valid 1..5, DK/missing -> NaN
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # Core numeric/categorical
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

    # Tolerance items: restrict to valid codes by type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: musical exclusiveness (count disliked across 18), listwise complete on all 18
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
    # Income per capita: REALINC/HOMPOP, require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference), NO Hispanic variable available in this extract -> do not fabricate
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Religion dummies
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy using coarse DENOM
    # 1 if RELIG==1 and DENOM in {1,6,7}; else 0; missing if RELIG missing
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across all 15 items (per summary)
    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    d["political_intolerance"] = np.where(intoler.notna().all(axis=1), intoler.sum(axis=1).astype(float), np.nan)

    # -------------------------
    # Models
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res1, tab1, fit1, dd1 = fit_ols_and_standardized_betas(d, y, x_m1, model_names[0])
    res2, tab2, fit2, dd2 = fit_ols_and_standardized_betas(d, y, x_m2, model_names[1])
    res3, tab3, fit3, dd3 = fit_ols_and_standardized_betas(d, y, x_m3, model_names[2])

    coef_long = pd.concat([tab1, tab2, tab3], ignore_index=True)
    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Table 1-style display (standardized betas only + stars; no SE rows)
    # -------------------------
    pretty_row = {
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
        "political_intolerance"
    ]

    table_main = build_table_cells(coef_long, model_names, row_order, pretty_row)

    fit = fitstats.set_index("model").reindex(model_names)
    extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
    for m in model_names:
        extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
        extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
        extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
        extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

    dv_label = "Number of music genres disliked"
    header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
    table1 = pd.concat([header, table_main, extra], axis=0)

    # -------------------------
    # Diagnostics for sample size mismatches (keep simple, factual)
    # -------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1": int(dd1.shape[0]),
        "N_model2": int(dd2.shape[0]),
        "N_model3": int(dd3.shape[0]),
        "political_intolerance_nonmissing": int(d["political_intolerance"].notna().sum()),
        "note_hispanic": "No Hispanic variable in provided fields; not included in models.",
        "note_polintol": "Political intolerance requires complete non-missing on all 15 tolerance items (strict).",
        "note_betas": "Standardized betas computed as b * SD(X)/SD(Y) on each model estimation sample; OLS unweighted.",
    }])

    # -------------------------
    # Save outputs
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"Dependent variable: {dv_label}")
    lines.append("DV construction: 18 music items; disliked = response 4 or 5; listwise complete across all 18 items.")
    lines.append("")
    lines.append("Estimation: OLS (unweighted).")
    lines.append("Displayed coefficients: standardized betas with significance stars from raw OLS p-values.")
    lines.append("Stars: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Table 1-style standardized coefficients:")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Model fit stats:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
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
        f.write("Cells: standardized betas (b*SDx/SDy) with stars from raw OLS p-values.\n")
        f.write("— indicates predictor not included in that model or dropped due to zero variance.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n")

    # TSV exports
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