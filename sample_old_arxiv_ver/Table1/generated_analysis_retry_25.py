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

    # GSS-style missing codes (do not treat 0 as missing)
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_codes(series, valid_codes):
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

    def zscore(s):
        s = s.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return s * np.nan
        return (s - mu) / sd

    # -------------------------
    # Political intolerance (strict: require all 15 items present)
    # -------------------------
    def intolerance_indicator(col, s):
        """
        Returns intolerant (1/0) with NaN for missing/invalid.
        Coding per mapping instruction.
        """
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            if col == "colcom":
                # 4=yes fired (intolerant), 5=not fired (tolerant)
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                # 4=allowed, 5=not allowed (intolerant)
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    def build_polintol_strict(df, tol_items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        pol = intoler.sum(axis=1, min_count=len(tol_items))
        pol = pol.where(intoler.notna().all(axis=1), np.nan)
        return pol, intoler

    # -------------------------
    # OLS on z-scored y and x (standardized betas = coefficients)
    # -------------------------
    def fit_standardized_ols(data, y, xvars, model_name):
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()

        # Drop no-variance predictors within estimation sample
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        if dd.shape[0] == 0 or len(x_keep) == 0:
            coef_long = pd.DataFrame(
                [{"model": model_name, "term": v, "cell": "—", "beta_std": np.nan, "p_raw": np.nan} for v in xvars]
            )
            fit = pd.DataFrame([{"model": model_name, "N": int(dd.shape[0]), "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan,
                                 "dropped_no_variance": ", ".join(dropped) if dropped else ""}])
            return None, coef_long, fit, dd

        # Standardize within estimation sample
        y_z = zscore(dd[y])
        Xz = pd.DataFrame({v: zscore(dd[v]) for v in x_keep})
        X = sm.add_constant(Xz, has_constant="add")
        res = sm.OLS(y_z, X).fit()

        # Build coefficient rows (betas only; stars based on model p-values)
        rows = []
        for v in xvars:
            if v in x_keep:
                beta = float(res.params.get(v, np.nan))
                p = float(res.pvalues.get(v, np.nan))
                cell = "—" if not np.isfinite(beta) else f"{beta:.3f}{stars(p)}"
                rows.append({"model": model_name, "term": v, "cell": cell, "beta_std": beta, "p_raw": p})
            else:
                rows.append({"model": model_name, "term": v, "cell": "—", "beta_std": np.nan, "p_raw": np.nan})

        coef_long = pd.DataFrame(rows)
        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": int(dd.shape[0]),
                "R2": float(res.rsquared),
                "Adj_R2": float(res.rsquared_adj),
                # Constant in standardized model is ~0 and not comparable; report raw-model constant too
                "const_raw": np.nan,
                "dropped_no_variance": ", ".join(dropped) if dropped else "",
            }]
        )

        # Also compute raw-model constant and fit with same estimation sample for reporting
        X_raw = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        res_raw = sm.OLS(dd[y].astype(float), X_raw).fit()
        fit.loc[0, "const_raw"] = float(res_raw.params.get("const", np.nan))
        return res, coef_long, fit, dd

    def build_table(coef_long, fitstats, row_order, model_order, label_map, dv_label):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_order).fillna("—")
        wide.index = [label_map.get(t, t) for t in wide.index]

        fit = fitstats.set_index("model").reindex(model_order)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

        # Add a DV header row (as a separate single-row frame for text output readability)
        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        out = pd.concat([header, wide, extra], axis=0)
        return out

    def missingness_table(df, vars_):
        out = []
        for v in vars_:
            out.append({"var": v, "missing": int(df[v].isna().sum()), "nonmissing": int(df[v].notna().sum())})
        return pd.DataFrame(out).sort_values(["missing", "var"], ascending=[False, True])

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

    required = (
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
        + music_items
        + tol_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music ratings: 1..5 valid
    for c in music_items:
        df[c] = keep_codes(df[c], {1, 2, 3, 4, 5})

    # Core covariates
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])
    df["sex"] = keep_codes(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_codes(df["race"], {1, 2, 3})
    df["region"] = keep_codes(df["region"], {1, 2, 3, 4})
    df["relig"] = keep_codes(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items validity
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: musical exclusiveness = count disliked across 18; require complete on all 18
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita = REALINC / HOMPOP; require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: ETHNIC==1 => Hispanic; if ETHNIC missing, assume non-Hispanic (0)
    # This avoids collapsing N due to item nonresponse in this extract.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), 0.0)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant (proxy from coarse DENOM): Protestant AND denom in {1,6,7}
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict count across all 15 items (complete cases only)
    d["political_intolerance"], intoler_df = build_polintol_strict(d, tol_items)

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

    m1_res, c1, f1, s1 = fit_standardized_ols(d, y, x_m1, model_names[0])
    m2_res, c2, f2, s2 = fit_standardized_ols(d, y, x_m2, model_names[1])
    m3_res, c3, f3, s3 = fit_standardized_ols(d, y, x_m3, model_names[2])

    coef_long = pd.concat([c1, c2, c3], ignore_index=True)
    fitstats = pd.concat([f1, f2, f3], ignore_index=True)

    # -------------------------
    # Build Table 1-style output (betas only; no SEs)
    # -------------------------
    pretty = {
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
    dv_label = "Number of music genres disliked (0–18)"
    table1 = build_table(coef_long, fitstats, row_order, model_names, pretty, dv_label=dv_label)

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame(
        [{
            "N_year_1993": int(df.shape[0]),
            "N_complete_music_18": int(d.shape[0]),
            "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
            "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
            "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
            "hispanic_assumed0_when_ethnic_missing": int((d["ethnic"].isna()).sum()),
            "hispanic_1_count": int((d["hispanic"] == 1).sum()),
            "polintol_nonmissing_strict15": int(d["political_intolerance"].notna().sum()),
        }]
    )

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs
    # -------------------------
    lines = []
    lines.append("Replication: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV (all models): {dv_label}")
    lines.append("DV construction: 18 music items; dislike=4/5; requires complete responses on all 18 items.")
    lines.append("")
    lines.append("Displayed coefficients: standardized betas from OLS on z-scored variables (within each model estimation sample).")
    lines.append("Stars: from OLS p-values (* p<.05, ** p<.01, *** p<.001).")
    lines.append("Note: Table prints betas only (no standard errors).")
    lines.append("")
    lines.append("Table 1-style standardized coefficients (betas only):")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Model fit stats:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 1 variables:")
    lines.append(miss_m1.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 2 variables:")
    lines.append(miss_m2.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 3 variables:")
    lines.append(miss_m3.to_string(index=False))
    lines.append("")
    lines.append("Raw OLS summaries (debug; not part of Table 1 formatting):")
    if m1_res is not None:
        lines.append("\n==== Model 1 (SES), standardized regression ====\n" + m1_res.summary().as_text())
    if m2_res is not None:
        lines.append("\n==== Model 2 (Demographic), standardized regression ====\n" + m2_res.summary().as_text())
    if m3_res is not None:
        lines.append("\n==== Model 3 (Political intolerance), standardized regression ====\n" + m3_res.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from OLS p-values.\n")
        f.write("No standard errors are displayed.\n\n")
        f.write(table1.to_string())
        f.write("\n")

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
        "estimation_samples": {"m1": s1, "m2": s2, "m3": s3},
    }