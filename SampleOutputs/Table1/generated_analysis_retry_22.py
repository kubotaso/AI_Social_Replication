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

    # Common GSS-style missing sentinels seen in many extracts.
    # NOTE: Do NOT treat 0 as missing globally.
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid_set):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_set)), np.nan)

    def zscore(s):
        s = s.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if (not np.isfinite(sd)) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

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
        Coding exactly per mapping:
          SPK*: 1 allowed, 2 not allowed -> intolerant=1 if 2
          LIB*: 1 remove, 2 not remove -> intolerant=1 if 1
          COL*: (4/5) with special case COLCOM (4 fired, 5 not fired)
                - COLCOM intolerant=1 if 4
                - others intolerant=1 if 5
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

    def fit_model(df, y, xvars, model_name):
        """
        Table 1 reports standardized coefficients (betas).
        Compute by: z-score y and included x's within the model estimation sample,
        fit OLS on standardized vars, report slopes.
        For fit stats (R2/AdjR2/Intercept), fit raw-scale OLS on the same sample.
        """
        cols = [y] + xvars
        d = df.loc[:, cols].dropna(how="any").copy()
        n = int(d.shape[0])

        # Standardized regression
        yz = zscore(d[y])
        Xz = pd.DataFrame(index=d.index)
        keep = []
        dropped_no_var = []
        for v in xvars:
            zv = zscore(d[v])
            if zv.notna().all():
                Xz[v] = zv
                keep.append(v)
            else:
                dropped_no_var.append(v)

        if len(keep) > 0:
            X_std = sm.add_constant(Xz[keep].astype(float), has_constant="add")
            res_std = sm.OLS(yz.astype(float), X_std).fit()
        else:
            # intercept-only standardized fit
            res_std = sm.OLS(yz.astype(float), np.ones((len(yz), 1), dtype=float)).fit()

        # Raw-scale regression for intercept and R2
        X_raw = sm.add_constant(d[xvars].astype(float), has_constant="add")
        res_raw = sm.OLS(d[y].astype(float), X_raw).fit()

        rows = []
        for v in xvars:
            if v in keep:
                b = float(res_std.params.get(v, np.nan))
                p = float(res_std.pvalues.get(v, np.nan))
                cell = f"{b:.3f}{stars(p)}" if np.isfinite(b) else "—"
            else:
                b = np.nan
                p = np.nan
                cell = "—"
            rows.append({"model": model_name, "term": v, "beta_std": b, "p": p, "cell": cell})

        coef_long = pd.DataFrame(rows)
        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": n,
                "R2": float(res_raw.rsquared) if n > 0 else np.nan,
                "Adj_R2": float(res_raw.rsquared_adj) if n > 0 else np.nan,
                "const_raw": float(res_raw.params.get("const", np.nan)) if n > 0 else np.nan,
                "dropped_no_variance": ", ".join(dropped_no_var) if dropped_no_var else ""
            }]
        )
        return res_std, res_raw, coef_long, fit, d

    def missingness_table(df, vars_):
        out = []
        for v in vars_:
            out.append({"var": v, "missing": int(df[v].isna().sum()), "nonmissing": int(df[v].notna().sum())})
        return pd.DataFrame(out).sort_values(["missing", "var"], ascending=[False, True])

    # -------------------------
    # Load
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Required column missing: year")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Variables per mapping
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
    # Clean/coerce columns
    # -------------------------
    # Music: only 1..5 are valid substantive categories
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # Core
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])
    df["sex"] = coerce_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = coerce_valid(df["race"], {1, 2, 3})
    df["ethnic"] = mask_missing(df["ethnic"])  # coding varies by extract; will derive hispanic conservatively
    df["relig"] = coerce_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = coerce_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: apply valid-code filters by type (after masking missing sentinels)
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count of genres disliked (complete-case across all 18)
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
    # Income per capita: realinc / hompop (require hompop > 0)
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Occupational prestige already in prestg80

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference), plus Hispanic dummy derived from ETHNIC if possible
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic:
    # Use ETHNIC if present. Because coding varies across extracts and we have no codebook here,
    # implement the most defensible rule available from the provided data alone:
    # - If ETHNIC is missing -> missing (do NOT assume 0; avoids silent miscoding).
    # - If ETHNIC is 1..99 -> treat ETHNIC==1 as Hispanic, else non-Hispanic.
    # This matches the earlier "best attempt" assumption but without forcing missing -> 0.
    d["hispanic"] = np.nan
    m_eth = d["ethnic"].notna()
    d.loc[m_eth, "hispanic"] = (d.loc[m_eth, "ethnic"] == 1).astype(float)

    # Religion dummies
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (coarse) using RELIG + DENOM as provided
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across all 15 items, sum (0-15)
    intoler_mat = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    d["political_intolerance"] = intoler_mat.sum(axis=1, min_count=len(tol_items))

    # -------------------------
    # Model specs (Table 1)
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    # -------------------------
    # Fit models
    # -------------------------
    m1_std, m1_raw, c1, f1, s1 = fit_model(d, y, x_m1, model_names[0])
    m2_std, m2_raw, c2, f2, s2 = fit_model(d, y, x_m2, model_names[1])
    m3_std, m3_raw, c3, f3, s3 = fit_model(d, y, x_m3, model_names[2])

    coef_long = pd.concat([c1, c2, c3], ignore_index=True)
    fitstats = pd.concat([f1, f2, f3], ignore_index=True)

    # -------------------------
    # Build Table 1-style display (NO SE rows)
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
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "conservative_protestant",
        "no_religion",
        "southern",
        "political_intolerance",
    ]

    wide = coef_long.pivot(index="term", columns="model", values="cell")
    wide = wide.reindex(index=row_order, columns=model_names).fillna("—")
    wide.index = [pretty.get(i, i) for i in wide.index]

    fit_w = fitstats.set_index("model").reindex(model_names)
    extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
    for m in model_names:
        extra.loc["Constant (raw)", m] = f"{fit_w.loc[m, 'const_raw']:.3f}" if pd.notna(fit_w.loc[m, "const_raw"]) else ""
        extra.loc["R²", m] = f"{fit_w.loc[m, 'R2']:.3f}" if pd.notna(fit_w.loc[m, "R2"]) else ""
        extra.loc["Adj. R²", m] = f"{fit_w.loc[m, 'Adj_R2']:.3f}" if pd.notna(fit_w.loc[m, "Adj_R2"]) else ""
        extra.loc["N", m] = str(int(fit_w.loc[m, "N"])) if pd.notna(fit_w.loc[m, "N"]) else ""

    table1_style = pd.concat([wide, extra], axis=0)

    # -------------------------
    # Diagnostics / missingness (use DV-complete pool)
    # -------------------------
    diag = pd.DataFrame(
        [{
            "N_year_1993": int(df.shape[0]),
            "N_complete_music_18": int(d.shape[0]),
            "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
            "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
            "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
            "political_intolerance_nonmissing_strict15": int(d["political_intolerance"].notna().sum()),
            "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
            "hispanic_1_count": int((d["hispanic"] == 1).sum(skipna=True)),
        }]
    )
    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    summary_lines.append("")
    summary_lines.append("Dependent variable: Number of music genres disliked (0–18).")
    summary_lines.append("DV construction: across 18 genre ratings; disliked = 4 or 5; requires complete responses on all 18 items.")
    summary_lines.append("")
    summary_lines.append("Models: OLS; table cells are standardized coefficients (betas).")
    summary_lines.append("Standardization method: within each model estimation sample, z-score DV and all predictors; run OLS; report slopes.")
    summary_lines.append("Stars: two-tailed p-values from the standardized regression: * p<.05, ** p<.01, *** p<.001.")
    summary_lines.append("")
    summary_lines.append("Table 1-style coefficients (standardized betas only) + fit statistics:")
    summary_lines.append(table1_style.to_string())
    summary_lines.append("")
    summary_lines.append("Model fit stats (raw-scale regression used for intercept/R2):")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Diagnostics:")
    summary_lines.append(diag.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness within DV-complete sample (Model 1 vars):")
    summary_lines.append(miss_m1.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness within DV-complete sample (Model 2 vars):")
    summary_lines.append(miss_m2.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness within DV-complete sample (Model 3 vars):")
    summary_lines.append(miss_m3.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Standardized-regression summaries (for reference only; Table 1 prints betas only):")
    summary_lines.append("\n==== Model 1 (SES) standardized ====\n" + m1_std.summary().as_text())
    summary_lines.append("\n==== Model 2 (Demographic) standardized ====\n" + m2_std.summary().as_text())
    summary_lines.append("\n==== Model 3 (Political intolerance) standardized ====\n" + m3_std.summary().as_text())
    summary_lines.append("\n==== Model 1 raw-scale ====\n" + m1_raw.summary().as_text())
    summary_lines.append("\n==== Model 2 raw-scale ====\n" + m2_raw.summary().as_text())
    summary_lines.append("\n==== Model 3 raw-scale ====\n" + m3_raw.summary().as_text())

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("DV: Number of music genres disliked\n")
        f.write("Cells: standardized betas (z-scored within model sample) with stars from standardized-regression p-values.\n\n")
        f.write(table1_style.to_string())
        f.write("\n")

    # machine-readable exports
    table1_style.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)
    miss_m1.to_csv("./output/missingness_m1.tsv", sep="\t", index=False)
    miss_m2.to_csv("./output/missingness_m2.tsv", sep="\t", index=False)
    miss_m3.to_csv("./output/missingness_m3.tsv", sep="\t", index=False)

    return {
        "table1_style": table1_style,
        "fit_stats": fitstats,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "missingness_m1": miss_m1,
        "missingness_m2": miss_m2,
        "missingness_m3": miss_m3,
        "estimation_samples": {"m1": s1, "m2": s2, "m3": s3},
    }