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

    # Conservative but broader "missing" sentinels commonly appearing in GSS extracts
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        7, 8, 9,
        77, 78, 79,
        87, 88, 89,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999,
        99997, 99998, 99999,
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(s, valid_set):
        s = mask_missing(s)
        return s.where(s.isin(set(valid_set)), np.nan)

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

    def zscore_in_sample(x):
        x = x.astype(float)
        mu = x.mean()
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index, dtype=float)
        return (x - mu) / sd

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
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    def build_polintol_complete(df, items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        # complete-case across all 15 items (per mapping summary)
        complete = intoler.notna().all(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        pol.loc[complete] = intoler.loc[complete].sum(axis=1).astype(float)
        return pol, intoler, complete

    def fit_table1_model(df, y, xvars, model_name):
        """
        Fit:
          - Raw OLS with intercept for conventional R2/AdjR2/constant/p-values
          - Standardized betas computed by running OLS on z-scored y and z-scored X (no intercept)
            on the SAME estimation sample (listwise deletion on raw variables).
        """
        use = df[[y] + xvars].dropna(how="any").copy()
        N = int(use.shape[0])
        if N == 0:
            raise ValueError(f"{model_name}: empty estimation sample after listwise deletion.")

        # Raw model (for R2/AdjR2/constant/p-values for stars)
        X_raw = sm.add_constant(use[xvars].astype(float), has_constant="add")
        y_raw = use[y].astype(float)
        res_raw = sm.OLS(y_raw.values, X_raw.values).fit()

        # Standardized betas (z-score within this same sample)
        yz = zscore_in_sample(use[y])
        Xz = pd.DataFrame({v: zscore_in_sample(use[v]) for v in xvars})
        # Drop any predictors that became all-NA due to zero variance in-sample
        varying = [v for v in xvars if Xz[v].notna().any()]
        coef = pd.DataFrame({"model": model_name, "term": xvars})
        coef["included"] = coef["term"].isin(varying)

        if len(varying) > 0:
            zuse = pd.concat([yz.rename("__y__"), Xz[varying]], axis=1).dropna(how="any")
            if zuse.shape[0] == 0:
                # Degenerate after standardization; mark missing
                coef["beta_std"] = np.nan
            else:
                res_std = sm.OLS(zuse["__y__"].values.astype(float), zuse[varying].values.astype(float)).fit()
                beta_map = {v: float(res_std.params[i]) for i, v in enumerate(varying)}
                coef["beta_std"] = coef["term"].map(beta_map).astype(float)
        else:
            coef["beta_std"] = np.nan

        # Stars: use p-values from the RAW model coefficients (conventional and comparable)
        # (Paper prints stars but not SEs; this keeps reporting faithful without adding SE rows.)
        p_map = {xvars[i]: float(res_raw.pvalues[i + 1]) for i in range(len(xvars))}  # +1 skips const
        coef["p_raw"] = coef["term"].map(p_map).astype(float)
        coef["sig"] = coef["p_raw"].map(stars)

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": N,
                "R2": float(res_raw.rsquared),
                "Adj_R2": float(res_raw.rsquared_adj),
                "const_raw": float(res_raw.params[0]),
            }]
        )
        return res_raw, coef, fit, use

    def build_table1_style(coef_list, fit_list, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coef_list, ignore_index=True)
        fit_long = pd.concat(fit_list, ignore_index=True).set_index("model").reindex(model_order)

        table = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            cm = coef_long.loc[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in cm.index:
                    table.loc[t, m] = "—"
                    continue
                r = cm.loc[t]
                if (not bool(r.get("included", True))) or pd.isna(r["beta_std"]):
                    table.loc[t, m] = "—"
                else:
                    table.loc[t, m] = f"{float(r['beta_std']):.3f}{r['sig']}"

        table.index = [label_map.get(t, t) for t in table.index]

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit_long.loc[m, "const_raw"]) else f"{float(fit_long.loc[m, 'const_raw']):.3f}"
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
    # Variables per mapping instruction
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
    # Music ratings: keep 1..5 only; DK/other -> NaN
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # SES vars
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    # Demographics
    df["sex"] = keep_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_valid(df["race"], {1, 2, 3})
    df["relig"] = keep_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = keep_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items validity by type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -----------------------------
    # DV: musical exclusiveness = count disliked across 18 items (strict complete-case on all 18)
    # -----------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -----------------------------
    # IV construction (match mapping summary; keep simple and strict)
    # -----------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: not constructible from provided mapping; keep as 0 (non-Hispanic) to avoid artificial listwise deletion
    # (This is the only way to keep models runnable and comparable with available variables.)
    d["hispanic"] = 0.0

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (coarse; implementable)
    # RELIG==1 and DENOM in {1,6,7}; if Protestant but denom missing => missing
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = np.nan
    m_prot_d = m_prot & d["denom"].notna()
    d.loc[m_prot_d, "conservative_protestant"] = d.loc[m_prot_d, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across all 15 items (per mapping summary)
    d["political_intolerance"], intoler_df, tol_complete = build_polintol_complete(d, tol_items)

    # -----------------------------
    # Models
    # -----------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    res1_raw, coef1, fit1, use1 = fit_table1_model(d, y, x_m1, model_names[0])
    res2_raw, coef2, fit2, use2 = fit_table1_model(d, y, x_m2, model_names[1])
    res3_raw, coef3, fit3, use3 = fit_table1_model(d, y, x_m3, model_names[2])

    # -----------------------------
    # Table formatting: standardized betas only (no SE rows), labeled, em-dash for excluded/NA
    # -----------------------------
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
            "N_complete_music_18": int(d.shape[0]),
            "N_model1_listwise": int(fit1.loc[0, "N"]),
            "N_model2_listwise": int(fit2.loc[0, "N"]),
            "N_model3_listwise": int(fit3.loc[0, "N"]),
            "black_1_count_in_music_complete": int((d["black"] == 1).sum(skipna=True)),
            "other_race_1_count_in_music_complete": int((d["other_race"] == 1).sum(skipna=True)),
            "political_intolerance_nonmissing": int(d["political_intolerance"].notna().sum()),
            "tolerance_complete_15_count": int(tol_complete.sum()) if tol_complete is not None else np.nan,
        }]
    )

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -----------------------------
    # Save outputs (human-readable)
    # -----------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("Cells: standardized coefficients (beta) only (computed by z-scoring X and Y within each model's estimation sample).")
    lines.append("Stars: based on two-tailed p-values from the raw OLS model (with intercept): * p<.05, ** p<.01, *** p<.001")
    lines.append("Standard errors are not shown. — indicates predictor not included or not estimable.")
    lines.append("")
    lines.append("DV construction: 18 genre ratings; dislike=4/5; strict complete-case across all 18 (drop any DK/missing).")
    lines.append("Income per capita: REALINC / HOMPOP (HOMPOP>0).")
    lines.append("Political intolerance: strict complete-case sum across 15 items (0–15).")
    lines.append("Hispanic: not constructible from provided variable set; set to 0 to avoid inducing missingness.")
    lines.append("")
    lines.append("Table 1-style standardized betas:")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics (raw OLS with intercept on the same estimation samples):")
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
    lines.append("\n==== Model 1 (SES) ====\n" + res1_raw.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + res2_raw.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3_raw.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("Standard errors not shown.\n")
        f.write("— indicates predictor not included or not estimable.\n")
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