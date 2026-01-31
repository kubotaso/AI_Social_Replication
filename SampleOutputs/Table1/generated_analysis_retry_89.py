def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    # Broad GSS-style missing codes; keep substantive 0/89, etc.
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        7, 8, 9,
        77, 78, 79,
        87, 88,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999,
        99997, 99998, 99999,
    }

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(series, valid_values):
        s = mask_missing(series)
        return s.where(s.isin(set(valid_values)), np.nan)

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
        Per mapping:
        - SPK*: 2 = not allowed (intolerant=1), 1 = allowed (0)
        - COL*: 5 = not allowed (1), 4 = allowed (0), except COLCOM: 4=fired (1), 5=not fired (0)
        - LIB*: 1 = remove (1), 2 = not remove (0)
        """
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)
        elif col.startswith("col"):
            m = s.isin([4, 5])
            if col == "colcom":
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    def build_polintol_complete_case(df, items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        complete = intoler.notna().all(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        pol.loc[complete] = intoler.loc[complete].sum(axis=1).astype(float)
        return pol, intoler, complete

    def standardize_within_sample(use_df, cols, ddof=1):
        z = {}
        for c in cols:
            x = use_df[c].astype(float).to_numpy()
            mu = np.nanmean(x)
            sd = np.nanstd(x, ddof=ddof)
            if not np.isfinite(sd) or sd == 0:
                z[c] = pd.Series(np.nan, index=use_df.index)
            else:
                z[c] = (use_df[c].astype(float) - mu) / sd
        return pd.DataFrame(z)

    def fit_ols_std_betas_via_z(df, y, xvars, model_name):
        """
        Compute standardized coefficients by:
          1) listwise deletion on y + xvars
          2) z-score y and all xvars within that estimation sample
          3) run OLS(z_y ~ 1 + z_X)
        Coefficients on z_X are standardized betas.
        """
        use = df[[y] + xvars].dropna(how="any").copy()
        if use.shape[0] == 0:
            raise ValueError(f"{model_name}: empty estimation sample after listwise deletion.")

        z = standardize_within_sample(use, [y] + xvars, ddof=1)
        z_y = z[y]
        z_X = sm.add_constant(z[xvars], has_constant="add")

        res = sm.OLS(z_y.astype(float), z_X.astype(float)).fit()

        coef = pd.DataFrame({"model": model_name, "term": xvars})
        coef["beta_std"] = coef["term"].map(lambda t: float(res.params.get(t, np.nan)))
        coef["p_raw"] = coef["term"].map(lambda t: float(res.pvalues.get(t, np.nan)))
        coef["sig"] = coef["p_raw"].map(stars)

        # Fit metrics should correspond to the original (unstandardized) regression with intercept.
        # But for OLS with identical sample and linear transforms, R2 matches.
        fit = pd.DataFrame([{
            "model": model_name,
            "N": int(use.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            # Constant in the paper is from raw model; compute it from raw model for reporting.
            "const_raw": np.nan,
        }])

        # Raw model for constant + full summary diagnostics if needed
        raw_X = sm.add_constant(use[xvars].astype(float), has_constant="add")
        raw_y = use[y].astype(float)
        raw_res = sm.OLS(raw_y, raw_X).fit()
        fit.loc[0, "const_raw"] = float(raw_res.params.get("const", np.nan))

        return raw_res, coef, fit, use

    def build_table1(coef_list, fit_list, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coef_list, ignore_index=True)
        fit_long = pd.concat(fit_list, ignore_index=True).set_index("model").reindex(model_order)

        tbl = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            cm = coef_long.loc[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in cm.index:
                    tbl.loc[t, m] = "—"
                else:
                    r = cm.loc[t]
                    if pd.isna(r["beta_std"]):
                        tbl.loc[t, m] = "—"
                    else:
                        tbl.loc[t, m] = f"{float(r['beta_std']):.3f}{str(r['sig'])}"

        tbl.index = [label_map.get(t, t) for t in tbl.index]

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit_long.loc[m, "const_raw"]) else f"{float(fit_long.loc[m, 'const_raw']):.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit_long.loc[m, "R2"]) else f"{float(fit_long.loc[m, 'R2']):.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit_long.loc[m, "Adj_R2"]) else f"{float(fit_long.loc[m, 'Adj_R2']):.3f}"
            extra.loc["N", m] = "" if pd.isna(fit_long.loc[m, "N"]) else str(int(fit_long.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        out = pd.concat([header, tbl, extra], axis=0)
        return out, coef_long, fit_long.reset_index()

    def missingness_table(df, cols):
        rows = []
        for c in cols:
            rows.append({
                "var": c,
                "missing": int(df[c].isna().sum()),
                "nonmissing": int(df[c].notna().sum()),
            })
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
    # Variables per mapping
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

    required = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"
    ] + music_items + tol_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -----------------------------
    # Clean / coerce base fields
    # -----------------------------
    # Music: only 1..5 valid; DK/missing -> NaN
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # SES numeric
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    # Demographics
    df["sex"] = keep_valid(df["sex"], {1, 2})

    # Age: keep 89 topcode as substantive; mask other missing codes
    age_raw = to_num(df["age"])
    df["age"] = age_raw.mask(age_raw.isin(MISSING_CODES), np.nan)

    df["race"] = keep_valid(df["race"], {1, 2, 3})
    df["region"] = keep_valid(df["region"], {1, 2, 3, 4})
    df["relig"] = keep_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: restrict to valid substantive codes by item type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -----------------------------
    # DV: Musical exclusiveness = count of disliked genres (strict complete-case on 18)
    # -----------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -----------------------------
    # IVs (Table 1)
    # -----------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP>0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female indicator
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: best-available from ETHNIC in this extract.
    # Treat ETHNIC==1 as Hispanic; if ETHNIC coded differently in a given release,
    # this will show up in diagnostics and can be adjusted.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # No religion indicator
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant indicator (proxy per mapping)
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna() & d["denom"].notna()
    d.loc[m_prot, "conservative_protestant"] = d.loc[m_prot, "denom"].isin([1, 6, 7]).astype(float)

    # Southern indicator
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance scale: strict complete-case across 15 items
    d["political_intolerance"], intoler_df, tol_complete = build_polintol_complete_case(d, tol_items)

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

    res1, coef1, fit1, use1 = fit_ols_std_betas_via_z(d, y, x_m1, model_names[0])
    res2, coef2, fit2, use2 = fit_ols_std_betas_via_z(d, y, x_m2, model_names[1])
    res3, coef3, fit3, use3 = fit_ols_std_betas_via_z(d, y, x_m3, model_names[2])

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
        "political_intolerance"
    ]

    table1, coef_long, fit_long = build_table1(
        coef_list=[coef1, coef2, coef3],
        fit_list=[fit1, fit2, fit3],
        model_order=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label
    )

    # -----------------------------
    # Diagnostics + missingness
    # -----------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1_listwise": int(fit1.loc[0, "N"]),
        "N_model2_listwise": int(fit2.loc[0, "N"]),
        "N_model3_listwise": int(fit3.loc[0, "N"]),
        "income_pc_missing_in_music_complete": int(d["income_pc"].isna().sum()),
        "prestg80_missing_in_music_complete": int(d["prestg80"].isna().sum()),
        "hispanic_nonmissing_in_music_complete": int(d["hispanic"].notna().sum()),
        "hispanic_1_count_in_music_complete": int((d["hispanic"] == 1).sum(skipna=True)),
        "political_intolerance_nonmissing_in_music_complete": int(d["political_intolerance"].notna().sum()),
        "tol_complete_case_count_in_music_complete": int(tol_complete.sum()),
    }])

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
    lines.append("Cells: standardized coefficients (beta) only; standard errors are not shown in Table 1.")
    lines.append("Standardization: z-score y and all X within each model's estimation sample, then OLS with intercept.")
    lines.append("Stars: two-tailed p-values from raw OLS with intercept: * p<.05, ** p<.01, *** p<.001")
    lines.append("— indicates predictor not included or missing in a given model (not expected here).")
    lines.append("")
    lines.append("Construction notes:")
    lines.append("- DV: 18 music items; disliked if response in {4,5}; strict complete-case across all 18 items.")
    lines.append("- Income per capita: REALINC / HOMPOP (HOMPOP>0).")
    lines.append("- Political intolerance: sum across 15 items; strict complete-case across all 15 items.")
    lines.append("- Hispanic: ETHNIC==1 (best available from provided extract).")
    lines.append("")
    lines.append("Table 1-style standardized betas:")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics:")
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
    lines.append("\n==== Model 1 (SES) ====\n" + res1.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + res2.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("Standard errors not shown.\n")
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