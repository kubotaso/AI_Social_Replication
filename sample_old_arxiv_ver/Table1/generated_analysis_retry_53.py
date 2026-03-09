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

    # Broad set of common GSS missing codes; keep 0 as valid
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(s, valid_set):
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

    def pop_sd(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

    # Political intolerance coding per mapping
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
        # strict complete-case sum across all 15 items
        pol = intoler.sum(axis=1, skipna=False).astype(float)
        return pol, intoler

    def fit_ols_and_std_betas(data, y, xvars, model_name):
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()

        # Prevent statsmodels "zero-size array" crash
        if dd.shape[0] == 0:
            coef_rows = []
            for v in xvars:
                coef_rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": False,
                        "beta_std": np.nan,
                        "b_raw": np.nan,
                        "p_raw": np.nan,
                        "cell": "—",
                    }
                )
            coef = pd.DataFrame(coef_rows)
            fit = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "N": 0,
                        "R2": np.nan,
                        "Adj_R2": np.nan,
                        "const_raw": np.nan,
                        "dropped_no_variance": "ALL (no complete cases)",
                    }
                ]
            )
            return None, coef, fit, dd

        # Drop predictors with no variance in estimation sample
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        if len(x_keep) == 0:
            # Another safe-guard: nothing to regress on besides constant
            coef_rows = []
            for v in xvars:
                coef_rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": False,
                        "beta_std": np.nan,
                        "b_raw": np.nan,
                        "p_raw": np.nan,
                        "cell": "—",
                    }
                )
            coef = pd.DataFrame(coef_rows)
            fit = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "N": int(dd.shape[0]),
                        "R2": np.nan,
                        "Adj_R2": np.nan,
                        "const_raw": float(dd[y].mean()),
                        "dropped_no_variance": ", ".join(dropped) if dropped else "",
                    }
                ]
            )
            return None, coef, fit, dd

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = pop_sd(yy.to_numpy())
        betas = {}
        for v in x_keep:
            sd_x = pop_sd(dd[v].to_numpy(dtype=float))
            b = float(res.params.get(v, np.nan))
            if not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0:
                betas[v] = np.nan
            else:
                betas[v] = b * (sd_x / sd_y)

        rows = []
        for v in xvars:
            if v in dropped:
                rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": False,
                        "beta_std": np.nan,
                        "b_raw": np.nan,
                        "p_raw": np.nan,
                        "cell": "—",
                    }
                )
            elif v in x_keep:
                p = float(res.pvalues.get(v, np.nan))
                beta = betas.get(v, np.nan)
                cell = "—" if pd.isna(beta) else f"{beta:.3f}{stars(p)}"
                rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": True,
                        "beta_std": beta,
                        "b_raw": float(res.params.get(v, np.nan)),
                        "p_raw": p,
                        "cell": cell,
                    }
                )
            else:
                # Not expected, but keep safe
                rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": False,
                        "beta_std": np.nan,
                        "b_raw": np.nan,
                        "p_raw": np.nan,
                        "cell": "—",
                    }
                )

        coef = pd.DataFrame(rows)
        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "N": int(dd.shape[0]),
                    "R2": float(res.rsquared),
                    "Adj_R2": float(res.rsquared_adj),
                    "const_raw": float(res.params.get("const", np.nan)),
                    "dropped_no_variance": ", ".join(dropped) if dropped else "",
                }
            ]
        )
        return res, coef, fit, dd

    def build_table1_display(coef_long, fitstats, model_names, row_order, label_map, dv_label):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_names).fillna("—")
        wide.index = [label_map.get(t, t) for t in wide.index]

        fit = fitstats.set_index("model").reindex(model_names)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_names, dtype=object)
        for m in model_names:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_names}, index=[""])
        out = pd.concat([header, wide, extra], axis=0)
        return out

    def missingness_table(d, vars_):
        rows = []
        for v in vars_:
            rows.append({"var": v, "nonmissing": int(d[v].notna().sum()), "missing": int(d[v].isna().sum())})
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
    # Variable lists (per mapping)
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
    core_needed = ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ballot"]
    needed = core_needed + music_items + tol_items
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music: 1..5 only (DK/missing -> NaN)
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # SES / demographics
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

    # Tolerance items: validate and mask missing
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: number of music genres disliked (strict complete-case on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs (Table 1)
    # -------------------------
    # Household income per capita = realinc / hompop; require hompop > 0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female (1/0)
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Age already cleaned (continuous)
    # Race dummies (white reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: not available in provided variable list -> do not fabricate
    # Use an "all-missing" column but do NOT include it in estimation to avoid empty samples.
    d["hispanic"] = np.nan

    # Conservative Protestant proxy (coarse, based on RELIG + DENOM)
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

    # Political intolerance (strict complete-case across all 15 items)
    d["political_intolerance"], intoler_df = build_polintol_strict(d, tol_items)

    # -------------------------
    # Models (OLS; standardized betas)
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]

    # IMPORTANT: to avoid empty estimation sample, only include "hispanic" if it has any data
    # (in this extract it is all-missing by construction).
    x_m2_base = [
        "female", "age", "black", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m2 = x_m1 + (["hispanic"] if d["hispanic"].notna().any() else []) + x_m2_base

    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    m1, tab1, fit1, dd1 = fit_ols_and_std_betas(d, y, x_m1, model_names[0])
    m2, tab2, fit2, dd2 = fit_ols_and_std_betas(d, y, x_m2, model_names[1])
    m3, tab3, fit3, dd3 = fit_ols_and_std_betas(d, y, x_m3, model_names[2])

    coef_long = pd.concat([tab1, tab2, tab3], ignore_index=True)
    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Table 1-style output (labeled, em-dash, NO SE ROWS)
    # -------------------------
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

    # Ensure the display contains the full Table-1 row set (even if a term isn't modeled)
    # Add "placeholder" rows (cell="—") for any missing term-model combos.
    all_terms = row_order
    all_models = model_names
    existing = coef_long[["model", "term"]].drop_duplicates()
    missing_pairs = []
    for m in all_models:
        for t in all_terms:
            if not ((existing["model"] == m) & (existing["term"] == t)).any():
                missing_pairs.append((m, t))
    if missing_pairs:
        filler = pd.DataFrame(
            [{"model": m, "term": t, "included": False, "beta_std": np.nan, "b_raw": np.nan, "p_raw": np.nan, "cell": "—"}
             for (m, t) in missing_pairs]
        )
        coef_long = pd.concat([coef_long, filler], ignore_index=True)

    dv_label = "Number of music genres disliked"
    table1 = build_table1_display(coef_long, fitstats, model_names, row_order, label_map, dv_label)

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "N_year_1993": int(df.shape[0]),
                "N_complete_music_18": int(d.shape[0]),
                "N_model1_listwise": int(dd1.shape[0]),
                "N_model2_listwise": int(dd2.shape[0]),
                "N_model3_listwise": int(dd3.shape[0]),
                "income_pc_nonmissing_in_dv_complete": int(d["income_pc"].notna().sum()),
                "prestg80_nonmissing_in_dv_complete": int(d["prestg80"].notna().sum()),
                "political_intolerance_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
                "hispanic_available_in_file": int(d["hispanic"].notna().sum()) > 0,
                "dv_strict_complete_case_items": 18,
                "polintol_strict_complete_case_items": 15,
            }
        ]
    )

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs
    # -------------------------
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    summary_lines.append("")
    summary_lines.append(f"Dependent variable (DV): {dv_label}")
    summary_lines.append("DV construction: 18 music items; disliked = 4 or 5; requires complete responses on all 18 items.")
    summary_lines.append("")
    summary_lines.append("Models: OLS on original DV scale.")
    summary_lines.append("Displayed coefficients: standardized betas computed from raw-model slopes via beta = b * SD(X) / SD(Y) using population SD (ddof=0) on each model's estimation sample.")
    summary_lines.append("Standard errors are not shown in the Table 1 reproduction.")
    summary_lines.append("Stars: from raw-model two-tailed p-values (* p<.05, ** p<.01, *** p<.001).")
    summary_lines.append("")
    summary_lines.append("NOTE: Hispanic indicator is not constructed because no Hispanic/ethnicity variable is available in the provided file/variable list.")
    summary_lines.append("")
    summary_lines.append("Table 1-style standardized coefficients:")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Diagnostics:")
    summary_lines.append(diag.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Missingness (within DV-complete sample):")
    summary_lines.append("\nModel 1 vars:\n" + miss_m1.to_string(index=False))
    summary_lines.append("\nModel 2 vars:\n" + miss_m2.to_string(index=False))
    summary_lines.append("\nModel 3 vars:\n" + miss_m3.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Raw OLS summaries (debug):")
    if m1 is not None:
        summary_lines.append("\n==== Model 1 (SES) ====\n" + m1.summary().as_text())
    else:
        summary_lines.append("\n==== Model 1 (SES) ====\n(No estimable model: no complete cases.)")
    if m2 is not None:
        summary_lines.append("\n==== Model 2 (Demographic) ====\n" + m2.summary().as_text())
    else:
        summary_lines.append("\n==== Model 2 (Demographic) ====\n(No estimable model: no complete cases.)")
    if m3 is not None:
        summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + m3.summary().as_text())
    else:
        summary_lines.append("\n==== Model 3 (Political intolerance) ====\n(No estimable model: no complete cases.)")

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw-model p-values.\n")
        f.write("— indicates predictor not included or not estimable.\n")
        f.write("Standard errors are not shown.\n\n")
        f.write(table1.to_string())
        f.write("\n")

    # Machine-readable outputs
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
        "estimation_samples": {"m1": dd1, "m2": dd2, "m3": dd3},
    }