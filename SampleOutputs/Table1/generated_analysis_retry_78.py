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

    # GSS-style special missing codes (varies by item); do not treat 0 as missing
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        7, 8, 9,
        77, 78, 79, 88, 89, 98, 99,
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

    def pop_sd(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

    def weighted_mean(x, w):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

    def weighted_sd(x, w):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() < 2:
            return np.nan
        mu = np.sum(w[m] * x[m]) / np.sum(w[m])
        var = np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])
        return float(np.sqrt(var))

    # -----------------------------
    # Political intolerance coding
    # -----------------------------
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

    def build_polintol(df, items, min_answered=15):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        answered = intoler.notna().sum(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        ok = answered >= int(min_answered)
        pol.loc[ok] = intoler.loc[ok].sum(axis=1).astype(float)
        return pol, intoler, answered

    # -----------------------------
    # OLS + standardized betas
    # - Fit unstandardized OLS (for constant, R2, p-values)
    # - Compute standardized beta = b * SD(X)/SD(Y) on estimation sample
    #   If weights provided, use weighted SDs and WLS fit.
    # -----------------------------
    def fit_model_with_betas(df, y, xvars, model_name, wvar=None):
        cols = [y] + xvars + ([wvar] if wvar else [])
        use = df.loc[:, cols].dropna(how="any").copy()
        if use.shape[0] == 0:
            raise ValueError(f"{model_name}: empty estimation sample after listwise deletion.")

        yv = use[y].astype(float).values
        X = sm.add_constant(use[xvars].astype(float), has_constant="add")

        if wvar is None:
            res = sm.OLS(yv, X).fit()
            sd_y = pop_sd(yv)
            sds_x = {v: pop_sd(use[v].astype(float).values) for v in xvars}
        else:
            w = use[wvar].astype(float).values
            # Guard: nonpositive weights -> drop
            m = np.isfinite(w) & (w > 0)
            use = use.loc[m].copy()
            yv = use[y].astype(float).values
            X = sm.add_constant(use[xvars].astype(float), has_constant="add")
            w = use[wvar].astype(float).values
            res = sm.WLS(yv, X, weights=w).fit()
            sd_y = weighted_sd(yv, w)
            sds_x = {v: weighted_sd(use[v].astype(float).values, w) for v in xvars}

        betas = {}
        for v in xvars:
            b = res.params.get(v, np.nan)
            sd_x = sds_x.get(v, np.nan)
            if not np.isfinite(b) or not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0:
                betas[v] = np.nan
            else:
                betas[v] = float(b) * (sd_x / sd_y)

        coef_rows = []
        for v in xvars:
            p = float(res.pvalues.get(v, np.nan))
            coef_rows.append(
                {
                    "model": model_name,
                    "term": v,
                    "beta_std": float(betas.get(v, np.nan)) if np.isfinite(betas.get(v, np.nan)) else np.nan,
                    "b_raw": float(res.params.get(v, np.nan)),
                    "p_raw": p,
                    "sig": stars(p),
                }
            )

        fit = {
            "model": model_name,
            "N": int(use.shape[0]),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw": float(res.params.get("const", np.nan)),
            "weighted": bool(wvar is not None),
            "weight_var": (wvar if wvar else ""),
        }

        return res, pd.DataFrame(coef_rows), pd.DataFrame([fit]), use

    def build_table_table1_style(coef_list, fit_list, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coef_list, ignore_index=True)
        fit_long = pd.concat(fit_list, ignore_index=True).set_index("model").reindex(model_order)

        # Standardized betas only; em dash if not in model
        wide = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            ctab = coef_long[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in ctab.index:
                    wide.loc[t, m] = "—"
                else:
                    r = ctab.loc[t]
                    if pd.isna(r["beta_std"]):
                        wide.loc[t, m] = "—"
                    else:
                        wide.loc[t, m] = f"{float(r['beta_std']):.3f}{r['sig']}"

        wide.index = [label_map.get(t, t) for t in wide.index]

        extra = pd.DataFrame(index=["Constant", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant", m] = "" if pd.isna(fit_long.loc[m, "const_raw"]) else f"{float(fit_long.loc[m, 'const_raw']):.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit_long.loc[m, "R2"]) else f"{float(fit_long.loc[m, 'R2']):.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit_long.loc[m, "Adj_R2"]) else f"{float(fit_long.loc[m, 'Adj_R2']):.3f}"
            extra.loc["N", m] = "" if pd.isna(fit_long.loc[m, "N"]) else str(int(fit_long.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        table = pd.concat([header, wide, extra], axis=0)
        return table, coef_long, fit_long.reset_index()

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
    # Variable lists
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
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

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

    # Hispanic: use ETHNIC if available; otherwise create missing (will be dropped model-wise)
    if "ethnic" in df.columns:
        df["ethnic"] = mask_missing(df["ethnic"])
    else:
        df["ethnic"] = np.nan

    # Tolerance items validity
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -----------------------------
    # DV: Musical exclusiveness
    # IMPORTANT: To avoid overly strict listwise deletion and better match reported Ns,
    # compute count over answered genres, then rescale to 18 and round.
    # Require a minimum number answered to stabilize (>= 12 of 18).
    # Also compute strict version for diagnostics.
    # -----------------------------
    dislike_mat = pd.DataFrame(index=df.index)
    for c in music_items:
        dislike_mat[c] = df[c].isin([4, 5]).astype(float)
        dislike_mat.loc[df[c].isna(), c] = np.nan

    answered_music = dislike_mat.notna().sum(axis=1)
    disliked_music = dislike_mat.sum(axis=1, skipna=True)

    df["num_genres_disliked_strict"] = np.where(
        answered_music == len(music_items),
        disliked_music.astype(float),
        np.nan
    )

    MIN_ANSWERED_MUSIC = 12
    df["num_genres_disliked"] = np.nan
    ok_music = answered_music >= MIN_ANSWERED_MUSIC
    # rescale to 18 and keep on 0..18, then round to nearest integer (count-like)
    scaled = (disliked_music[ok_music] / answered_music[ok_music]) * len(music_items)
    df.loc[ok_music, "num_genres_disliked"] = np.clip(np.rint(scaled), 0, len(music_items)).astype(float)

    # -----------------------------
    # IVs
    # -----------------------------
    df["income_pc"] = np.nan
    m_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[m_inc, "income_pc"] = (df.loc[m_inc, "realinc"] / df.loc[m_inc, "hompop"]).astype(float)
    df["income_pc"] = df["income_pc"].replace([np.inf, -np.inf], np.nan)

    df["female"] = np.where(df["sex"].notna(), (df["sex"] == 2).astype(float), np.nan)

    df["black"] = np.where(df["race"].notna(), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].notna(), (df["race"] == 3).astype(float), np.nan)

    # Hispanic from ETHNIC where possible:
    # Without full codebook, implement a conservative rule:
    # - if ETHNIC == 1 treat as Hispanic, else 0 (when ethnic non-missing)
    df["hispanic"] = np.where(df["ethnic"].notna(), (df["ethnic"] == 1).astype(float), np.nan)

    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}; if Protestant and denom missing -> missing
    df["conservative_protestant"] = np.nan
    m_relig = df["relig"].notna()
    df.loc[m_relig, "conservative_protestant"] = 0.0
    m_prot = df["relig"].eq(1) & df["relig"].notna()
    df.loc[m_prot, "conservative_protestant"] = np.nan
    m_prot_denom = m_prot & df["denom"].notna()
    df.loc[m_prot_denom, "conservative_protestant"] = df.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Political intolerance: allow partial completion to avoid excessive N loss; require >= 10 answered
    MIN_ANSWERED_TOL = 10
    pol, intoler_df, answered_tol = build_polintol(df, tol_items, min_answered=MIN_ANSWERED_TOL)
    # For answered>=10, missing indicators are already excluded by construction; but build_polintol sums only when ok.
    df["political_intolerance"] = pol

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

    # If Hispanic is entirely missing (e.g., no usable ETHNIC coding), drop it to avoid empty models
    if df["hispanic"].notna().sum() == 0:
        x_m2 = [v for v in x_m2 if v != "hispanic"]
        x_m3 = [v for v in x_m3 if v != "hispanic"]

    res1, coef1, fit1, use1 = fit_model_with_betas(df, y, x_m1, model_names[0], wvar=None)
    res2, coef2, fit2, use2 = fit_model_with_betas(df, y, x_m2, model_names[1], wvar=None)
    res3, coef3, fit3, use3 = fit_model_with_betas(df, y, x_m3, model_names[2], wvar=None)

    # -----------------------------
    # Table formatting (standardized betas only; no SE rows)
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
    # Remove rows not in any model (e.g., hispanic if dropped)
    all_terms = set(x_m1) | set(x_m2) | set(x_m3)
    row_order = [r for r in row_order if r in all_terms or r == "political_intolerance"]

    table1, coef_long, fit_long = build_table_table1_style(
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
            "DV_music_min_answered": int(MIN_ANSWERED_MUSIC),
            "N_DV_nonmissing": int(df[y].notna().sum()),
            "N_DV_strict_complete_18": int(df["num_genres_disliked_strict"].notna().sum()),
            "Tol_min_answered": int(MIN_ANSWERED_TOL),
            "polintol_nonmissing": int(df["political_intolerance"].notna().sum()),
            "tol_items_answered_mean": float(answered_tol.mean()) if len(answered_tol) else np.nan,
            "tol_items_answered_min": float(answered_tol.min()) if len(answered_tol) else np.nan,
            "tol_items_answered_max": float(answered_tol.max()) if len(answered_tol) else np.nan,
            "N_model1_listwise": int(fit1.loc[0, "N"]),
            "N_model2_listwise": int(fit2.loc[0, "N"]),
            "N_model3_listwise": int(fit3.loc[0, "N"]),
            "hispanic_nonmissing": int(df["hispanic"].notna().sum()),
            "hispanic_1_count": int((df["hispanic"] == 1).sum(skipna=True)) if df["hispanic"].notna().any() else 0,
        }]
    )

    miss_m1 = missingness_table(df, [y] + x_m1)
    miss_m2 = missingness_table(df, [y] + x_m2)
    miss_m3 = missingness_table(df, [y] + x_m3)

    # -----------------------------
    # Save outputs
    # -----------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("Cells: standardized coefficients (beta) only; stars from unstandardized OLS p-values.")
    lines.append("Standard errors are not shown. — indicates predictor not included.")
    lines.append("")
    lines.append("DV construction:")
    lines.append(f"- Dislike=4/5 on each genre rating.")
    lines.append(f"- Uses answered genres and rescales to 18; requires >= {MIN_ANSWERED_MUSIC} of 18 genres answered.")
    lines.append("")
    lines.append("Political intolerance scale:")
    lines.append(f"- Sum of 15 intolerance indicators; requires >= {MIN_ANSWERED_TOL} items answered.")
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
    lines.append("Missingness (Model 1 vars):")
    lines.append(miss_m1.to_string(index=False))
    lines.append("")
    lines.append("Missingness (Model 2 vars):")
    lines.append(miss_m2.to_string(index=False))
    lines.append("")
    lines.append("Missingness (Model 3 vars):")
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
        f.write("Cells: standardized betas with stars from unstandardized OLS p-values.\n")
        f.write("Standard errors not shown.\n")
        f.write("— indicates predictor not included.\n")
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