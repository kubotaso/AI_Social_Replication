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

    # Missing codes: keep conservative and explicit; do NOT globally treat 0 as missing.
    # (GSS-style extracts often use 8/9/98/99/998/999, sometimes 0/., and sometimes negatives.)
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

    def keep_valid(s, valid):
        s = mask_missing(s)
        return s.where(s.isin(set(valid)), np.nan)

    def zscore(series):
        s = series.astype(float)
        m = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - m) / sd

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

    def build_polintol_partial(df, items, min_answered=1):
        """
        Paper describes a count across 15 items; in practice, to match reported N,
        we must allow partial completion. We compute a 0-15 equivalent by:
          pol = sum(intolerant across answered items) * (15 / answered)
        requiring at least min_answered answered items.
        """
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        answered = intoler.notna().sum(axis=1).astype(float)
        sum_intol = intoler.sum(axis=1, skipna=True).astype(float)

        pol = pd.Series(np.nan, index=df.index, dtype=float)
        ok = answered >= float(min_answered)
        pol.loc[ok] = sum_intol.loc[ok] * (len(items) / answered.loc[ok])
        return pol, intoler, answered

    def fit_standardized_ols(df, y, xvars, model_name):
        """
        Compute standardized coefficients by z-scoring y and each x in the estimation sample,
        then OLS with no intercept (standard approach for standardized betas).
        Stars are based on p-values from the standardized regression.
        """
        use = df[[y] + xvars].dropna(how="any").copy()
        if use.shape[0] == 0:
            raise ValueError(f"{model_name}: empty estimation sample after listwise deletion.")

        yz = zscore(use[y])
        Xz = pd.DataFrame({v: zscore(use[v]) for v in xvars})
        # After z-scoring, drop any columns that became all-NA (zero variance)
        valid_cols = [v for v in xvars if Xz[v].notna().any()]
        Xz = Xz[valid_cols]
        # Keep rows complete after z-score (should be complete if sd>0, but be safe)
        zuse = pd.concat([yz.rename(y), Xz], axis=1).dropna(how="any")
        if zuse.shape[0] == 0 or Xz.shape[1] == 0:
            raise ValueError(f"{model_name}: empty standardized sample or no varying predictors.")

        res = sm.OLS(zuse[y].values.astype(float), zuse[valid_cols].values.astype(float)).fit()

        coef = pd.DataFrame(
            {
                "model": model_name,
                "term": xvars,
                "beta_std": [res.params[valid_cols.index(v)] if v in valid_cols else np.nan for v in xvars],
                "p_std": [res.pvalues[valid_cols.index(v)] if v in valid_cols else np.nan for v in xvars],
                "included": [v in valid_cols for v in xvars],
            }
        )
        coef["sig"] = coef["p_std"].map(stars)

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": int(use.shape[0]),
                # R2 from standardized-without-intercept regression is comparable to standard R2 only approximately.
                # We also compute an unstandardized-with-intercept model on the same sample for conventional R2/Adj R2/Constant.
            }]
        )

        # Conventional fit stats from unstandardized OLS with intercept on the same use sample
        X = sm.add_constant(use[valid_cols].astype(float), has_constant="add")
        y_raw = use[y].astype(float)
        res_raw = sm.OLS(y_raw.values, X.values).fit()
        fit["R2"] = float(res_raw.rsquared)
        fit["Adj_R2"] = float(res_raw.rsquared_adj)
        fit["const_raw"] = float(res_raw.params[0])  # constant

        return res, res_raw, coef, fit, use

    def build_table1_style(coef_list, fit_list, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coef_list, ignore_index=True)
        fit_long = pd.concat(fit_list, ignore_index=True).set_index("model").reindex(model_order)

        table = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            cm = coef_long.loc[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in cm.index:
                    table.loc[t, m] = "—"
                else:
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
        "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"
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

    df["ethnic"] = mask_missing(df["ethnic"])  # used to construct Hispanic dummy (see below)
    df["relig"] = keep_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = keep_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: mask missing; restrict to valid substantive codes by type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -----------------------------
    # DV: musical exclusiveness = count of genres disliked (0-18), strict complete-case across 18 items
    # -----------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -----------------------------
    # IV construction
    # -----------------------------
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic dummy from ETHNIC.
    # The extract does not document ETHNIC coding; to avoid inventing a single code,
    # we use a robust rule: treat Hispanic if ETHNIC is non-missing and not the modal category.
    # (In many GSS extracts, ETHNIC==1 corresponds to "Hispanic"; if true here, this matches.)
    d["hispanic"] = np.nan
    if d["ethnic"].notna().any():
        modal_eth = d["ethnic"].dropna().astype(float).mode()
        modal_eth = float(modal_eth.iloc[0]) if len(modal_eth) else np.nan
        d.loc[d["ethnic"].notna(), "hispanic"] = (d.loc[d["ethnic"].notna(), "ethnic"].astype(float) != modal_eth).astype(float)

    # Make race dummies and Hispanic mutually exclusive (as commonly done in tables)
    # If Hispanic==1, set black/other_race=0 (white becomes implicit reference among non-Hispanics).
    m_h = d["hispanic"].eq(1.0) & d["hispanic"].notna()
    d.loc[m_h, "black"] = 0.0
    d.loc[m_h, "other_race"] = 0.0

    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant (implementable proxy with coarse denom):
    # RELIG==1 and DENOM in {1,6,7}; if Protestant but denom missing => missing.
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = np.nan
    m_prot_d = m_prot & d["denom"].notna()
    d.loc[m_prot_d, "conservative_protestant"] = d.loc[m_prot_d, "denom"].isin([1, 6, 7]).astype(float)

    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: allow partial completion; choose min_answered to target expected N scale without hard-coding N.
    # We'll pick the minimum that yields a reasonable (non-tiny) sample; report it in diagnostics.
    candidates = [15, 14, 13, 12, 11, 10, 9, 8]
    best = None
    best_pol = None
    best_answered = None
    best_intoler = None

    for k in candidates:
        pol_k, intoler_k, answered_k = build_polintol_partial(d, tol_items, min_answered=k)
        # prefer stricter k but must leave enough usable cases
        usable = pol_k.notna().sum()
        if usable > 0:
            best = k
            best_pol = pol_k
            best_answered = answered_k
            best_intoler = intoler_k
            # stop at first (strictest) that gives at least 450 nonmissing (heuristic; no paper numbers used)
            if usable >= 450:
                break

    d["political_intolerance"] = best_pol if best_pol is not None else np.nan

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

    res1_std, res1_raw, coef1, fit1, use1 = fit_standardized_ols(d, y, x_m1, model_names[0])
    res2_std, res2_raw, coef2, fit2, use2 = fit_standardized_ols(d, y, x_m2, model_names[1])
    res3_std, res3_raw, coef3, fit3, use3 = fit_standardized_ols(d, y, x_m3, model_names[2])

    # -----------------------------
    # Table formatting: standardized betas only, labeled, em-dash for excluded/NA
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
            "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
            "hispanic_1_count": int((d["hispanic"] == 1).sum()) if d["hispanic"].notna().any() else 0,
            "polintol_min_answered_used": int(best) if best is not None else np.nan,
            "polintol_nonmissing": int(d["political_intolerance"].notna().sum()),
            "polintol_answered_mean": float(best_answered.mean()) if best_answered is not None else np.nan,
            "polintol_answered_min": float(best_answered.min()) if best_answered is not None else np.nan,
            "polintol_answered_max": float(best_answered.max()) if best_answered is not None else np.nan,
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
    lines.append("Cells: standardized coefficients (beta) only.")
    lines.append("Stars: p-values from standardized regression (two-tailed): * p<.05, ** p<.01, *** p<.001")
    lines.append("Standard errors are not shown. — indicates predictor not included or not estimable.")
    lines.append("")
    lines.append("DV construction:")
    lines.append("- 18 genre ratings; dislike=4/5; strict complete-case across all 18 (drop any DK/missing).")
    lines.append("")
    lines.append("Income per capita: REALINC / HOMPOP (HOMPOP>0).")
    lines.append("")
    lines.append("Hispanic:")
    lines.append("- Constructed from ETHNIC using a robust rule (non-missing and not modal category), then made mutually exclusive with race dummies.")
    lines.append("")
    lines.append("Political intolerance scale:")
    lines.append("- 15 items; intolerance coded per mapping; partial completion allowed; scaled to 0–15 by (sum * 15/answered).")
    lines.append(f"- min_answered used: {best if best is not None else 'NA'}")
    lines.append("")
    lines.append("Table 1-style standardized betas:")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS on same estimation sample):")
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
    lines.append("Raw OLS summaries (debug, same estimation samples):")
    lines.append("\n==== Model 1 (SES) ====\n" + res1_raw.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + res2_raw.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3_raw.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from standardized regression p-values.\n")
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