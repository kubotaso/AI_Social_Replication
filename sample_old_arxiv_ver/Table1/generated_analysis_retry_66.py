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

    # Common GSS missing sentinels (conservative; do not treat 0 as missing globally)
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        8, 9, 98, 99, 998, 999, 9998, 9999
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(s, valid_set):
        s = mask_missing(s)
        valid = set(valid_set)
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

    def wmean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

    def wsd_pop(x, w):
        # population (ddof=0) weighted SD
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() < 2:
            return np.nan
        xm = wmean(x[m], w[m])
        v = np.sum(w[m] * (x[m] - xm) ** 2) / np.sum(w[m])
        return float(np.sqrt(v))

    def _get_weight_series(df):
        # Use WTSSALL if present; otherwise unweighted (all ones).
        # (GSS typically uses WTSSALL for person-level analyses.)
        if "wtssall" in df.columns:
            w = mask_missing(df["wtssall"]).astype(float)
            w = w.where(w > 0, np.nan)
            return w
        return pd.Series(1.0, index=df.index, dtype=float)

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
        ok = intoler.notna().all(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        pol.loc[ok] = intoler.loc[ok].sum(axis=1).astype(float)
        answered = intoler.notna().sum(axis=1).astype(float)
        return pol, answered

    def fit_weighted_ols_and_standardized_betas(data, y, xvars, model_name, weight_col=None):
        cols = [y] + xvars + ([] if weight_col is None else [weight_col])
        dd = data.loc[:, cols].copy()

        # listwise deletion for model variables (weights too if provided)
        dd = dd.dropna(how="any")

        if dd.shape[0] == 0:
            miss_counts = data.loc[:, [y] + xvars].isna().sum().sort_values(ascending=False)
            raise ValueError(
                f"{model_name}: empty estimation sample after listwise deletion.\n"
                f"Missing counts among model columns:\n{miss_counts.to_string()}"
            )

        # Determine weights
        if weight_col is None:
            w = pd.Series(1.0, index=dd.index, dtype=float)
        else:
            w = dd[weight_col].astype(float)
            w = w.where(np.isfinite(w) & (w > 0), np.nan)
            dd = dd.loc[w.notna()].copy()
            w = w.loc[dd.index].astype(float)

        # Drop predictors with no variance (weighted variance ~0 or unique<=1)
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
                continue
            s = dd[v].astype(float).values
            ws = w.values
            sd = wsd_pop(s, ws)
            if (not np.isfinite(sd)) or sd == 0:
                dropped.append(v)
            else:
                x_keep.append(v)

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)

        # WLS
        res = sm.WLS(yy, X, weights=w.astype(float)).fit()

        # Standardized betas computed using weighted SDs on THIS model estimation sample
        sdy = wsd_pop(yy.values, w.values)
        beta = {}
        for v in x_keep:
            sdx = wsd_pop(dd[v].astype(float).values, w.values)
            b = res.params.get(v, np.nan)
            if not np.isfinite(sdx) or not np.isfinite(sdy) or sdx == 0 or sdy == 0:
                beta[v] = np.nan
            else:
                beta[v] = float(b) * (sdx / sdy)

        coef = pd.DataFrame(
            {
                "model": model_name,
                "term": xvars,
                "included": [v in x_keep for v in xvars],
                "beta_std": [beta.get(v, np.nan) for v in xvars],
                "b_raw": [float(res.params.get(v, np.nan)) for v in xvars],
                "p_raw": [float(res.pvalues.get(v, np.nan)) for v in xvars],
            }
        )
        coef["sig"] = coef["p_raw"].map(stars)

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

    def build_table_only_betas(coefs, fits, model_order, row_order, label_map, dv_label):
        # coefs: list of coef dfs; fits: list of fit dfs
        coef_long = pd.concat(coefs, ignore_index=True)
        fit_long = pd.concat(fits, ignore_index=True)

        wide = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            ctab = coef_long.loc[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in ctab.index:
                    wide.loc[t, m] = "—"
                    continue
                r = ctab.loc[t]
                if (not bool(r["included"])) or pd.isna(r["beta_std"]):
                    wide.loc[t, m] = "—"
                else:
                    wide.loc[t, m] = f"{float(r['beta_std']):.3f}{r['sig']}"

        wide.index = [label_map.get(t, t) for t in wide.index]

        fits_ix = fit_long.set_index("model").reindex(model_order)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fits_ix.loc[m, "const_raw"]) else f"{float(fits_ix.loc[m, 'const_raw']):.3f}"
            extra.loc["R²", m] = "" if pd.isna(fits_ix.loc[m, "R2"]) else f"{float(fits_ix.loc[m, 'R2']):.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fits_ix.loc[m, "Adj_R2"]) else f"{float(fits_ix.loc[m, 'Adj_R2']):.3f}"
            extra.loc["N", m] = "" if pd.isna(fits_ix.loc[m, "N"]) else str(int(fits_ix.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        return pd.concat([header, wide, extra], axis=0), coef_long, fit_long

    def missingness_table(data, vars_):
        rows = []
        for v in vars_:
            rows.append({"var": v, "missing": int(data[v].isna().sum()), "nonmissing": int(data[v].notna().sum())})
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

    # Hispanic/ethnicity: the file includes 'ethnic' in sample; use it if present.
    # If not present, create a missing column (will drop cases when needed).
    if "ethnic" not in df.columns:
        df["ethnic"] = np.nan

    # Weights: optional
    has_wt = "wtssall" in df.columns
    if has_wt:
        df["wtssall"] = mask_missing(df["wtssall"])

    required = (
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
        + music_items
        + tol_items
        + (["wtssall"] if has_wt else [])
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean / coerce
    # -------------------------
    # Music: only 1..5 (DK/missing -> NaN)
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # SES + demographics
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

    # Tolerance items: validate values by item family
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        else:  # col*
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count of genres disliked (strict complete-case on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # Attach weights (optional)
    if has_wt:
        d["wtssall"] = d["wtssall"].astype(float)
    wcol = "wtssall" if has_wt else None

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita: REALINC / HOMPOP (require HOMPOP>0)
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white is reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic indicator from ETHNIC (present in this file); do NOT force mutual exclusivity with race
    # Implemented as: hispanic = 1 if ETHNIC == 1 else 0 (when ETHNIC non-missing).
    # If the extract uses different coding, this is still the most faithful use of available ETHNIC.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    d["conservative_protestant"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel, "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # No religion + Southern
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance (strict complete-case across 15 items, per mapping)
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

    res1, coef1, fit1, use1 = fit_weighted_ols_and_standardized_betas(d, y, x_m1, model_names[0], weight_col=wcol)
    res2, coef2, fit2, use2 = fit_weighted_ols_and_standardized_betas(d, y, x_m2, model_names[1], weight_col=wcol)
    res3, coef3, fit3, use3 = fit_weighted_ols_and_standardized_betas(d, y, x_m3, model_names[2], weight_col=wcol)

    # -------------------------
    # Table formatting (NO SE rows; labeled; em-dash for excluded)
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

    table1, coef_long, fit_long = build_table_only_betas(
        coefs=[coef1, coef2, coef3],
        fits=[fit1, fit2, fit3],
        model_order=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label,
    )

    # -------------------------
    # Diagnostics / missingness
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "N_year_1993": int(df.shape[0]),
                "N_complete_music_18": int(d.shape[0]),
                "weights_used": bool(has_wt),
                "weight_var": "wtssall" if has_wt else "",
                "N_model1_listwise": int(fit1.loc[0, "N"]),
                "N_model2_listwise": int(fit2.loc[0, "N"]),
                "N_model3_listwise": int(fit3.loc[0, "N"]),
                "hispanic_nonmissing_in_dv_complete": int(d["hispanic"].notna().sum()),
                "hispanic_1_count_in_dv_complete": int((d["hispanic"] == 1).sum(skipna=True)),
                "polintol_nonmissing_in_dv_complete": int(d["political_intolerance"].notna().sum()),
                "polintol_answered_mean_in_dv_complete": float(np.nanmean(pol_answered.values)) if len(pol_answered) else np.nan,
                "polintol_answered_min_in_dv_complete": float(np.nanmin(pol_answered.values)) if len(pol_answered) else np.nan,
                "polintol_answered_max_in_dv_complete": float(np.nanmax(pol_answered.values)) if len(pol_answered) else np.nan,
                "note_standardization": "Standardized betas computed as beta = b * SD_w(x) / SD_w(y) using weighted (pop) SDs on each model's estimation sample (WTSSALL if available).",
                "note_stars": "Stars from two-tailed p-values of the corresponding raw coefficients in the same (WLS/OLS) model.",
            }
        ]
    )

    miss_m1 = missingness_table(d, [y] + x_m1 + ([] if wcol is None else [wcol]))
    miss_m2 = missingness_table(d, [y] + x_m2 + ([] if wcol is None else [wcol]))
    miss_m3 = missingness_table(d, [y] + x_m3 + ([] if wcol is None else [wcol]))

    # -------------------------
    # Save outputs
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS/WLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("DV construction: 18 music items; dislike = 4 or 5; strict complete-case across all 18 items.")
    lines.append("")
    if has_wt:
        lines.append("Estimation: WLS using WTSSALL (if present in file).")
    else:
        lines.append("Estimation: OLS (no weight variable found).")
    lines.append("Displayed coefficients: standardized betas only (no standard errors printed).")
    lines.append("Standardization: beta = b * SD(x)/SD(y) (weighted if WTSSALL present), computed within each model estimation sample.")
    lines.append("Stars: two-tailed p-values from raw (unstandardized) coefficients in the same model.")
    lines.append("")
    lines.append("Table 1-style standardized coefficients:")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fit_long.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Dropped for no variance (if any):")
    lines.append(fit_long.loc[:, ["model", "dropped_no_variance"]].to_string(index=False))
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
    lines.append("Raw model summaries (debug):")
    lines.append("\n==== Model 1 (SES) ====\n" + res1.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + res2.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + res3.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with significance stars.\n")
        f.write("— indicates predictor not included / not estimated.\n")
        f.write("Note: no standard errors are printed.\n\n")
        f.write(table1.to_string())
        f.write("\n")

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