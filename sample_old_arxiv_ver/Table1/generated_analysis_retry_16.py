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

    # Conservative GSS-style missing sentinels (do NOT treat 0 as missing globally)
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid):
        s = mask_missing(series)
        return s.where(s.isin(list(valid)), np.nan)

    def pop_sd(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

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

    def zscore(series):
        s = series.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

    def intolerance_indicator(col, s):
        """
        s already numeric with missings masked.
        Returns 1=intolerant, 0=tolerant, NaN=missing.
        """
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            m = s.isin([4, 5])
            if col == "colcom":
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired
            else:
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    def build_polintol(df, tol_items, require_complete=True):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        if require_complete:
            m = intoler.notna().all(axis=1)
            pol = pd.Series(np.nan, index=df.index, dtype=float)
            pol.loc[m] = intoler.loc[m].sum(axis=1).astype(float)
            return pol, intoler.notna().sum(axis=1)
        else:
            # fallback (not used by default)
            pol = intoler.sum(axis=1, min_count=1)
            return pol.astype(float), intoler.notna().sum(axis=1)

    def fit_ols_and_betas(data, y, xvars, model_name):
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()

        X = sm.add_constant(dd[xvars].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = pop_sd(yy.values)
        betas = {}
        for v in xvars:
            sd_x = pop_sd(dd[v].values)
            b = res.params.get(v, np.nan)
            if (not np.isfinite(sd_x)) or (not np.isfinite(sd_y)) or sd_x == 0 or sd_y == 0:
                betas[v] = np.nan
            else:
                betas[v] = float(b) * (sd_x / sd_y)

        coef = pd.DataFrame(
            {
                "model": model_name,
                "term": xvars,
                "beta_std": [betas.get(v, np.nan) for v in xvars],
                "b_raw": [res.params.get(v, np.nan) for v in xvars],
                "p_raw": [res.pvalues.get(v, np.nan) for v in xvars],
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
                }
            ]
        )
        return res, coef, fit, dd

    def build_table_only_betas(coef_tabs, fit_tabs, row_order, model_order, pretty_map):
        long = pd.concat(coef_tabs, ignore_index=True)

        def cell(beta, sig):
            if pd.isna(beta):
                return ""
            return f"{beta:.3f}{sig}"

        long["cell"] = [cell(b, s) for b, s in zip(long["beta_std"], long["sig"])]
        wide = long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(row_order)
        wide = wide.reindex(columns=model_order)

        # Rename rows to match Table 1 labels
        wide.index = [pretty_map.get(i, i) for i in wide.index]

        fit = pd.concat(fit_tabs, ignore_index=True).set_index("model").reindex(model_order)

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))

        out = pd.concat([wide, extra], axis=0)
        return out, long, fit.reset_index()

    # -------------------------
    # Load + restrict to 1993
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

    needed = (
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
        + music_items
        + tol_items
    )
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music: keep 1..5 only (DK/missing -> NaN)
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # Core numeric/categorical (do not over-restrict ranges; just mask missings)
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    df["sex"] = coerce_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])

    df["race"] = coerce_valid(df["race"], {1, 2, 3})
    df["region"] = coerce_valid(df["region"], {1, 2, 3, 4})
    df["relig"] = coerce_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: mask missings; keep only valid substantive codes per item type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness (count disliked; listwise complete on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        col = f"dislike_{c}"
        d[col] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(col)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs (construct with minimal extra missingness)
    # -------------------------
    # Income per capita
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference), independent of Hispanic
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic from ETHNIC:
    # Keep as observed when ETHNIC is observed; otherwise missing (do NOT set constant 0).
    # This prevents the "dropped_no_variance" bug and matches the intended spec.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy:
    # If RELIG observed and not Protestant -> 0
    # If Protestant and DENOM observed -> 1 if denom in {1,6,7} else 0
    # If Protestant but DENOM missing -> 0 (proxy to avoid unnecessary NA loss)
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance scale (0-15): strict complete across 15 items (per summary)
    d["political_intolerance"], pol_items_answered = build_polintol(d, tol_items, require_complete=True)

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

    # Fit on raw variables; compute standardized betas from raw OLS (beta=b*sdx/sdy)
    m1, tab1, fit1, dd1 = fit_ols_and_betas(d, y, x_m1, model_names[0])
    m2, tab2, fit2, dd2 = fit_ols_and_betas(d, y, x_m2, model_names[1])
    m3, tab3, fit3, dd3 = fit_ols_and_betas(d, y, x_m3, model_names[2])

    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Diagnostics: where N is lost (per model variable set)
    # -------------------------
    def miss_counts(data, vars_):
        out = []
        for v in vars_:
            out.append({"var": v, "missing": int(data[v].isna().sum()), "nonmissing": int(data[v].notna().sum())})
        return pd.DataFrame(out).sort_values(["missing", "var"], ascending=[False, True])

    diag_overall = pd.DataFrame(
        [
            {
                "N_year_1993": int(df.shape[0]),
                "N_complete_music_18": int(d.shape[0]),
                "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
                "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
                "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
                "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
                "hispanic_1": int((d["hispanic"] == 1).sum(skipna=True)),
                "political_intolerance_nonmissing": int(d["political_intolerance"].notna().sum()),
                "polintol_items_answered_mean": float(pol_items_answered.mean()) if len(pol_items_answered) else np.nan,
                "polintol_items_answered_min": float(pol_items_answered.min()) if len(pol_items_answered) else np.nan,
                "polintol_items_answered_max": float(pol_items_answered.max()) if len(pol_items_answered) else np.nan,
            }
        ]
    )

    miss_m1 = miss_counts(d, [y] + x_m1)
    miss_m2 = miss_counts(d, [y] + x_m2)
    miss_m3 = miss_counts(d, [y] + x_m3)

    # -------------------------
    # Output: Table 1-style (standardized betas only) + fit rows
    # -------------------------
    pretty_map = {
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

    row_order = x_m3
    table_disp, coef_long, fit_for_display = build_table_only_betas(
        [tab1, tab2, tab3],
        [fit1, fit2, fit3],
        row_order=row_order,
        model_order=model_names,
        pretty_map=pretty_map,
    )

    # -------------------------
    # Save outputs (human-readable text)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append("DV: Number of music genres disliked (0–18).")
    lines.append("DV construction: across 18 genre ratings, dislike = 4 or 5; requires complete responses on all 18 items.")
    lines.append("")
    lines.append("Models: OLS. Displayed coefficients are standardized betas: beta = b * SD(x) / SD(y) computed on each model estimation sample.")
    lines.append("Stars: two-tailed p-values from raw OLS coefficients: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Table 1-style coefficients (standardized betas only) + fit rows:")
    lines.append(table_disp.to_string())
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics (overall):")
    lines.append(diag_overall.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 1 variables):")
    lines.append(miss_m1.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 2 variables):")
    lines.append(miss_m2.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 3 variables):")
    lines.append(miss_m3.to_string(index=False))
    lines.append("")
    lines.append("Raw OLS summaries:")
    lines.append("\n==== Model 1 (SES) ====\n" + m1.summary().as_text())
    lines.append("\n==== Model 2 (Demographic) ====\n" + m2.summary().as_text())
    lines.append("\n==== Model 3 (Political intolerance) ====\n" + m3.summary().as_text())

    summary_text = "\n".join(lines)
    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("Cells: standardized betas for predictors (stars from raw OLS p-values).\n")
        f.write("Additional rows: Constant (raw), R², Adj. R², N.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table_disp.to_string())
        f.write("\n")

    coef_long_out = coef_long.copy()
    coef_long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    table_disp.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    diag_overall.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)
    miss_m1.to_csv("./output/missingness_m1.tsv", sep="\t", index=False)
    miss_m2.to_csv("./output/missingness_m2.tsv", sep="\t", index=False)
    miss_m3.to_csv("./output/missingness_m3.tsv", sep="\t", index=False)

    return {
        "table1_style": table_disp,
        "fit_stats": fitstats,
        "coefficients_long": coef_long_out,
        "diagnostics_overall": diag_overall,
        "missingness_m1": miss_m1,
        "missingness_m2": miss_m2,
        "missingness_m3": miss_m3,
        "estimation_samples": {"m1": dd1, "m2": dd2, "m3": dd3},
    }