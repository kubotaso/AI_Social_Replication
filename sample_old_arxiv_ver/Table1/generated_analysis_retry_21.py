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

    # Conservative "missing" sentinels commonly used in GSS extracts.
    # IMPORTANT: do not treat 0 as missing globally.
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid_set):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_set)), np.nan)

    def zscore(series):
        s = series.astype(float)
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

    def fit_model_standardized(df, y, xvars, model_name):
        """
        Model-specific listwise deletion on y + xvars.
        Standardize y and x within estimation sample; OLS on standardized vars.
        Return:
          - statsmodels result (on standardized vars)
          - long coef table with standardized betas and stars (from standardized regression p-values)
          - fit stats table (N, R2, AdjR2, intercept from raw-scale regression)
          - estimation sample (raw scale)
        """
        cols = [y] + xvars
        d = df.loc[:, cols].dropna(how="any").copy()
        n = int(d.shape[0])
        if n == 0:
            coef_long = pd.DataFrame(
                [{"model": model_name, "term": v, "beta_std": np.nan, "p": np.nan, "cell": "—"} for v in xvars]
            )
            fit = pd.DataFrame([{"model": model_name, "N": 0, "R2": np.nan, "Adj_R2": np.nan, "const_raw": np.nan}])
            return None, coef_long, fit, d

        # Standardized regression for standardized betas
        yz = zscore(d[y])
        Xz = pd.DataFrame(index=d.index)
        keep = []
        for v in xvars:
            zv = zscore(d[v])
            if zv.notna().all():  # zscore returns all-NaN if sd==0 or sd invalid
                Xz[v] = zv
                keep.append(v)

        if len(keep) == 0:
            # Intercept-only standardized fit
            X = np.ones((len(yz), 1), dtype=float)
            res_std = sm.OLS(yz.values, X).fit()
            coef_long = pd.DataFrame(
                [{"model": model_name, "term": v, "beta_std": np.nan, "p": np.nan, "cell": "—"} for v in xvars]
            )
        else:
            X = sm.add_constant(Xz[keep].astype(float), has_constant="add")
            res_std = sm.OLS(yz.astype(float), X).fit()
            rows = []
            for v in xvars:
                if v in keep:
                    b = float(res_std.params.get(v, np.nan))
                    p = float(res_std.pvalues.get(v, np.nan))
                    cell = f"{b:.3f}{stars(p)}" if np.isfinite(b) else "—"
                    rows.append({"model": model_name, "term": v, "beta_std": b, "p": p, "cell": cell})
                else:
                    rows.append({"model": model_name, "term": v, "beta_std": np.nan, "p": np.nan, "cell": "—"})
            coef_long = pd.DataFrame(rows)

        # Raw-scale regression to get intercept comparable to typical table "Constant"
        X_raw = sm.add_constant(d[xvars].astype(float), has_constant="add")
        res_raw = sm.OLS(d[y].astype(float), X_raw).fit()

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": n,
                "R2": float(res_raw.rsquared),
                "Adj_R2": float(res_raw.rsquared_adj),
                "const_raw": float(res_raw.params.get("const", np.nan)),
            }]
        )
        return res_std if len(keep) else res_std, coef_long, fit, d

    def missingness(df, vars_):
        out = []
        for v in vars_:
            out.append({"var": v, "missing": int(df[v].isna().sum()), "nonmissing": int(df[v].notna().sum())})
        return pd.DataFrame(out).sort_values(["missing", "var"], ascending=[False, True])

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
    required = (
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ballot"]
        + music_items
        + tol_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean / coerce variables
    # -------------------------
    # Music items: keep only 1..5; DK/missing -> NaN
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # Core covariates
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

    # Tolerance items: mask missing; keep only substantive codes by item type
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Number of music genres disliked (0-18), complete across all 18 items
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP > 0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female indicator
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: not available in provided field list -> exclude from modeling by not including it at all.
    # (Including it as NaN forces Model 2/3 N to 0.)
    # We'll still show a "Hispanic" row in the output table as em-dash (—).
    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy using RELIG + DENOM (coarse mapping)
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern indicator
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across all 15 items; sum of intolerance indicators (0-15)
    intoler_mat = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    d["political_intolerance"] = intoler_mat.sum(axis=1, min_count=len(tol_items))  # NaN unless all 15 present

    # -------------------------
    # Model specs
    # -------------------------
    y = "num_genres_disliked"

    x_m1 = ["educ", "income_pc", "prestg80"]

    # IMPORTANT FIX: do NOT include "hispanic" variable since it is not constructible here;
    # including it would drop all rows.
    x_m2 = x_m1 + [
        "female", "age", "black", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]

    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    # -------------------------
    # Fit models with model-specific complete cases
    # -------------------------
    m1, c1, f1, s1 = fit_model_standardized(d, y, x_m1, model_names[0])
    m2, c2, f2, s2 = fit_model_standardized(d, y, x_m2, model_names[1])
    m3, c3, f3, s3 = fit_model_standardized(d, y, x_m3, model_names[2])

    coef_long = pd.concat([c1, c2, c3], ignore_index=True)
    fitstats = pd.concat([f1, f2, f3], ignore_index=True)

    # -------------------------
    # Build Table 1-style display (standardized betas only + constant + fit stats)
    # Include "Hispanic" row as em-dash for transparency.
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

    # Row order mirroring the paper (with Hispanic included as a placeholder row)
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
    col_order = model_names

    wide = coef_long.pivot(index="term", columns="model", values="cell")
    wide = wide.reindex(index=[r for r in row_order if r in wide.index])
    wide = wide.reindex(columns=col_order)

    # Add placeholder Hispanic row if absent
    if "hispanic" not in wide.index:
        wide.loc["hispanic", :] = "—"

    # Ensure all desired rows exist (fill missing with em-dash)
    for r in row_order:
        if r not in wide.index:
            wide.loc[r, :] = "—"
    wide = wide.reindex(row_order)
    wide = wide.fillna("—")

    # Apply pretty labels
    wide.index = [pretty.get(i, i) for i in wide.index]

    # Fit rows: Constant (raw), R2, AdjR2, N
    fit_w = fitstats.set_index("model").reindex(col_order)

    extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=col_order, dtype=object)
    for m in col_order:
        if m in fit_w.index:
            const = fit_w.loc[m, "const_raw"]
            r2 = fit_w.loc[m, "R2"]
            ar2 = fit_w.loc[m, "Adj_R2"]
            n = fit_w.loc[m, "N"]
        else:
            const = r2 = ar2 = n = np.nan

        extra.loc["Constant (raw)", m] = f"{float(const):.3f}" if pd.notna(const) else ""
        extra.loc["R²", m] = f"{float(r2):.3f}" if pd.notna(r2) else ""
        extra.loc["Adj. R²", m] = f"{float(ar2):.3f}" if pd.notna(ar2) else ""
        extra.loc["N", m] = str(int(n)) if pd.notna(n) else ""

    table1_style = pd.concat([wide, extra], axis=0)

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
            "political_intolerance_nonmissing_strict15": int(d["political_intolerance"].notna().sum()),
        }]
    )

    miss_m1 = missingness(d, [y] + x_m1)
    miss_m2 = missingness(d, [y] + x_m2)
    miss_m3 = missingness(d, [y] + x_m3)

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    header = []
    header.append("Replication output: Table 1-style OLS (1993 GSS)")
    header.append("")
    header.append("Dependent variable: Number of music genres disliked (0–18).")
    header.append("DV construction: across 18 genre ratings; disliked = 4 or 5; requires complete responses on all 18 items.")
    header.append("")
    header.append("Models: OLS; table cells are standardized coefficients (betas).")
    header.append("Standardization method: within each model estimation sample, z-score DV and all included predictors; run OLS; report slopes.")
    header.append("Stars: two-tailed p-values from that standardized regression: * p<.05, ** p<.01, *** p<.001.")
    header.append("")
    header.append("Important note: A Hispanic indicator is not constructible from the provided columns, so it is not included in estimation.")
    header.append("It is displayed as '—' for transparency.")
    header.append("")

    summary_lines = []
    summary_lines.extend(header)
    summary_lines.append("Table 1-style coefficients (standardized betas) + fit statistics:")
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
    summary_lines.append("Standardized-regression summaries (for reference; Table 1 prints betas only):")
    summary_lines.append("\n==== Model 1 (SES) ====\n" + (m1.summary().as_text() if m1 is not None else "Not estimated"))
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + (m2.summary().as_text() if m2 is not None else "Not estimated"))
    summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + (m3.summary().as_text() if m3 is not None else "Not estimated"))

    summary_text = "\n".join(summary_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("DV: Number of music genres disliked\n")
        f.write("Cells: standardized betas (z-scored within model sample) with stars from standardized-regression p-values.\n")
        f.write("— indicates not in model or not constructible.\n\n")
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