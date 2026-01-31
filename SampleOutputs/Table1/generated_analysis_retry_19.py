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

    # Treat common GSS-style sentinels as missing; do not treat 0 as missing globally.
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(s, valid):
        s = mask_missing(s)
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

    def zscore(s):
        s = s.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

    # Political intolerance item coding
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
                # fired/not fired: intolerance is "yes, fired" (=4)
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                # allowed/not allowed: intolerance is "not allowed" (=5)
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    def fit_standardized_beta_table(dd, y, xvars, model_name):
        """
        Run OLS on standardized y and standardized X (no intercept needed),
        but we keep an intercept to align with typical OLS implementation.
        For standardized variables, coefficients are standardized betas.
        Stars are computed from this standardized regression's p-values.
        """
        cols = [y] + xvars
        d = dd.loc[:, cols].dropna(how="any").copy()

        # Standardize within estimation sample
        yz = zscore(d[y])
        Xz = pd.DataFrame({v: zscore(d[v]) for v in xvars})
        # Drop any predictors that become all-missing after zscore (e.g., zero variance)
        keep = [v for v in xvars if Xz[v].notna().sum() == len(Xz)]
        dropped = [v for v in xvars if v not in keep]

        # Recompute with kept vars
        X = sm.add_constant(Xz[keep], has_constant="add")
        res = sm.OLS(yz, X).fit()

        # Build coefficient rows in the full requested order (with em-dash for not-in-model / dropped)
        rows = []
        for v in xvars:
            if v in keep:
                b = float(res.params.get(v, np.nan))
                p = float(res.pvalues.get(v, np.nan))
                cell = f"{b:.3f}{stars(p)}" if np.isfinite(b) else "—"
                rows.append({"model": model_name, "term": v, "cell": cell, "beta": b, "p": p, "included": True})
            else:
                rows.append({"model": model_name, "term": v, "cell": "—", "beta": np.nan, "p": np.nan, "included": False})

        fit = {
            "model": model_name,
            "N": int(len(d)),
            "R2": float(res.rsquared),
            "Adj_R2": float(res.rsquared_adj),
            "const_raw_on_z": float(res.params.get("const", np.nan)),
            "dropped_no_variance_or_bad_z": ", ".join(dropped) if dropped else "",
        }
        return res, pd.DataFrame(rows), pd.DataFrame([fit]), d

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
    # Music items: only 1..5 are valid; DK/missing -> NaN
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # SES covariates
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    # Demographics
    df["sex"] = coerce_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = coerce_valid(df["race"], {1, 2, 3})
    df["relig"] = coerce_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["region"] = coerce_valid(df["region"], {1, 2, 3, 4})
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: keep only substantive codes
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: Musical exclusiveness = count of genres disliked (4/5), listwise complete on all 18
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs construction
    # -------------------------
    # Income per capita: REALINC / HOMPOP; require HOMPOP > 0
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female dummy
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (White is reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: NOT AVAILABLE in provided extract -> leave as missing (excluded from models)
    # This matches the mapping instruction rather than guessing from other fields.
    d["hispanic"] = np.nan

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant (coarse implementable proxy per mapping)
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across all 15 items (paper-style count)
    intoler_mat = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    d["political_intolerance"] = intoler_mat.sum(axis=1, min_count=len(tol_items))
    # min_count=len(...) ensures NaN unless all 15 present

    # -------------------------
    # Model specifications
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
    # Fit models (model-wise listwise deletion happens inside)
    # -------------------------
    m1, t1, f1, s1 = fit_standardized_beta_table(d, y, x_m1, model_names[0])
    m2, t2, f2, s2 = fit_standardized_beta_table(d, y, x_m2, model_names[1])
    m3, t3, f3, s3 = fit_standardized_beta_table(d, y, x_m3, model_names[2])

    coef_long = pd.concat([t1, t2, t3], ignore_index=True)
    fitstats = pd.concat([f1, f2, f3], ignore_index=True)

    # -------------------------
    # Build Table 1-style display (standardized betas only; no SE rows)
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
    row_order = x_m3  # Table 1 order (with em-dash where not included)
    col_order = model_names

    wide = coef_long.pivot(index="term", columns="model", values="cell").reindex(row_order)
    wide = wide.reindex(columns=col_order)
    wide.index = [pretty.get(i, i) for i in wide.index]
    wide = wide.fillna("—")

    fit_wide = fitstats.set_index("model").reindex(col_order)
    extra = pd.DataFrame(index=["R²", "Adj. R²", "N"], columns=col_order, dtype=object)
    for m in col_order:
        extra.loc["R²", m] = f"{fit_wide.loc[m, 'R2']:.3f}" if pd.notna(fit_wide.loc[m, "R2"]) else ""
        extra.loc["Adj. R²", m] = f"{fit_wide.loc[m, 'Adj_R2']:.3f}" if pd.notna(fit_wide.loc[m, "Adj_R2"]) else ""
        extra.loc["N", m] = str(int(fit_wide.loc[m, "N"])) if pd.notna(fit_wide.loc[m, "N"]) else ""

    table1_style = pd.concat([wide, extra], axis=0)

    # -------------------------
    # Diagnostics
    # -------------------------
    diag = pd.DataFrame(
        [{
            "N_year_1993": int(df.shape[0]),
            "N_complete_music_18": int(d.shape[0]),
            "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
            "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
            "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
            "polintol_nonmissing_strict15": int(d["political_intolerance"].notna().sum()),
            "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
        }]
    )

    # -------------------------
    # Save outputs
    # -------------------------
    header_lines = [
        "Replication output: Table 1-style OLS (1993 GSS)",
        "",
        "Dependent variable: Number of music genres disliked (0–18).",
        "DV construction: across 18 genre ratings, dislike = 4 or 5; requires complete responses on all 18 items.",
        "",
        "Models: OLS; reported cells are standardized coefficients (betas).",
        "Standardization: variables (DV and predictors) are z-scored within each model estimation sample; OLS run on standardized variables.",
        "Stars: two-tailed p-values from the standardized regression: * p<.05, ** p<.01, *** p<.001",
        "",
        "Notes:",
        "- Hispanic indicator is not available in the provided variable extract; it is therefore missing and appears as — in Model 2/3.",
        "- Political intolerance scale is a strict complete-case sum across 15 tolerance items (0–15).",
        "",
        "Table 1-style coefficients (standardized betas only):",
        table1_style.to_string(),
        "",
        "Model fit statistics:",
        fitstats.to_string(index=False),
        "",
        "Diagnostics:",
        diag.to_string(index=False),
        "",
        "Raw model summaries (standardized DV and predictors):",
        "\n==== Model 1 (SES) ====\n" + m1.summary().as_text(),
        "\n==== Model 2 (Demographic) ====\n" + m2.summary().as_text(),
        "\n==== Model 3 (Political intolerance) ====\n" + m3.summary().as_text(),
    ]
    summary_text = "\n".join(header_lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("DV: Number of music genres disliked\n")
        f.write("Cells: standardized betas (z-score within model sample) with stars from standardized regression p-values.\n")
        f.write("— indicates not in model or not constructible from provided fields.\n\n")
        f.write(table1_style.to_string())
        f.write("\n")

    # machine-readable outputs too
    table1_style.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)

    return {
        "table1_style": table1_style,
        "fit_stats": fitstats,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "estimation_samples": {"m1": s1, "m2": s2, "m3": s3},
    }