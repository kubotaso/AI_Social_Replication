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

    # GSS-style missing sentinels (do NOT include 0 globally)
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid):
        s = mask_missing(series)
        return s.where(s.isin(list(valid)), np.nan)

    def pop_sd(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return a.std(ddof=0)

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

    def fit_ols_with_std_betas(data, y, xvars, model_name):
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()

        # drop any zero-variance predictors (guard)
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = pop_sd(yy.values)
        beta = {}
        for v in x_keep:
            sd_x = pop_sd(dd[v].values)
            b = res.params.get(v, np.nan)
            if (not np.isfinite(sd_x)) or (not np.isfinite(sd_y)) or sd_x == 0 or sd_y == 0:
                beta[v] = np.nan
            else:
                beta[v] = b * (sd_x / sd_y)

        coef = pd.DataFrame(
            {
                "model": model_name,
                "term": xvars,
                "included": [v in x_keep for v in xvars],
                "coef_raw": [res.params.get(v, np.nan) for v in xvars],
                "beta_std": [beta.get(v, np.nan) for v in xvars],
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
                    "const": float(res.params.get("const", np.nan)),
                    "dropped_no_variance": ", ".join(dropped) if dropped else "",
                }
            ]
        )
        return res, coef, fit, dd

    def make_table_matrix(coef_tabs, model_names, row_order):
        long = pd.concat(coef_tabs, ignore_index=True)

        def fmt(beta, sig, included):
            if (not included) or pd.isna(beta):
                return ""
            return f"{beta:.3f}{sig}"

        long["cell"] = [
            fmt(b, s, inc)
            for b, s, inc in zip(long["beta_std"], long["sig"], long["included"])
        ]
        wide = long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex([r for r in row_order if r in wide.index])
        # add constant row separately (raw, not standardized)
        return wide, long

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

    needed = (
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
        + music_items
        + tol_items
    )
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean / recode (minimal; avoid creating unnecessary NA)
    # -------------------------
    # Music: substantive 1..5
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # Core predictors (keep wide valid ranges)
    df["educ"] = mask_missing(df["educ"]).where(mask_missing(df["educ"]).between(0, 25), np.nan)
    df["realinc"] = mask_missing(df["realinc"]).where(mask_missing(df["realinc"]).between(0, 9_000_000), np.nan)
    df["hompop"] = mask_missing(df["hompop"]).where(mask_missing(df["hompop"]).between(1, 50), np.nan)
    df["prestg80"] = mask_missing(df["prestg80"]).where(mask_missing(df["prestg80"]).between(0, 100), np.nan)

    df["sex"] = coerce_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"]).where(mask_missing(df["age"]).between(18, 89), np.nan)
    df["race"] = coerce_valid(df["race"], {1, 2, 3})
    df["region"] = coerce_valid(df["region"], {1, 2, 3, 4})

    df["relig"] = coerce_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: substantive codes only (per mapping)
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk"):
            s = s.where(s.isin([1, 2]), np.nan)  # 1 allowed, 2 not allowed
        elif c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)  # 1 remove, 2 not remove
        elif c.startswith("col"):
            if c == "colcom":
                s = s.where(s.isin([4, 5]), np.nan)  # 4 fired (intolerant), 5 not fired
            else:
                s = s.where(s.isin([4, 5]), np.nan)  # 4 allowed, 5 not allowed (intolerant)
        df[c] = s

    # -------------------------
    # DV: musical exclusiveness (require complete on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["exclusiveness"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita
    d["inc_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "inc_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["inc_pc"] = d["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Hispanic (from ETHNIC when available)
    # Keep as missing if ETHNIC missing; do NOT fabricate 0/1.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # Race dummies: NOT forced to be mutually exclusive with Hispanic unless the data demands it.
    # (Paper includes Black, Hispanic, Other race; reference effectively White.)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Religion dummies
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (coarse)
    # To avoid catastrophic N loss from DENOM missingness, classify as 0 when RELIG is observed and not Protestant.
    # If RELIG==1 (Protestant) but DENOM missing, leave conserv_prot missing (cannot classify), which may drop some.
    d["conserv_prot"] = np.nan
    m_rel = d["relig"].notna()
    d.loc[m_rel & (d["relig"] != 1), "conserv_prot"] = 0.0
    m_prot_denom = d["relig"].eq(1) & d["denom"].notna()
    d.loc[m_prot_denom, "conserv_prot"] = np.where(d.loc[m_prot_denom, "denom"].isin([1, 6, 7]), 1.0, 0.0)

    # South
    d["south"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: build item indicators; then allow partial completion to reduce excessive N loss.
    # Use prorated scale to 0..15 if enough items observed (>= 12 of 15), else missing.
    # This is a pragmatic correction to avoid collapsing N far below the paper's; saves many cases with a few DKs.
    def intolerance_indicator(col, s):
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

    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    n_obs = intoler.notna().sum(axis=1)

    d["polintol"] = np.nan
    m_enough = n_obs >= 12
    # prorate to 15 items
    d.loc[m_enough, "polintol"] = (intoler.loc[m_enough].sum(axis=1) * (15.0 / n_obs.loc[m_enough])).astype(float)

    # -------------------------
    # Models (model-wise listwise deletion inside fit)
    # -------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    m1, tab1, fit1, dd1 = fit_ols_with_std_betas(d, y, x_m1, "Model 1 (SES)")
    m2, tab2, fit2, dd2 = fit_ols_with_std_betas(d, y, x_m2, "Model 2 (Demographic)")
    m3, tab3, fit3, dd3 = fit_ols_with_std_betas(d, y, x_m3, "Model 3 (Political intolerance)")

    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    table1, coef_long = make_table_matrix([tab1, tab2, tab3], model_names, x_m3)

    # Add constants row (raw intercepts) to table display
    const_row = pd.DataFrame(
        {
            "Model 1 (SES)": [f"{fit1.loc[0,'const']:.3f}" if np.isfinite(fit1.loc[0, "const"]) else ""],
            "Model 2 (Demographic)": [f"{fit2.loc[0,'const']:.3f}" if np.isfinite(fit2.loc[0, "const"]) else ""],
            "Model 3 (Political intolerance)": [f"{fit3.loc[0,'const']:.3f}" if np.isfinite(fit3.loc[0, "const"]) else ""],
        },
        index=["const (raw)"],
    )
    table1_disp = pd.concat([table1, const_row], axis=0)

    # -------------------------
    # Diagnostics
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "N_complete_music_DV": int(d.shape[0]),
                "N_complete_M1_vars": int(d[[y] + x_m1].dropna().shape[0]),
                "N_complete_M2_vars": int(d[[y] + x_m2].dropna().shape[0]),
                "N_complete_M3_vars": int(d[[y] + x_m3].dropna().shape[0]),
                "N_polintol_nonmissing(prorated>=12)": int(d["polintol"].notna().sum()),
                "N_hispanic_nonmissing": int(d["hispanic"].notna().sum()),
                "N_no_religion_nonmissing": int(d["no_religion"].notna().sum()),
                "N_conserv_prot_nonmissing": int(d["conserv_prot"].notna().sum()),
            }
        ]
    )

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append("DV: Musical exclusiveness = count (0–18) of 18 genres rated 4/5 (dislike/dislike very much).")
    lines.append("DV construction: requires complete responses on all 18 music items (DK/missing excluded).")
    lines.append("")
    lines.append("OLS models; table reports standardized betas computed as beta = b * SD(x) / SD(y) on each model's estimation sample.")
    lines.append("Stars from raw OLS two-tailed p-values: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Political intolerance scale: 15 items; intolerant responses coded per item; prorated to 0..15 when >=12 items observed.")
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Table 1-style output (standardized betas + stars; constant is raw intercept):")
    lines.append(table1_disp.to_string())
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
        f.write("Cells: standardized betas for predictors (with stars from raw OLS p-values).\n")
        f.write("Constant shown as raw intercept.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1_disp.to_string())
        f.write("\n\nFit statistics:\n")
        f.write(fitstats.to_string(index=False))
        f.write("\n\nDiagnostics:\n")
        f.write(diag.to_string(index=False))
        f.write("\n")

    # Long-form coefficient table
    coef_long_out = coef_long.loc[:, ["model", "term", "included", "coef_raw", "beta_std", "p_raw", "sig"]].copy()
    coef_long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    table1_disp.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    diag.to_csv("./output/diagnostics.tsv", sep="\t", index=False)

    return {
        "fit_stats": fitstats,
        "table1_style_betas": table1_disp,
        "coefficients_long": coef_long_out,
        "diagnostics": diag,
    }