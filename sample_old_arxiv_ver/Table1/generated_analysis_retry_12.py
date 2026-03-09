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

    # Conservative set of common GSS missing codes (do NOT treat 0 as missing globally)
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def coerce_valid(series, valid_set):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_set)), np.nan)

    def pop_sd(x):
        a = np.asarray(x, dtype=float)
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

    def standardized_betas_from_fit(res, dd, y, x_keep):
        sd_y = pop_sd(dd[y].values)
        betas = {}
        for v in x_keep:
            sd_x = pop_sd(dd[v].values)
            b = res.params.get(v, np.nan)
            if (not np.isfinite(sd_x)) or (not np.isfinite(sd_y)) or sd_x == 0 or sd_y == 0:
                betas[v] = np.nan
            else:
                betas[v] = float(b) * (sd_x / sd_y)
        return betas

    def fit_ols_model(data, y, xvars, model_name):
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()

        # Guard: drop any zero-variance predictors (but keep them in output table as blank)
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        betas = standardized_betas_from_fit(res, dd, y, x_keep)

        coef = pd.DataFrame(
            {
                "model": model_name,
                "term": xvars,
                "included": [v in x_keep for v in xvars],
                "coef_raw": [res.params.get(v, np.nan) for v in xvars],
                "beta_std": [betas.get(v, np.nan) for v in xvars],
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

    def format_table(coef_tabs, fit_tabs, row_order, model_order):
        long = pd.concat(coef_tabs, ignore_index=True)

        def cell(beta, sig, included):
            if (not included) or pd.isna(beta):
                return ""
            return f"{beta:.3f}{sig}"

        long["cell"] = [
            cell(b, s, inc)
            for b, s, inc in zip(long["beta_std"], long["sig"], long["included"])
        ]
        wide = long.pivot(index="term", columns="model", values="cell")

        # Enforce row and column order
        wide = wide.reindex([r for r in row_order if r in wide.index])
        wide = wide.reindex(columns=model_order)

        # Add constant and fit lines (as in typical Table 1 presentation)
        fit = pd.concat(fit_tabs, ignore_index=True).set_index("model").reindex(model_order)

        extra = pd.DataFrame(index=["Constant (raw)", "R2", "Adj R2", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const"]) else f"{fit.loc[m, 'const']:.3f}"
            extra.loc["R2", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj R2", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
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

    base_needed = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "ethnic", "relig", "denom", "region",
        "ballot"
    ]
    needed = base_needed + music_items + tol_items
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean core fields
    # -------------------------
    # Music: substantive 1..5
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # Continuous-ish: keep broad plausible ranges; only mask explicit missing codes
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

    # Hispanic (ETHNIC). Do not guess; if ETHNIC missing, keep missing.
    # In many GSS extracts, ETHNIC==1 corresponds to "Hispanic"; we implement that as instructed by feedback.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # Race dummies (White reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (coarse, but avoid collapsing N):
    # - If RELIG observed and not Protestant => 0
    # - If RELIG==1 and DENOM observed => 1 for denom in {1,6,7} else 0
    # - If RELIG==1 and DENOM missing => set 0 (keep sample; coarse proxy)
    d["conserv_prot"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1)
    d.loc[m_prot & d["denom"].notna(), "conserv_prot"] = np.where(
        d.loc[m_prot & d["denom"].notna(), "denom"].isin([1, 6, 7]),
        1.0,
        0.0
    )
    # If Protestant but denom missing, keep at 0.0 (already set)

    # South
    d["south"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance scale:
    # Build 15 binary intolerance indicators; then require complete on all 15 (per mapping instruction).
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

    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    m_tol_complete = intoler.notna().all(axis=1)
    d["polintol"] = np.nan
    d.loc[m_tol_complete, "polintol"] = intoler.loc[m_tol_complete].sum(axis=1).astype(float)

    # -------------------------
    # Models (model-wise listwise deletion happens in fit)
    # -------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    m1, tab1, fit1, dd1 = fit_ols_model(d, y, x_m1, model_names[0])
    m2, tab2, fit2, dd2 = fit_ols_model(d, y, x_m2, model_names[1])
    m3, tab3, fit3, dd3 = fit_ols_model(d, y, x_m3, model_names[2])

    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Human-readable Table 1 output (properly labeled; one row per predictor)
    # -------------------------
    row_order = x_m3  # predictors only; constant/fit lines added in formatter
    table1_disp, coef_long, fit_for_display = format_table(
        [tab1, tab2, tab3],
        [fit1, fit2, fit3],
        row_order=row_order,
        model_order=model_names,
    )

    # More readable labels for saving (keep internal names in returned objects)
    pretty_map = {
        "educ": "Education (years)",
        "inc_pc": "Household income per capita",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "conserv_prot": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
        "polintol": "Political intolerance",
    }
    table1_pretty = table1_disp.copy()
    table1_pretty.index = [pretty_map.get(i, i) for i in table1_pretty.index]

    # -------------------------
    # Diagnostics (to catch the N collapses precisely)
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "N_year_1993": int(df.shape[0]),
                "N_complete_music_items": int(d.shape[0]),
                "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
                "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
                "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
                "N_hispanic_nonmissing": int(d["hispanic"].notna().sum()),
                "N_polintol_complete15": int(d["polintol"].notna().sum()),
                "N_conserv_prot_nonmissing": int(d["conserv_prot"].notna().sum()),
            }
        ]
    )

    # -------------------------
    # Save outputs
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append("DV: Musical exclusiveness = count (0–18) of 18 genres rated 4/5 (dislike/dislike very much).")
    lines.append("DV construction: requires complete responses on all 18 music items (DK/missing excluded).")
    lines.append("")
    lines.append("OLS models; cells are standardized betas computed as beta = b * SD(x)/SD(y) on each model's estimation sample.")
    lines.append("Stars are from raw OLS two-tailed p-values: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Political intolerance: sum of 15 intolerance indicators (complete on all 15 items).")
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Table 1-style coefficients (standardized betas; plus Constant/R2/AdjR2/N lines):")
    lines.append(table1_pretty.to_string())
    lines.append("")
    lines.append("Fit statistics (machine-readable):")
    lines.append(fitstats.to_string(index=False))
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
        f.write("Additional lines: Constant (raw), R2, Adj R2, N.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1_pretty.to_string())
        f.write("\n")

    coef_long_out = coef_long.loc[:, ["model", "term", "included", "coef_raw", "beta_std", "p_raw", "sig"]].copy()
    coef_long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    table1_pretty.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    diag.to_csv("./output/diagnostics.tsv", sep="\t", index=False)

    return {
        "table1_style": table1_pretty,
        "fit_stats": fitstats,
        "coefficients_long": coef_long_out,
        "diagnostics": diag,
        "estimation_samples": {
            "m1": dd1,
            "m2": dd2,
            "m3": dd3,
        },
    }