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

    # Try to be conservative but realistic for GSS-style missing codes.
    # (We avoid treating 0 as missing globally.)
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

    def build_table(coef_tabs, fit_tabs, row_order, model_order):
        long = pd.concat(coef_tabs, ignore_index=True)

        def cell(beta, sig):
            if pd.isna(beta):
                return ""
            return f"{beta:.3f}{sig}"

        long["cell"] = [cell(b, s) for b, s in zip(long["beta_std"], long["sig"])]
        wide = long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(row_order)
        wide = wide.reindex(columns=model_order)

        fit = pd.concat(fit_tabs, ignore_index=True).set_index("model").reindex(model_order)

        extra = pd.DataFrame(index=["Constant (raw)", "R2", "Adj R2", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
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

    # Core numeric/categorical
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

    # Tolerance items: code using the mapping; for COLCOM allow 4/5; for other COL* allow 4/5.
    # Missing codes already masked; keep only valid substantive codes.
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
    # DV: musical exclusiveness (count disliked; listwise complete on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Hispanic: use ETHNIC if available.
    # IMPORTANT FIX: do not force mutual exclusivity that creates missingness.
    # We code Hispanic as 1 if ETHNIC==1, 0 if ETHNIC is a known non-missing other value.
    # This preserves variation and avoids collapsing N.
    d["hispanic"] = np.nan
    m_eth = d["ethnic"].notna()
    d.loc[m_eth, "hispanic"] = (d.loc[m_eth, "ethnic"] == 1).astype(float)

    # Race dummies (white reference), independent of Hispanic (as in many regression specifications).
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: minimize missingness by setting to 0 when RELIG observed and not Protestant.
    # If Protestant and DENOM missing, set to 0 (proxy approach to avoid unnecessary listwise deletion).
    d["conservative_protestant"] = np.nan
    m_relig_obs = d["relig"].notna()
    d.loc[m_relig_obs & (d["relig"] != 1), "conservative_protestant"] = 0.0
    m_prot = d["relig"].eq(1) & m_relig_obs
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: 15-item count.
    # IMPORTANT FIX: do not require all 15 items. Use count across available items, but require a minimum coverage.
    # This aligns with typical index construction when split ballots/partial missingness exist and helps recover N.
    def intolerance_indicator(col, s):
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            # Special-case COLCOM per mapping: 4=fired (intolerant), 5=not fired (tolerant)
            m = s.isin([4, 5])
            if col == "colcom":
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    intoler = pd.DataFrame({c: intolerance_indicator(c, d[c]) for c in tol_items})
    intoler_nonmissing = intoler.notna().sum(axis=1)

    # Require at least 12/15 answered to limit noise but avoid excessive N loss
    d["political_intolerance"] = np.nan
    m_ok = intoler_nonmissing >= 12
    d.loc[m_ok, "political_intolerance"] = intoler.loc[m_ok].sum(axis=1).astype(float)

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

    m1, tab1, fit1, dd1 = fit_ols_and_betas(d, y, x_m1, model_names[0])
    m2, tab2, fit2, dd2 = fit_ols_and_betas(d, y, x_m2, model_names[1])
    m3, tab3, fit3, dd3 = fit_ols_and_betas(d, y, x_m3, model_names[2])

    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # -------------------------
    # Output: Table 1-style (standardized betas only) + fit rows
    # -------------------------
    row_order = x_m3
    table_disp, coef_long, fit_for_display = build_table(
        [tab1, tab2, tab3],
        [fit1, fit2, fit3],
        row_order=row_order,
        model_order=model_names,
    )

    pretty_map = {
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic (ETHNIC==1)",
        "other_race": "Other race",
        "conservative_protestant": "Conservative Protestant (proxy)",
        "no_religion": "No religion",
        "southern": "Southern",
        "political_intolerance": "Political intolerance (count; >=12 items)",
        "Constant (raw)": "Constant (raw)",
        "R2": "R²",
        "Adj R2": "Adj. R²",
        "N": "N",
    }
    table_pretty = table_disp.copy()
    table_pretty.index = [pretty_map.get(i, i) for i in table_pretty.index]

    # -------------------------
    # Diagnostics: where N is lost
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "N_year_1993": int(df.shape[0]),
                "N_complete_music_18": int(d.shape[0]),
                "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
                "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
                "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
                "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
                "hispanic_1": int((d["hispanic"] == 1).sum(skipna=True)),
                "polintol_nonmissing": int(d["political_intolerance"].notna().sum()),
                "polintol_items_answered_mean": float(intoler_nonmissing.mean()) if len(intoler_nonmissing) else np.nan,
            }
        ]
    )

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append("DV: Musical exclusiveness = number of music genres disliked (0–18).")
    lines.append("DV construction: count of the 18 genre ratings where response is 4/5 (dislike/dislike very much).")
    lines.append("DV inclusion: requires complete non-missing responses on all 18 music items.")
    lines.append("")
    lines.append("Models: OLS on raw variables; table cells show standardized betas computed on each model estimation sample.")
    lines.append("Stars: two-tailed p-values from raw OLS coefficients: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Key implementation notes:")
    lines.append("- Hispanic is coded from ETHNIC==1 (0/1) when ETHNIC is observed; otherwise missing.")
    lines.append("- Race dummies (Black/Other) use RACE and are not forced to be mutually exclusive with Hispanic.")
    lines.append("- Political intolerance is a count across the 15 tolerance items; requires >=12 answered items.")
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Table 1-style coefficients (standardized betas) + fit rows:")
    lines.append(table_pretty.to_string())
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
        f.write(table_pretty.to_string())
        f.write("\n")

    coef_long_out = coef_long.copy()
    coef_long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    table_pretty.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    diag.to_csv("./output/diagnostics.tsv", sep="\t", index=False)

    return {
        "table1_style": table_pretty,
        "fit_stats": fitstats,
        "coefficients_long": coef_long_out,
        "diagnostics": diag,
        "estimation_samples": {"m1": dd1, "m2": dd2, "m3": dd3},
    }