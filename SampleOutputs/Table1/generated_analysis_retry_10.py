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

    # Conservative "special missing" masking: do not treat 0 as missing globally.
    # (GSS uses many 8/9/98/99/... style sentinels; exact codes vary by variable.)
    def mask_gss_missing(series):
        s = to_num(series)
        missing_codes = {8, 9, 98, 99, 998, 999, 9998, 9999}
        return s.mask(s.isin(list(missing_codes)), np.nan)

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
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.nan
        return x.std(ddof=0)

    def ensure_cols(df_, cols):
        miss = [c for c in cols if c not in df_.columns]
        if miss:
            raise ValueError(f"Missing required columns: {miss}")

    def coerce_valid(series, valid_set):
        s = mask_gss_missing(series)
        return s.where(s.isin(list(valid_set)), np.nan)

    # Standardized betas computed from unstandardized OLS b: beta = b * sd(x)/sd(y)
    # (Intercept is reported separately as unstandardized.)
    def fit_model(data, yvar, xvars, model_name):
        cols = [yvar] + xvars
        dd = data.loc[:, cols].copy().dropna(how="any")

        # Drop any zero-variance predictors (should not happen if coding is correct, but guard anyway)
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        y = dd[yvar].astype(float)
        res = sm.OLS(y, X).fit()

        sd_y = pop_sd(y.values)
        beta_map = {}
        for v in x_keep:
            sd_x = pop_sd(dd[v].values)
            b = res.params.get(v, np.nan)
            if (not np.isfinite(sd_x)) or (not np.isfinite(sd_y)) or sd_x == 0 or sd_y == 0:
                beta_map[v] = np.nan
            else:
                beta_map[v] = b * (sd_x / sd_y)

        coef_tab = pd.DataFrame(
            {
                "term": xvars,
                "included": [v in x_keep for v in xvars],
                "beta_std": [beta_map.get(v, np.nan) for v in xvars],
                "p_raw": [res.pvalues.get(v, np.nan) for v in xvars],
            }
        )
        coef_tab["sig"] = coef_tab["p_raw"].map(stars)

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
        return res, coef_tab, fit, dd

    def make_table1_matrix(coef_tabs, model_names, row_order):
        blocks = []
        for ct, nm in zip(coef_tabs, model_names):
            t = ct.copy()
            t["model"] = nm
            blocks.append(t)
        long = pd.concat(blocks, ignore_index=True)

        def fmt(beta, sig, included):
            if (not included) or pd.isna(beta):
                return ""
            return f"{beta:.3f}{sig}"

        long["cell"] = [fmt(b, s, inc) for b, s, inc in zip(long["beta_std"], long["sig"], long["included"])]
        wide = long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex([r for r in row_order if r in wide.index])
        return wide, long

    # -------------------------
    # Load + restrict to 1993
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    ensure_cols(df, ["year"])
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

    base_cols = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "ethnic", "relig", "denom", "region"
    ]

    ensure_cols(df, music_items + tol_items + base_cols)

    # -------------------------
    # Clean / recode (minimal; avoid unintended NA creation)
    # -------------------------
    # Music: valid 1..5 only
    for c in music_items:
        df[c] = coerce_valid(df[c], {1, 2, 3, 4, 5})

    # Core continuous
    df["educ"] = mask_gss_missing(df["educ"]).where(mask_gss_missing(df["educ"]).between(0, 25), np.nan)
    df["realinc"] = mask_gss_missing(df["realinc"]).where(mask_gss_missing(df["realinc"]).between(0, 9_000_000), np.nan)
    df["hompop"] = mask_gss_missing(df["hompop"]).where(mask_gss_missing(df["hompop"]).between(1, 50), np.nan)
    df["prestg80"] = mask_gss_missing(df["prestg80"]).where(mask_gss_missing(df["prestg80"]).between(0, 100), np.nan)

    # Demographics
    df["sex"] = coerce_valid(df["sex"], {1, 2})
    df["age"] = mask_gss_missing(df["age"]).where(mask_gss_missing(df["age"]).between(18, 89), np.nan)
    df["race"] = coerce_valid(df["race"], {1, 2, 3})
    df["region"] = coerce_valid(df["region"], {1, 2, 3, 4})
    df["relig"] = coerce_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_gss_missing(df["denom"])
    df["ethnic"] = mask_gss_missing(df["ethnic"])

    # Tolerance items: substantive codes only
    for c in tol_items:
        s = mask_gss_missing(df[c])
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
    # DV: Musical exclusiveness = count of 18 genres disliked (4/5), requiring complete on all 18
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

    # Race + Hispanic: make mutually exclusive dummies so "Other race" doesn't get sign flips due to overlap.
    # Reference group: White non-Hispanic.
    # Hispanic: in many GSS codings ETHNIC==1 is a Hispanic-origin category (as present in this extract).
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    d["black"] = np.nan
    d["other_race"] = np.nan
    m_race = d["race"].notna() & d["hispanic"].notna()
    # Mutually exclusive: if hispanic==1, set black/other_race to 0 (regardless of race code).
    d.loc[m_race, "black"] = np.where((d.loc[m_race, "hispanic"] == 0) & (d.loc[m_race, "race"] == 2), 1.0, 0.0)
    d.loc[m_race, "other_race"] = np.where((d.loc[m_race, "hispanic"] == 0) & (d.loc[m_race, "race"] == 3), 1.0, 0.0)

    # Religion dummies
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (coarse DENOM recode): RELIG==1 and DENOM in {1,6,7}
    # Require denom non-missing for correct classification; otherwise missing (avoid silently coding as 0).
    d["conserv_prot"] = np.nan
    m_cp = d["relig"].notna() & d["denom"].notna()
    d.loc[m_cp, "conserv_prot"] = np.where((d.loc[m_cp, "relig"] == 1) & (d.loc[m_cp, "denom"].isin([1, 6, 7])), 1.0, 0.0)

    # South
    d["south"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict 15-item count (0-15), complete on all 15
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
    d["polintol"] = np.nan
    m_tol_complete = intoler.notna().all(axis=1)
    d.loc[m_tol_complete, "polintol"] = intoler.loc[m_tol_complete].sum(axis=1).astype(float)

    # -------------------------
    # Models (model-wise listwise deletion happens inside fit_model)
    # -------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    m1, tab1, fit1, dd1 = fit_model(d, y, x_m1, "Model 1 (SES)")
    m2, tab2, fit2, dd2 = fit_model(d, y, x_m2, "Model 2 (Demographic)")
    m3, tab3, fit3, dd3 = fit_model(d, y, x_m3, "Model 3 (Political intolerance)")

    fitstats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # Table 1-style matrix
    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    table1, coef_long = make_table1_matrix([tab1, tab2, tab3], model_names, x_m3)

    # -------------------------
    # Diagnostics to help reconcile N differences
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "base_N_complete_music_DV": int(d.shape[0]),
                "N_complete_M1_vars": int(d[[y] + x_m1].dropna().shape[0]),
                "N_complete_M2_vars": int(d[[y] + x_m2].dropna().shape[0]),
                "N_complete_M3_vars": int(d[[y] + x_m3].dropna().shape[0]),
                "N_polintol_complete15": int(m_tol_complete.sum()),
                "N_hispanic_nonmissing": int(d["hispanic"].notna().sum()),
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
    lines.append("OLS models; table reports standardized betas computed as beta = b * SD(x) / SD(y) on each model's estimation sample.")
    lines.append("Stars from raw OLS two-tailed p-values: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Diagnostics (to reconcile sample sizes):")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Table 1-style standardized coefficients (betas) + stars:")
    lines.append(table1.to_string())
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
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n\nFit statistics:\n")
        f.write(fitstats.to_string(index=False))
        f.write("\n\nDiagnostics:\n")
        f.write(diag.to_string(index=False))
        f.write("\n")

    # Long-form coefficient table
    coef_long_out = coef_long.loc[:, ["model", "term", "included", "beta_std", "p_raw", "sig"]].copy()
    coef_long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    table1.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    diag.to_csv("./output/diagnostics.tsv", sep="\t", index=False)

    return {
        "fit_stats": fitstats,
        "table1_style_betas": table1,
        "coefficients_long": coef_long_out,
        "diagnostics": diag,
    }