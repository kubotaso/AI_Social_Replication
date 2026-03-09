def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def gss_na_to_nan(s: pd.Series) -> pd.Series:
        """
        Best-effort GSS missing recode:
        Converts common special codes (>=90, 98, 99, 998, 999, etc.) to NaN.
        Keeps ordinary low-range categorical codes (1..k) intact.
        """
        s = to_num(s)
        if s.isna().all():
            return s
        # Common GSS DK/NA/refused/inap patterns:
        # - Many variables use 8/9 or 98/99, and some use 0/8/9
        # - Income-like often uses 9999998/9999999, etc. Here we only have REALINC numeric.
        # Apply conservative rules:
        #   - set values in {8,9,98,99,998,999,9998,9999} to NaN
        #   - set any value >= 90 to NaN *only* if the series appears categorical (few unique values)
        special = {8, 9, 98, 99, 998, 999, 9998, 9999}
        out = s.copy()
        out = out.mask(out.isin(list(special)), np.nan)

        nunq = out.nunique(dropna=True)
        # If it looks categorical, also mask >= 90
        if nunq > 0 and nunq <= 20:
            out = out.mask(out >= 90, np.nan)
        return out

    def pop_sd(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.nan
        return x.std(ddof=0)

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

    def assert_variation(d, cols, context=""):
        bad = []
        for c in cols:
            if c not in d.columns:
                bad.append((c, "missing_column"))
                continue
            nunq = d[c].nunique(dropna=True)
            if nunq <= 1:
                bad.append((c, f"no_variation(n_unique={nunq})"))
        if bad:
            msg = "\n".join([f"  - {c}: {why}" for c, why in bad])
            raise ValueError(f"Predictor variation check failed {context}:\n{msg}")

    # -----------------------------
    # 1) Restrict to 1993 only
    # -----------------------------
    if "year" not in df.columns:
        raise ValueError("Required column 'year' not found.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # 2) Recode missingness for variables we use
    # -----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    ses_items = ["educ", "realinc", "hompop", "prestg80"]
    demo_items = ["sex", "age", "race", "ethnic", "relig", "denom", "region"]
    tol_items = [
        "spkath", "colath", "libath",
        "spkrac", "colrac", "librac",
        "spkcom", "colcom", "libcom",
        "spkmil", "colmil", "libmil",
        "spkhomo", "colhomo", "libhomo"
    ]

    required = set(music_items + ses_items + demo_items + tol_items + ["ballot"])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply NA recoding to all used columns except realinc (numeric) where only extreme sentinels are plausible
    for c in music_items + ["educ", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"] + tol_items:
        df[c] = gss_na_to_nan(df[c])

    # REALINC: treat nonpositive or absurd sentinels as missing, but keep large valid incomes
    df["realinc"] = to_num(df["realinc"])
    df.loc[df["realinc"].isin([0, 8, 9, 98, 99, 998, 999, 9998, 9999]), "realinc"] = np.nan
    df.loc[df["realinc"] < 0, "realinc"] = np.nan
    # If a dataset uses huge sentinels (>= 9e6), mark missing
    df.loc[df["realinc"] >= 9_000_000, "realinc"] = np.nan

    # -----------------------------
    # 3) DV: Musical exclusiveness
    #    - dislike if 4/5; require complete non-missing on all 18 items
    # -----------------------------
    # Ensure only valid response range 1..5 are treated as substantive; others -> NaN
    for c in music_items:
        df.loc[~df[c].isin([1, 2, 3, 4, 5]) & df[c].notna(), c] = np.nan

    d0 = df.dropna(subset=music_items).copy()

    for c in music_items:
        d0[f"dislike_{c}"] = d0[c].isin([4, 5]).astype(int)

    d0["exclusiveness"] = d0[[f"dislike_{c}" for c in music_items]].sum(axis=1).astype(float)

    # -----------------------------
    # 4) IV construction
    # -----------------------------
    # SES
    d0["educ"] = to_num(d0["educ"])
    d0.loc[(d0["educ"] < 0) | (d0["educ"] > 25), "educ"] = np.nan

    d0["hompop"] = to_num(d0["hompop"])
    d0.loc[(d0["hompop"] <= 0) | (d0["hompop"] > 50), "hompop"] = np.nan

    d0["prestg80"] = to_num(d0["prestg80"])
    d0.loc[(d0["prestg80"] < 0) | (d0["prestg80"] > 100), "prestg80"] = np.nan

    d0["inc_pc"] = np.nan
    m_inc = d0["realinc"].notna() & d0["hompop"].notna() & (d0["hompop"] > 0)
    d0.loc[m_inc, "inc_pc"] = d0.loc[m_inc, "realinc"] / d0.loc[m_inc, "hompop"]
    d0["inc_pc"] = d0["inc_pc"].replace([np.inf, -np.inf], np.nan)
    # Winsorize extreme inc_pc to reduce undue leverage (does not change missingness)
    if d0["inc_pc"].notna().sum() > 20:
        lo, hi = d0["inc_pc"].quantile([0.01, 0.99])
        d0["inc_pc"] = d0["inc_pc"].clip(lower=lo, upper=hi)

    # Female
    d0["female"] = np.where(d0["sex"].isin([1, 2]), (d0["sex"] == 2).astype(float), np.nan)

    # Age (keep 89 top-code as 89)
    d0["age"] = to_num(d0["age"])
    d0.loc[(d0["age"] < 18) | (d0["age"] > 89), "age"] = np.nan  # GSS adult; 89 is top-coded

    # Race dummies (white reference)
    d0["black"] = np.where(d0["race"].isin([1, 2, 3]), (d0["race"] == 2).astype(float), np.nan)
    d0["other_race"] = np.where(d0["race"].isin([1, 2, 3]), (d0["race"] == 3).astype(float), np.nan)

    # Hispanic dummy from ETHNIC (available in provided variables)
    # Treat 20-29 as Hispanic (based on sample values like 29) and 1 as "not Hispanic".
    # If ETHNIC missing -> NaN.
    d0["hispanic"] = np.nan
    m_eth = d0["ethnic"].notna()
    d0.loc[m_eth, "hispanic"] = 0.0
    d0.loc[m_eth & d0["ethnic"].between(20, 29, inclusive="both"), "hispanic"] = 1.0

    # No religion
    d0["no_religion"] = np.where(d0["relig"].isin([1, 2, 3, 4, 5]), (d0["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (as implementable with coarse DENOM)
    d0["conserv_prot"] = np.nan
    m_rd = d0["relig"].notna() & d0["denom"].notna()
    d0.loc[m_rd, "conserv_prot"] = 0.0
    d0.loc[m_rd & (d0["relig"] == 1) & (d0["denom"].isin([1, 6, 7])), "conserv_prot"] = 1.0

    # South
    d0["south"] = np.where(d0["region"].isin([1, 2, 3, 4]), (d0["region"] == 3).astype(float), np.nan)

    # -----------------------------
    # 5) Political intolerance scale (0-15)
    #    Use all 15 items; create count requiring complete non-missing.
    # -----------------------------
    # Ensure tolerance items only have expected codes; otherwise treat as missing.
    for c in tol_items:
        s = d0[c]
        if c.startswith("spk"):
            d0.loc[~s.isin([1, 2]) & s.notna(), c] = np.nan
        elif c.startswith("lib"):
            d0.loc[~s.isin([1, 2]) & s.notna(), c] = np.nan
        elif c.startswith("col"):
            if c == "colcom":
                d0.loc[~s.isin([4, 5]) & s.notna(), c] = np.nan
            else:
                d0.loc[~s.isin([4, 5]) & s.notna(), c] = np.nan

    def intolerance_indicator(col, s):
        s = to_num(s)
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
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    intoler = pd.DataFrame({c: intolerance_indicator(c, d0[c]) for c in tol_items})
    d0["polintol"] = np.nan
    m_tol = intoler.notna().all(axis=1)
    d0.loc[m_tol, "polintol"] = intoler.loc[m_tol].sum(axis=1).astype(float)

    # -----------------------------
    # 6) Models and standardized betas
    # -----------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    def fit_model(data, yvar, xvars, model_name):
        cols = [yvar] + xvars
        d = data.loc[:, cols].dropna(how="any").copy()

        # Variation checks (catch broken dummies like all-0 or all-missing)
        assert_variation(d, xvars, context=f"for {model_name}")

        yv = d[yvar].astype(float)
        X = sm.add_constant(d[xvars].astype(float), has_constant="add")
        model = sm.OLS(yv, X).fit()

        sd_y = pop_sd(yv.values)
        betas = {}
        for v in xvars:
            sd_x = pop_sd(d[v].values)
            b = model.params.get(v, np.nan)
            if np.isnan(sd_y) or sd_y == 0 or np.isnan(sd_x) or sd_x == 0:
                betas[v] = np.nan
            else:
                betas[v] = b * (sd_x / sd_y)

        tab = pd.DataFrame({
            "term": ["const"] + xvars,
            "coef_raw": [model.params.get("const", np.nan)] + [model.params.get(v, np.nan) for v in xvars],
            "beta_std": [np.nan] + [betas[v] for v in xvars],
            "p_raw": [model.pvalues.get("const", np.nan)] + [model.pvalues.get(v, np.nan) for v in xvars],
        })
        tab["sig"] = tab["p_raw"].map(stars)
        fit = {"N": int(d.shape[0]), "R2": float(model.rsquared), "Adj_R2": float(model.rsquared_adj)}
        return model, tab, fit

    m1, tab1, fit1 = fit_model(d0, y, x_m1, "Model 1 (SES)")
    m2, tab2, fit2 = fit_model(d0, y, x_m2, "Model 2 (Demographic)")
    m3, tab3, fit3 = fit_model(d0, y, x_m3, "Model 3 (Political intolerance)")

    # -----------------------------
    # 7) Table 1-style output: standardized betas only (no SEs); raw intercept
    # -----------------------------
    def make_table(tabs, names, order_all):
        long = []
        for name, t in zip(names, tabs):
            tt = t.copy()
            tt.insert(0, "model", name)
            long.append(tt)
        long = pd.concat(long, ignore_index=True)

        disp = long.copy()
        disp["value"] = ""

        is_const = disp["term"].eq("const")
        is_pred = ~is_const

        disp.loc[is_pred, "value"] = disp.loc[is_pred, "beta_std"].map(
            lambda x: "" if pd.isna(x) else f"{x: .3f}"
        ) + disp.loc[is_pred, "sig"]

        disp.loc[is_const, "value"] = disp.loc[is_const, "coef_raw"].map(
            lambda x: "" if pd.isna(x) else f"{x: .3f}"
        )

        pivot = disp.pivot(index="term", columns="model", values="value")

        # enforce order, then const
        idx = [v for v in order_all if v in pivot.index]
        if "const" in pivot.index:
            idx.append("const")
        extras = [t for t in pivot.index if t not in idx]
        pivot = pivot.reindex(idx + extras)

        return pivot, long

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    table1, long_all = make_table([tab1, tab2, tab3], model_names, x_m3)

    fitstats = pd.DataFrame([
        {"model": model_names[0], **fit1},
        {"model": model_names[1], **fit2},
        {"model": model_names[2], **fit3},
    ])

    # -----------------------------
    # 8) Diagnostics: missingness after DV construction
    # -----------------------------
    def missing_report(data, cols):
        d = data.loc[:, cols].copy()
        return pd.DataFrame({
            "var": cols,
            "missing_n": [int(d[c].isna().sum()) for c in cols],
            "missing_pct": [float(d[c].isna().mean()) for c in cols],
            "n_unique_nonmissing": [int(d[c].nunique(dropna=True)) for c in cols],
        }).sort_values(["missing_n", "var"], ascending=[False, True])

    miss_m1 = missing_report(d0, [y] + x_m1)
    miss_m2 = missing_report(d0, [y] + x_m2)
    miss_m3 = missing_report(d0, [y] + x_m3)

    # -----------------------------
    # 9) Save human-readable outputs
    # -----------------------------
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS (1993 GSS music module)")
    summary_lines.append("")
    summary_lines.append("DV: Musical exclusiveness = count (0-18) of genres disliked (4/5) across 18 items;")
    summary_lines.append("    listwise complete across all 18 genre ratings (DK/missing excluded).")
    summary_lines.append("")
    summary_lines.append("OLS on raw DV; reported coefficients for predictors are standardized betas computed as:")
    summary_lines.append("    beta = b * sd(x) / sd(y) using each model's estimation sample.")
    summary_lines.append("Intercept is the raw OLS intercept (unstandardized).")
    summary_lines.append("Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed, from raw OLS p-values).")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Table 1-style coefficient table:")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Missingness diagnostics (after DV construction):")
    summary_lines.append("")
    summary_lines.append("Model 1 variables:")
    summary_lines.append(miss_m1.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 2 variables:")
    summary_lines.append(miss_m2.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 3 variables:")
    summary_lines.append(miss_m3.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Raw model summaries (OLS):")
    summary_lines.append("\n==== Model 1 (SES) ====\n" + m1.summary().as_text())
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + m2.summary().as_text())
    summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + m3.summary().as_text())

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("Cells: standardized beta for predictors (with stars from raw OLS p-values); raw intercept for constant.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n\nFit statistics:\n")
        f.write(fitstats.to_string(index=False))
        f.write("\n")

    long_out = long_all.loc[:, ["model", "term", "coef_raw", "beta_std", "p_raw", "sig"]].copy()
    long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)

    ns = pd.DataFrame({"model": model_names, "N_used": [fit1["N"], fit2["N"], fit3["N"]]})
    ns.to_csv("./output/model_sample_sizes.tsv", sep="\t", index=False)

    return {
        "fit_stats": fitstats,
        "table1_style_betas": table1,
        "coefficients_long": long_out,
        "missingness_model1": miss_m1,
        "missingness_model2": miss_m2,
        "missingness_model3": miss_m3,
    }