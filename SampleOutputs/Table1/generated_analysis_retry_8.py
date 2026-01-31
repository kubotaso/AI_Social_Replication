def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def is_missing_code(v):
        # Broad GSS-style missing / inapplicable sentinels
        return v in {0, 8, 9, 98, 99, 998, 999, 9998, 9999}

    def clean_cat(series, valid=None):
        s = to_num(series)
        s = s.mask(s.apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
        if valid is not None:
            s = s.mask(~s.isin(list(valid)), np.nan)
        return s

    def clean_cont(series, lower=None, upper=None, allow_zero=True):
        s = to_num(series)
        s = s.mask(s.apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
        if not allow_zero:
            s = s.mask(s == 0, np.nan)
        if lower is not None:
            s = s.mask(s < lower, np.nan)
        if upper is not None:
            s = s.mask(s > upper, np.nan)
        return s

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

    # -------------------------
    # 1) Restrict to 1993
    # -------------------------
    if "year" not in df.columns:
        raise ValueError("Column 'year' is required.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # 2) Variable lists
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

    required = set(
        ["year", "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race",
         "relig", "denom", "region", "ballot"]
        + music_items + tol_items
    )
    missing_cols = [c for c in sorted(required) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # 3) Clean / recode base fields (minimal, faithful)
    # -------------------------
    # Music items: keep only 1..5; set other codes to NaN
    for c in music_items:
        s = clean_cat(df[c], valid={1, 2, 3, 4, 5})
        df[c] = s

    df["educ"] = clean_cont(df["educ"], lower=0, upper=25, allow_zero=True)
    df["realinc"] = clean_cont(df["realinc"], lower=0, upper=9_000_000, allow_zero=False)
    df["hompop"] = clean_cont(df["hompop"], lower=1, upper=50, allow_zero=False)
    df["prestg80"] = clean_cont(df["prestg80"], lower=0, upper=100, allow_zero=True)

    df["sex"] = clean_cat(df["sex"], valid={1, 2})
    df["age"] = clean_cont(df["age"], lower=18, upper=89, allow_zero=False)

    df["race"] = clean_cat(df["race"], valid={1, 2, 3})
    df["relig"] = clean_cat(df["relig"], valid={1, 2, 3, 4, 5})
    df["denom"] = clean_cont(df["denom"], lower=0, upper=999, allow_zero=True)  # keep as numeric; we'll validate where used
    df["region"] = clean_cat(df["region"], valid={1, 2, 3, 4})
    df["ballot"] = clean_cont(df["ballot"], lower=1, upper=9, allow_zero=False)

    # Tolerance items: keep only legitimate substantive codes
    for c in tol_items:
        s = to_num(df[c])
        s = s.mask(s.apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
        if c.startswith("spk"):
            s = s.mask(~s.isin([1, 2]), np.nan)
        elif c.startswith("lib"):
            s = s.mask(~s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            if c == "colcom":
                s = s.mask(~s.isin([4, 5]), np.nan)  # 4 fired (intolerant), 5 not fired
            else:
                s = s.mask(~s.isin([4, 5]), np.nan)  # 4 allowed, 5 not allowed (intolerant)
        df[c] = s

    # -------------------------
    # 4) DV: Musical exclusiveness (listwise complete on all 18 items)
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    d["exclusiveness"] = d[[f"dislike_{c}" for c in music_items]].sum(axis=1).astype(float)

    # -------------------------
    # 5) IVs
    # -------------------------
    # Income per capita
    d["inc_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "inc_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["inc_pc"] = d["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white is reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: NOT AVAILABLE in provided fields -> do not fabricate; keep as 0 (observed) to avoid sample loss.
    # This preserves sample size while reflecting that the variable cannot be constructed from this extract.
    d["hispanic"] = 0.0

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}; if RELIG observed but denom missing, set 0 (no NA loss)
    d["conserv_prot"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1)
    d.loc[m_prot & d["denom"].notna() & d["denom"].isin([1, 6, 7]), "conserv_prot"] = 1.0

    # South
    d["south"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance (0-15): require complete on all 15 items (as described)
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
    d["polintol"] = np.nan
    m_complete_tol = intoler.notna().all(axis=1)
    d.loc[m_complete_tol, "polintol"] = intoler.loc[m_complete_tol].sum(axis=1).astype(float)

    # -------------------------
    # 6) Models (OLS) + standardized betas computed on estimation sample
    # -------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    def fit_model(data, yvar, xvars, name):
        cols = [yvar] + xvars
        dd = data.loc[:, cols].copy().dropna(how="any")

        # drop predictors with no variance (but keep them in the display as blank)
        x_keep, dropped = [], []
        for v in xvars:
            nunq = dd[v].nunique(dropna=True)
            if nunq <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[yvar].astype(float)
        model = sm.OLS(yy, X).fit()

        sd_y = pop_sd(yy.values)
        betas = {}
        for v in x_keep:
            sd_x = pop_sd(dd[v].values)
            b = model.params.get(v, np.nan)
            betas[v] = np.nan if (not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0) else b * (sd_x / sd_y)

        tab = pd.DataFrame({
            "term": ["const"] + xvars,
            "included": [True] + [v in x_keep for v in xvars],
            "coef_raw": [model.params.get("const", np.nan)] + [model.params.get(v, np.nan) for v in xvars],
            "beta_std": [np.nan] + [betas.get(v, np.nan) for v in xvars],
            "p_raw": [model.pvalues.get("const", np.nan)] + [model.pvalues.get(v, np.nan) for v in xvars],
        })
        tab["sig"] = tab["p_raw"].map(stars)

        fit = {
            "model": name,
            "N": int(dd.shape[0]),
            "R2": float(model.rsquared),
            "Adj_R2": float(model.rsquared_adj),
            "dropped_no_variance": ", ".join(dropped) if dropped else ""
        }
        return model, tab, fit, dd

    m1, tab1, fit1, d1 = fit_model(d, y, x_m1, "Model 1 (SES)")
    m2, tab2, fit2, d2 = fit_model(d, y, x_m2, "Model 2 (Demographic)")
    m3, tab3, fit3, d3 = fit_model(d, y, x_m3, "Model 3 (Political intolerance)")

    # -------------------------
    # 7) Build Table 1-style display (standardized betas + stars; raw constant)
    # -------------------------
    def make_table(tabs, names, order_terms):
        blocks = []
        for t, nm in zip(tabs, names):
            tt = t.copy()
            tt.insert(0, "model", nm)
            blocks.append(tt)
        long = pd.concat(blocks, ignore_index=True)

        disp = long.copy()

        def fmt(row):
            if row["term"] == "const":
                return "" if pd.isna(row["coef_raw"]) else f"{row['coef_raw']:.3f}"
            if not row["included"] or pd.isna(row["beta_std"]):
                return ""
            return f"{row['beta_std']:.3f}{row['sig']}"

        disp["cell"] = disp.apply(fmt, axis=1)
        wide = disp.pivot(index="term", columns="model", values="cell")

        idx = [t for t in order_terms if t in wide.index]
        if "const" in wide.index and "const" not in idx:
            idx = idx + ["const"]
        extras = [t for t in wide.index if t not in idx]
        wide = wide.reindex(idx + extras)
        return wide, long

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    table1, long_all = make_table([tab1, tab2, tab3], model_names, x_m3)

    fitstats = pd.DataFrame([fit1, fit2, fit3])

    # -------------------------
    # 8) Save outputs
    # -------------------------
    summary = []
    summary.append("Replication output: Table 1-style OLS (1993 GSS)")
    summary.append("")
    summary.append("DV: musical exclusiveness = count (0-18) of 18 genres rated 4/5 (dislike / dislike very much).")
    summary.append("DV scoring: listwise complete responses across all 18 music items (DK/missing excluded).")
    summary.append("")
    summary.append("Models: OLS; table shows standardized betas (beta = b * SD(x)/SD(y)) computed on each model's estimation sample.")
    summary.append("Stars from raw OLS two-tailed p-values: * p<.05, ** p<.01, *** p<.001")
    summary.append("")
    summary.append("Fit statistics:")
    summary.append(fitstats[["model", "N", "R2", "Adj_R2", "dropped_no_variance"]].to_string(index=False))
    summary.append("")
    summary.append("Table 1-style coefficients (standardized betas; raw constant):")
    summary.append(table1.to_string())
    summary.append("")
    summary.append("Raw OLS summaries:")
    summary.append("\n==== Model 1 (SES) ====\n" + m1.summary().as_text())
    summary.append("\n==== Model 2 (Demographic) ====\n" + m2.summary().as_text())
    summary.append("\n==== Model 3 (Political intolerance) ====\n" + m3.summary().as_text())

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("Cells: standardized betas for predictors (with stars from raw OLS p-values); raw intercept for constant.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n\nFit statistics:\n")
        f.write(fitstats[["model", "N", "R2", "Adj_R2", "dropped_no_variance"]].to_string(index=False))
        f.write("\n")

    long_out = long_all.loc[:, ["model", "term", "included", "coef_raw", "beta_std", "p_raw", "sig"]].copy()
    long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)

    # Return key results
    return {
        "fit_stats": fitstats,
        "table1_style_betas": table1,
        "coefficients_long": long_out,
        "n_dv_complete_music": int(d.shape[0]),
        "n_complete_polintol": int(m_complete_tol.sum()),
    }