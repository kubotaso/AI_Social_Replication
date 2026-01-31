def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def pop_sd(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
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

    # --------------------------
    # 1) Restrict to 1993
    # --------------------------
    if "year" in df.columns:
        df = df.loc[df["year"] == 1993].copy()
    elif "YEAR" in df.columns:
        df = df.loc[df["YEAR"] == 1993].copy()

    # --------------------------
    # 2) DV: musical exclusiveness
    #    Count of 18 genres disliked (4/5), listwise complete on all 18 items.
    # --------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music columns: {missing_music}")

    for c in music_items:
        df[c] = to_num(df[c])

    d0 = df.dropna(subset=music_items).copy()
    for c in music_items:
        d0[f"dislike_{c}"] = d0[c].isin([4, 5]).astype(int)
    d0["exclusiveness"] = d0[[f"dislike_{c}" for c in music_items]].sum(axis=1).astype(float)

    # --------------------------
    # 3) Predictors (Table 1)
    # --------------------------
    # SES
    for c in ["educ", "realinc", "hompop", "prestg80"]:
        if c in d0.columns:
            d0[c] = to_num(d0[c])
        else:
            d0[c] = np.nan

    d0["inc_pc"] = np.nan
    m_inc = d0["realinc"].notna() & d0["hompop"].notna() & (d0["hompop"] > 0)
    d0.loc[m_inc, "inc_pc"] = (d0.loc[m_inc, "realinc"] / d0.loc[m_inc, "hompop"]).astype(float)
    d0["inc_pc"] = d0["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    if "sex" in d0.columns:
        d0["sex"] = to_num(d0["sex"])
        d0["female"] = np.where(d0["sex"].isin([1, 2]), (d0["sex"] == 2).astype(float), np.nan)
    else:
        d0["female"] = np.nan

    # Age
    if "age" in d0.columns:
        d0["age"] = to_num(d0["age"])
    else:
        d0["age"] = np.nan

    # Race
    if "race" in d0.columns:
        d0["race"] = to_num(d0["race"])
        d0["black"] = np.where(d0["race"].isin([1, 2, 3]), (d0["race"] == 2).astype(float), np.nan)
        d0["other_race"] = np.where(d0["race"].isin([1, 2, 3]), (d0["race"] == 3).astype(float), np.nan)
    else:
        d0["black"] = np.nan
        d0["other_race"] = np.nan

    # Hispanic: FIXED to avoid NA-collapsing
    # Use ETHNIC if present; treat 2 as Hispanic, 1 as not Hispanic, everything else -> 0 (not NA)
    # This avoids wiping out the sample when ETHNIC is present but multi-category.
    if "ethnic" in d0.columns:
        d0["ethnic"] = to_num(d0["ethnic"])
        d0["hispanic"] = 0.0
        d0.loc[d0["ethnic"] == 2, "hispanic"] = 1.0
        d0.loc[d0["ethnic"].isna(), "hispanic"] = np.nan  # keep truly missing as missing
    else:
        d0["hispanic"] = np.nan

    # Religion / denom
    if "relig" in d0.columns:
        d0["relig"] = to_num(d0["relig"])
        d0["no_religion"] = np.where(d0["relig"].isin([1, 2, 3, 4, 5]), (d0["relig"] == 4).astype(float), np.nan)
    else:
        d0["relig"] = np.nan
        d0["no_religion"] = np.nan

    if "denom" in d0.columns:
        d0["denom"] = to_num(d0["denom"])
    else:
        d0["denom"] = np.nan

    # Conservative Protestant approximation: RELIG==1 and DENOM in {1,6,7}
    d0["conserv_prot"] = np.nan
    m_rd = d0["relig"].notna() & d0["denom"].notna()
    d0.loc[m_rd, "conserv_prot"] = 0.0
    d0.loc[m_rd & (d0["relig"] == 1) & (d0["denom"].isin([1, 6, 7])), "conserv_prot"] = 1.0

    # South
    if "region" in d0.columns:
        d0["region"] = to_num(d0["region"])
        d0["south"] = np.where(d0["region"].isin([1, 2, 3, 4]), (d0["region"] == 3).astype(float), np.nan)
    else:
        d0["south"] = np.nan

    # --------------------------
    # 4) Political intolerance scale (0-15)
    #    FIXED to avoid over-missingness: require all 15 non-missing for the scale,
    #    but code dummies correctly; keep scale missing only when any item missing.
    # --------------------------
    tol_items = [
        "spkath", "colath", "libath",
        "spkrac", "colrac", "librac",
        "spkcom", "colcom", "libcom",
        "spkmil", "colmil", "libmil",
        "spkhomo", "colhomo", "libhomo"
    ]
    for c in tol_items:
        if c in d0.columns:
            d0[c] = to_num(d0[c])
        else:
            d0[c] = np.nan

    def intolerance_indicator(col, s):
        s = to_num(s)
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

    intoler = pd.DataFrame({c: intolerance_indicator(c, d0[c]) for c in tol_items})
    d0["polintol"] = np.nan
    m_tol = intoler.notna().all(axis=1)
    d0.loc[m_tol, "polintol"] = intoler.loc[m_tol].sum(axis=1).astype(float)

    # Ensure numeric/finites
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    for c in set([y] + x_m3):
        if c in d0.columns:
            d0[c] = to_num(d0[c]).replace([np.inf, -np.inf], np.nan)

    # --------------------------
    # 5) Fit models: OLS on raw DV; compute standardized betas from estimation sample
    # --------------------------
    def fit_model(data, yvar, xvars):
        cols = [yvar] + xvars
        d = data.loc[:, cols].copy().dropna(axis=0, how="any")

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

        fit = {
            "N": int(d.shape[0]),
            "R2": float(model.rsquared),
            "Adj_R2": float(model.rsquared_adj),
        }
        return model, tab, fit, d

    m1, tab1, fit1, d1 = fit_model(d0, y, x_m1)
    m2, tab2, fit2, d2 = fit_model(d0, y, x_m2)
    m3, tab3, fit3, d3 = fit_model(d0, y, x_m3)

    # --------------------------
    # 6) Build Table 1-style output (standardized betas; raw constant WITHOUT stars)
    # --------------------------
    def make_table(tabs, names, order):
        long = []
        for name, t in zip(names, tabs):
            tt = t.copy()
            tt.insert(0, "model", name)
            long.append(tt)
        long = pd.concat(long, ignore_index=True)

        disp = long.copy()
        disp["value"] = ""

        # predictors: standardized betas + stars
        m_pred = disp["term"] != "const"
        disp.loc[m_pred, "value"] = disp.loc[m_pred, "beta_std"].map(
            lambda x: "" if pd.isna(x) else f"{x: .3f}"
        ) + disp.loc[m_pred, "sig"]

        # constant: raw, NO stars (to mirror typical Table 1 formatting)
        m_const = disp["term"] == "const"
        disp.loc[m_const, "value"] = disp.loc[m_const, "coef_raw"].map(
            lambda x: "" if pd.isna(x) else f"{x: .3f}"
        )

        pivot = disp.pivot(index="term", columns="model", values="value")

        idx = []
        for v in order:
            if v in pivot.index:
                idx.append(v)
        if "const" in pivot.index:
            idx.append("const")
        extras = [t for t in pivot.index.tolist() if t not in idx]
        pivot = pivot.reindex(index=idx + extras)

        return pivot, long

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    table1, long_all = make_table([tab1, tab2, tab3], model_names, x_m3)

    fitstats = pd.DataFrame([
        {"model": model_names[0], **fit1},
        {"model": model_names[1], **fit2},
        {"model": model_names[2], **fit3},
    ])

    # --------------------------
    # 7) Missingness diagnostics (post-DV construction)
    # --------------------------
    def missing_report(data, cols):
        d = data.loc[:, cols].copy()
        return pd.DataFrame({
            "var": cols,
            "missing_n": [int(d[c].isna().sum()) for c in cols],
            "missing_pct": [float(d[c].isna().mean()) for c in cols],
        }).sort_values(["missing_n", "var"], ascending=[False, True])

    miss_m1 = missing_report(d0, [y] + x_m1)
    miss_m2 = missing_report(d0, [y] + x_m2)
    miss_m3 = missing_report(d0, [y] + x_m3)

    # --------------------------
    # 8) Save outputs
    # --------------------------
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS (1993 GSS music module)")
    summary_lines.append("")
    summary_lines.append("DV: Musical exclusiveness = count (0-18) of genres disliked (4/5) across 18 items;")
    summary_lines.append("    listwise complete on all 18 genre ratings (DK/missing excluded).")
    summary_lines.append("")
    summary_lines.append("Estimation: OLS on raw DV; reported coefficients are standardized betas for predictors")
    summary_lines.append("            computed from each model's estimation sample; constant is raw intercept (no stars).")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Table 1-style coefficient table (standardized betas; raw constant):")
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
        f.write("Cells: standardized beta for predictors (with stars from raw OLS p-values); raw intercept for constant (no stars).\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n\nFit statistics:\n")
        f.write(fitstats.to_string(index=False))
        f.write("\n")

    long_out = long_all.loc[:, ["model", "term", "coef_raw", "beta_std", "p_raw", "sig"]].copy()
    long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)

    # Also save model Ns explicitly
    ns = pd.DataFrame({
        "model": model_names,
        "N_used": [fit1["N"], fit2["N"], fit3["N"]],
    })
    ns.to_csv("./output/model_sample_sizes.tsv", sep="\t", index=False)

    return {
        "fit_stats": fitstats,
        "table1_style_betas": table1,
        "coefficients_long": long_out,
        "missingness_model1": miss_m1,
        "missingness_model2": miss_m2,
        "missingness_model3": miss_m3,
    }