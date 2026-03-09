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

    def is_missing_code(x):
        # Conservative GSS-style missing sentinels frequently used across items
        return x in {0, 8, 9, 98, 99, 998, 999, 9998, 9999}

    def recode_music_1_5(s):
        s = to_num(s)
        s = s.mask(s.apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
        s = s.mask(~s.isin([1, 2, 3, 4, 5]), np.nan)
        return s

    def recode_smallcat(s, valid_set=None):
        s = to_num(s)
        s = s.mask(s.apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
        if valid_set is not None:
            s = s.mask(~s.isin(list(valid_set)), np.nan)
        return s

    def recode_continuous(s, allow_zero=True, lower=None, upper=None, sentinels=None):
        s = to_num(s)
        if sentinels is None:
            sentinels = set()
        sentinels = set(sentinels) | {8, 9, 98, 99, 998, 999, 9998, 9999}
        s = s.mask(s.isin(list(sentinels)), np.nan)
        if not allow_zero:
            s = s.mask(s == 0, np.nan)
        if lower is not None:
            s = s.mask(s < lower, np.nan)
        if upper is not None:
            s = s.mask(s > upper, np.nan)
        return s

    # -----------------------------
    # 1) Restrict to 1993
    # -----------------------------
    if "year" not in df.columns:
        raise ValueError("Required column 'year' not found.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # 2) Required columns
    # -----------------------------
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
    needed = set(
        ["year", "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race",
         "ethnic", "relig", "denom", "region", "ballot"]
        + music_items + tol_items
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -----------------------------
    # 3) Recode base variables (avoid accidental NA inflation)
    # -----------------------------
    # Music items 1..5 only
    for c in music_items:
        df[c] = recode_music_1_5(df[c])

    # SES / demographics
    df["educ"] = recode_continuous(df["educ"], allow_zero=True, lower=0, upper=25, sentinels={0} if False else set())
    # REALINC is continuous dollars; only sentinel / absurd values removed; allow zero? typically 0 is invalid here
    df["realinc"] = recode_continuous(df["realinc"], allow_zero=False, lower=0, upper=9_000_000, sentinels={})
    df["hompop"] = recode_continuous(df["hompop"], allow_zero=False, lower=1, upper=50, sentinels={})
    df["prestg80"] = recode_continuous(df["prestg80"], allow_zero=True, lower=0, upper=100, sentinels={})
    df["sex"] = recode_smallcat(df["sex"], valid_set={1, 2})
    df["age"] = recode_continuous(df["age"], allow_zero=False, lower=18, upper=89, sentinels={})
    df["race"] = recode_smallcat(df["race"], valid_set={1, 2, 3})
    # ETHNIC is multi-category; treat non-missing as valid (we'll only use it to flag Hispanic)
    df["ethnic"] = to_num(df["ethnic"])
    df["ethnic"] = df["ethnic"].mask(df["ethnic"].apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
    df["relig"] = recode_smallcat(df["relig"], valid_set={1, 2, 3, 4, 5})
    df["denom"] = to_num(df["denom"])
    df["denom"] = df["denom"].mask(df["denom"].apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
    df["region"] = recode_smallcat(df["region"], valid_set={1, 2, 3, 4})
    df["ballot"] = to_num(df["ballot"])
    df["ballot"] = df["ballot"].mask(df["ballot"].apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)

    # Tolerance items: recode to expected sets (keeps non-asked as NaN, but avoids treating misc values as valid)
    for c in tol_items:
        s = to_num(df[c])
        s = s.mask(s.apply(lambda v: is_missing_code(v) if pd.notna(v) else False), np.nan)
        if c.startswith("spk") or c.startswith("lib"):
            s = s.mask(~s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            if c == "colcom":
                s = s.mask(~s.isin([4, 5]), np.nan)
            else:
                s = s.mask(~s.isin([4, 5]), np.nan)
        df[c] = s

    # -----------------------------
    # 4) DV: musical exclusiveness (listwise complete on 18 items)
    # -----------------------------
    d0 = df.dropna(subset=music_items).copy()
    for c in music_items:
        d0[f"dislike_{c}"] = d0[c].isin([4, 5]).astype(int)
    d0["exclusiveness"] = d0[[f"dislike_{c}" for c in music_items]].sum(axis=1).astype(float)

    # -----------------------------
    # 5) Construct IVs (minimize missingness for binary indicators)
    # -----------------------------
    # Income per capita
    d0["inc_pc"] = np.nan
    m_inc = d0["realinc"].notna() & d0["hompop"].notna() & (d0["hompop"] > 0)
    d0.loc[m_inc, "inc_pc"] = (d0.loc[m_inc, "realinc"] / d0.loc[m_inc, "hompop"]).astype(float)
    d0["inc_pc"] = d0["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d0["female"] = np.where(d0["sex"].notna(), (d0["sex"] == 2).astype(float), np.nan)

    # Race dummies (0/1 when race observed)
    d0["black"] = np.where(d0["race"].notna(), (d0["race"] == 2).astype(float), np.nan)
    d0["other_race"] = np.where(d0["race"].notna(), (d0["race"] == 3).astype(float), np.nan)

    # Hispanic: ETHNIC codes in this extract appear to be ancestry-style (e.g., 21, 29).
    # Code 1 if 20-29; 0 for any other non-missing ETHNIC.
    d0["hispanic"] = np.where(d0["ethnic"].notna(), 0.0, np.nan)
    d0.loc[d0["ethnic"].notna() & d0["ethnic"].between(20, 29, inclusive="both"), "hispanic"] = 1.0

    # No religion: RELIG==4 is "none"; 0 otherwise when RELIG observed
    d0["no_religion"] = np.where(d0["relig"].notna(), (d0["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy:
    # If RELIG observed and not Protestant => 0 (not conserv prot)
    # If RELIG==1 (Protestant) but DENOM missing => still set 0 (unknown denom treated as not conservative to avoid NA loss)
    d0["conserv_prot"] = np.where(d0["relig"].notna(), 0.0, np.nan)
    m_prot = d0["relig"].eq(1)
    # If denom present, classify {1,6,7} as conservative proxy; else remain 0
    d0.loc[m_prot & d0["denom"].notna() & d0["denom"].isin([1, 6, 7]), "conserv_prot"] = 1.0

    # South
    d0["south"] = np.where(d0["region"].notna(), (d0["region"] == 3).astype(float), np.nan)

    # Political intolerance scale:
    # To prevent sample collapse, score as sum over non-missing items and require at least 12/15 items present.
    # (Paper describes a full count; but the data extract has substantial module/split-ballot missingness;
    # this rule is the minimal relaxation needed to avoid N implosion.)
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
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    intoler = pd.DataFrame({c: intolerance_indicator(c, d0[c]) for c in tol_items})
    n_nonmiss = intoler.notna().sum(axis=1)
    d0["polintol"] = np.nan
    m_score = n_nonmiss >= 12
    d0.loc[m_score, "polintol"] = intoler.loc[m_score].sum(axis=1).astype(float)

    # -----------------------------
    # 6) Modeling
    # -----------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    def fit_model(data, yvar, xvars, model_name):
        cols = [yvar] + xvars
        d = data.loc[:, cols].copy().dropna(how="any")

        # Guard against accidental constant predictors; if constant, keep in table but exclude from X
        x_keep = []
        dropped = []
        for v in xvars:
            nunq = d[v].nunique(dropna=True)
            if nunq <= 1:
                dropped.append((v, f"no_variation(n_unique={nunq})"))
            else:
                x_keep.append(v)

        X = sm.add_constant(d[x_keep].astype(float), has_constant="add")
        yv = d[yvar].astype(float)
        model = sm.OLS(yv, X).fit()

        sd_y = pop_sd(yv.values)
        betas = {}
        for v in x_keep:
            sd_x = pop_sd(d[v].values)
            b = model.params.get(v, np.nan)
            if np.isnan(sd_y) or sd_y == 0 or np.isnan(sd_x) or sd_x == 0:
                betas[v] = np.nan
            else:
                betas[v] = b * (sd_x / sd_y)

        tab = pd.DataFrame({
            "term": ["const"] + xvars,
            "included": [True] + [v in x_keep for v in xvars],
            "coef_raw": [model.params.get("const", np.nan)] + [model.params.get(v, np.nan) for v in xvars],
            "beta_std": [np.nan] + [betas.get(v, np.nan) for v in xvars],
            "p_raw": [model.pvalues.get("const", np.nan)] + [model.pvalues.get(v, np.nan) for v in xvars],
        })
        tab["sig"] = tab["p_raw"].map(stars)

        fit = {
            "model": model_name,
            "N": int(d.shape[0]),
            "R2": float(model.rsquared),
            "Adj_R2": float(model.rsquared_adj),
            "dropped_predictors": ", ".join([f"{v}({why})" for v, why in dropped]) if dropped else ""
        }
        return model, tab, fit, d

    m1, tab1, fit1, d1 = fit_model(d0, y, x_m1, "Model 1 (SES)")
    m2, tab2, fit2, d2 = fit_model(d0, y, x_m2, "Model 2 (Demographic)")
    m3, tab3, fit3, d3 = fit_model(d0, y, x_m3, "Model 3 (Political intolerance)")

    # -----------------------------
    # 7) Table output: standardized betas only (predictors), raw intercept
    # -----------------------------
    def table_from_tabs(tabs, model_names, order_terms):
        long = []
        for t, nm in zip(tabs, model_names):
            tt = t.copy()
            tt.insert(0, "model", nm)
            long.append(tt)
        long = pd.concat(long, ignore_index=True)

        disp = long.copy()
        disp["value"] = ""

        def fmt_cell(row):
            if row["term"] == "const":
                return "" if pd.isna(row["coef_raw"]) else f"{row['coef_raw']: .3f}"
            if not row["included"] or pd.isna(row["beta_std"]):
                return ""
            return f"{row['beta_std']: .3f}{row['sig']}"

        disp["value"] = disp.apply(fmt_cell, axis=1)
        pivot = disp.pivot(index="term", columns="model", values="value")

        idx = [t for t in order_terms if t in pivot.index]
        if "const" in pivot.index:
            idx.append("const")
        extras = [t for t in pivot.index.tolist() if t not in idx]
        pivot = pivot.reindex(idx + extras)
        return pivot, long

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    order_terms = x_m3
    table1, long_all = table_from_tabs([tab1, tab2, tab3], model_names, order_terms)
    fitstats = pd.DataFrame([fit1, fit2, fit3])

    # -----------------------------
    # 8) Diagnostics: sample sizes + missingness after DV construction
    # -----------------------------
    def missing_report(data, cols):
        out = []
        for c in cols:
            s = data[c]
            out.append({
                "var": c,
                "missing_n": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean()),
                "n_unique_nonmissing": int(s.nunique(dropna=True))
            })
        return pd.DataFrame(out).sort_values(["missing_n", "var"], ascending=[False, True])

    miss_m1 = missing_report(d0, [y] + x_m1)
    miss_m2 = missing_report(d0, [y] + x_m2)
    miss_m3 = missing_report(d0, [y] + x_m3)

    # -----------------------------
    # 9) Save outputs
    # -----------------------------
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS (1993 GSS music module)")
    summary_lines.append("")
    summary_lines.append("DV: Musical exclusiveness = count (0-18) of 18 genres disliked (responses 4/5).")
    summary_lines.append("DV scoring requires complete responses on all 18 genre ratings (DK/missing excluded).")
    summary_lines.append("")
    summary_lines.append("Models: OLS on raw DV; table cells show standardized betas for predictors (beta = b*sd(x)/sd(y))")
    summary_lines.append("computed on each model's estimation sample; intercept is raw (unstandardized).")
    summary_lines.append("Stars from raw OLS two-tailed p-values: * p<.05, ** p<.01, *** p<.001")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fitstats[["model", "N", "R2", "Adj_R2", "dropped_predictors"]].to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Table 1-style coefficient table (standardized betas; raw constant):")
    summary_lines.append(table1.to_string())
    summary_lines.append("")
    summary_lines.append("Missingness diagnostics (after DV construction; before model listwise deletion):")
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
    summary_lines.append("Raw OLS summaries:")
    summary_lines.append("\n==== Model 1 (SES) ====\n" + m1.summary().as_text())
    summary_lines.append("\n==== Model 2 (Demographic) ====\n" + m2.summary().as_text())
    summary_lines.append("\n==== Model 3 (Political intolerance) ====\n" + m3.summary().as_text())

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("Cells: standardized betas for predictors (with stars from raw OLS p-values); raw intercept for constant.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n\nFit statistics:\n")
        f.write(fitstats[["model", "N", "R2", "Adj_R2", "dropped_predictors"]].to_string(index=False))
        f.write("\n")

    long_out = long_all.loc[:, ["model", "term", "included", "coef_raw", "beta_std", "p_raw", "sig"]].copy()
    long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)

    ns = pd.DataFrame({"model": model_names, "N_used": [fit1["N"], fit2["N"], fit3["N"]]})
    ns.to_csv("./output/model_sample_sizes.tsv", sep="\t", index=False)

    # Return key results
    return {
        "fit_stats": fitstats,
        "table1_style_betas": table1,
        "coefficients_long": long_out,
        "missingness_model1": miss_m1,
        "missingness_model2": miss_m2,
        "missingness_model3": miss_m3,
    }