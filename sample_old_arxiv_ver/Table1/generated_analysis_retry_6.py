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

    def gss_recode_missing(s: pd.Series) -> pd.Series:
        """
        Conservative, variable-agnostic recode:
        - numeric coerce
        - set common DK/NA codes to NaN: 8/9, 98/99, 998/999, 9998/9999
        - for small-range categorical vars, also set values >= 90 to NaN (but avoid doing this
          for continuous vars like income by excluding them from this function).
        """
        s = to_num(s)
        if s.isna().all():
            return s
        special = {8, 9, 98, 99, 998, 999, 9998, 9999}
        out = s.mask(s.isin(list(special)), np.nan)

        nunq = out.nunique(dropna=True)
        if nunq > 0 and nunq <= 30:
            out = out.mask(out >= 90, np.nan)
        return out

    # -----------------------------
    # 1) Restrict to 1993 only
    # -----------------------------
    if "year" not in df.columns:
        raise ValueError("Required column 'year' not found in the dataset.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # 2) Variable lists
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
    base_needed = (
        music_items
        + ["educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
        + tol_items
    )
    missing_cols = [c for c in base_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -----------------------------
    # 3) Recode missingness (avoid breaking REALINC)
    # -----------------------------
    # Recode typical GSS missing for categorical/small-range items
    recode_cols = music_items + ["educ", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"] + tol_items
    for c in recode_cols:
        df[c] = gss_recode_missing(df[c])

    # REALINC: numeric; only treat explicit sentinels/extremes as missing
    df["realinc"] = to_num(df["realinc"])
    df.loc[df["realinc"].isin([0, 8, 9, 98, 99, 998, 999, 9998, 9999]), "realinc"] = np.nan
    df.loc[df["realinc"] < 0, "realinc"] = np.nan
    df.loc[df["realinc"] >= 9_000_000, "realinc"] = np.nan

    # -----------------------------
    # 4) DV: Musical exclusiveness (listwise complete on all 18 items)
    # -----------------------------
    # Only 1..5 are valid; anything else -> NaN
    for c in music_items:
        df.loc[df[c].notna() & ~df[c].isin([1, 2, 3, 4, 5]), c] = np.nan

    d0 = df.dropna(subset=music_items).copy()
    for c in music_items:
        d0[f"dislike_{c}"] = d0[c].isin([4, 5]).astype(int)
    d0["exclusiveness"] = d0[[f"dislike_{c}" for c in music_items]].sum(axis=1).astype(float)

    # -----------------------------
    # 5) IV construction
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
    d0.loc[m_inc, "inc_pc"] = (d0.loc[m_inc, "realinc"] / d0.loc[m_inc, "hompop"]).astype(float)
    d0["inc_pc"] = d0["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Demographics
    d0["female"] = np.where(d0["sex"].isin([1, 2]), (d0["sex"] == 2).astype(float), np.nan)

    d0["age"] = to_num(d0["age"])
    # Do not impose age>=18 filter (paper sample is adults, but GSS is adults; keep only implausible values missing)
    d0.loc[(d0["age"] < 0) | (d0["age"] > 89), "age"] = np.nan

    d0["black"] = np.where(d0["race"].isin([1, 2, 3]), (d0["race"] == 2).astype(float), np.nan)
    d0["other_race"] = np.where(d0["race"].isin([1, 2, 3]), (d0["race"] == 3).astype(float), np.nan)

    # Hispanic: ETHNIC is multi-category; treat 20-29 as Hispanic (matches sample values like 21, 29).
    # Keep non-missing values outside 20-29 as 0.
    d0["hispanic"] = np.nan
    m_eth = d0["ethnic"].notna()
    d0.loc[m_eth, "hispanic"] = 0.0
    d0.loc[m_eth & d0["ethnic"].between(20, 29, inclusive="both"), "hispanic"] = 1.0

    # Religion
    # NOTE: prior runtime error came from no_religion having no variation after listwise deletion.
    # Fix: compute it robustly but DO NOT crash if it becomes constant; drop constant predictors at fit time.
    d0["no_religion"] = np.where(d0["relig"].isin([1, 2, 3, 4, 5]), (d0["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    d0["conserv_prot"] = np.nan
    m_rd = d0["relig"].notna() & d0["denom"].notna()
    d0.loc[m_rd, "conserv_prot"] = 0.0
    d0.loc[m_rd & (d0["relig"] == 1) & (d0["denom"].isin([1, 6, 7])), "conserv_prot"] = 1.0

    d0["south"] = np.where(d0["region"].isin([1, 2, 3, 4]), (d0["region"] == 3).astype(float), np.nan)

    # Political intolerance items: validate codes then build count
    for c in tol_items:
        s = d0[c]
        if c.startswith("spk"):
            d0.loc[s.notna() & ~s.isin([1, 2]), c] = np.nan
        elif c.startswith("lib"):
            d0.loc[s.notna() & ~s.isin([1, 2]), c] = np.nan
        elif c.startswith("col"):
            if c == "colcom":
                d0.loc[s.notna() & ~s.isin([4, 5]), c] = np.nan
            else:
                d0.loc[s.notna() & ~s.isin([4, 5]), c] = np.nan

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

    # -----------------------------
    # 6) Model fitting with safe handling of constant predictors
    # -----------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    def drop_constant_predictors(d, xvars):
        keep = []
        dropped = []
        for v in xvars:
            if v not in d.columns:
                dropped.append((v, "missing_column"))
                continue
            nunq = d[v].nunique(dropna=True)
            if nunq <= 1:
                dropped.append((v, f"no_variation(n_unique={nunq})"))
            else:
                keep.append(v)
        return keep, dropped

    def fit_model(data, yvar, xvars, model_name):
        cols = [yvar] + xvars
        d = data.loc[:, cols].copy().dropna(how="any")

        x_keep, dropped = drop_constant_predictors(d, xvars)
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
            "term": ["const"] + xvars,  # keep original requested ordering, even if dropped
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
        return model, tab, fit, d, x_keep, dropped

    m1, tab1, fit1, d1, x1_keep, drop1 = fit_model(d0, y, x_m1, "Model 1 (SES)")
    m2, tab2, fit2, d2, x2_keep, drop2 = fit_model(d0, y, x_m2, "Model 2 (Demographic)")
    m3, tab3, fit3, d3, x3_keep, drop3 = fit_model(d0, y, x_m3, "Model 3 (Political intolerance)")

    # -----------------------------
    # 7) Table 1-style output (standardized betas only; intercept raw)
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

        is_const = disp["term"].eq("const")
        is_pred = ~is_const

        # Predictors: show standardized beta if included, else blank
        def fmt_beta(row):
            if not row["included"] or pd.isna(row["beta_std"]):
                return ""
            return f"{row['beta_std']: .3f}{row['sig']}"

        disp.loc[is_pred, "value"] = disp.loc[is_pred].apply(fmt_beta, axis=1)

        # Constant: raw intercept (always from fitted model; if model exists)
        disp.loc[is_const, "value"] = disp.loc[is_const, "coef_raw"].map(lambda x: "" if pd.isna(x) else f"{x: .3f}")

        pivot = disp.pivot(index="term", columns="model", values="value")

        idx = [t for t in order_terms if t in pivot.index]
        if "const" in pivot.index:
            idx.append("const")
        extras = [t for t in pivot.index.tolist() if t not in idx]
        pivot = pivot.reindex(idx + extras)
        return pivot, long

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    order_terms = x_m3  # desired overall ordering across models
    table1, long_all = table_from_tabs([tab1, tab2, tab3], model_names, order_terms)

    fitstats = pd.DataFrame([fit1, fit2, fit3])

    # -----------------------------
    # 8) Missingness report (after DV construction)
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
    summary_lines.append("DV: Musical exclusiveness = count (0-18) of genres disliked (4/5) across 18 items;")
    summary_lines.append("    listwise complete across all 18 genre ratings (DK/missing excluded).")
    summary_lines.append("")
    summary_lines.append("OLS on raw DV; predictors displayed as standardized betas computed as beta = b * sd(x)/sd(y)")
    summary_lines.append("using the estimation sample for each model. Stars from raw OLS two-tailed p-values.")
    summary_lines.append("Stars: * p<.05, ** p<.01, *** p<.001")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fitstats[["model", "N", "R2", "Adj_R2", "dropped_predictors"]].to_string(index=False))
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
    summary_lines.append("Raw OLS summaries:")
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