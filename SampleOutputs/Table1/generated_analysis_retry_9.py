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

    # Conservative GSS missing-code handler: only mask very common DK/NA/refused style sentinels.
    # Do NOT mask "0" generically (it can be valid for some fields).
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

    # -------------------------
    # 1) Restrict to 1993
    # -------------------------
    if "year" not in df.columns:
        raise ValueError("Column 'year' is required.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # 2) Required variables for Table 1 replication (based on provided file)
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
    base_cols = ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]

    missing_cols = [c for c in (music_items + tol_items + base_cols + ["year"]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(set(missing_cols))}")

    # -------------------------
    # 3) Clean variables (minimal, faithful; avoid accidental NA creation)
    # -------------------------
    # Music: valid substantive is 1..5; others -> NaN
    for c in music_items:
        s = mask_gss_missing(df[c])
        df[c] = s.where(s.isin([1, 2, 3, 4, 5]), np.nan)

    # Core continuous
    df["educ"] = mask_gss_missing(df["educ"])
    # keep plausible years, but do not over-trim (paper uses years as coded)
    df["educ"] = df["educ"].where(df["educ"].between(0, 25), np.nan)

    df["realinc"] = mask_gss_missing(df["realinc"])
    # allow 0? realinc=0 can be legitimate; do not force missing.
    df["realinc"] = df["realinc"].where(df["realinc"].between(0, 9_000_000), np.nan)

    df["hompop"] = mask_gss_missing(df["hompop"])
    # hompop should be >=1; set 0/negative missing to avoid division
    df["hompop"] = df["hompop"].where(df["hompop"].between(1, 50), np.nan)

    df["prestg80"] = mask_gss_missing(df["prestg80"])
    df["prestg80"] = df["prestg80"].where(df["prestg80"].between(0, 100), np.nan)

    # Demographics
    df["sex"] = mask_gss_missing(df["sex"]).where(mask_gss_missing(df["sex"]).isin([1, 2]), np.nan)
    df["age"] = mask_gss_missing(df["age"])
    df["age"] = df["age"].where(df["age"].between(18, 89), np.nan)

    df["race"] = mask_gss_missing(df["race"]).where(mask_gss_missing(df["race"]).isin([1, 2, 3]), np.nan)
    df["relig"] = mask_gss_missing(df["relig"]).where(mask_gss_missing(df["relig"]).isin([1, 2, 3, 4, 5]), np.nan)

    # denom is a recode; accept small integer categories if present; otherwise keep numeric as-is but mask missing codes.
    df["denom"] = mask_gss_missing(df["denom"])
    # region
    df["region"] = mask_gss_missing(df["region"]).where(mask_gss_missing(df["region"]).isin([1, 2, 3, 4]), np.nan)

    # ethnic (used to construct hispanic)
    df["ethnic"] = mask_gss_missing(df["ethnic"])
    # do not force a tight valid set; just keep numeric and rely on == checks

    # Tolerance: keep only legitimate substantive codes per item type
    for c in tol_items:
        s = mask_gss_missing(df[c])
        if c.startswith("spk"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            if c == "colcom":
                s = s.where(s.isin([4, 5]), np.nan)  # 4 fired (intolerant), 5 not fired
            else:
                s = s.where(s.isin([4, 5]), np.nan)  # 4 allowed, 5 not allowed (intolerant)
        df[c] = s

    # -------------------------
    # 4) DV: Musical exclusiveness (strict complete across all 18 items, per instructions)
    # -------------------------
    d_music = df.dropna(subset=music_items).copy()
    for c in music_items:
        d_music[f"dislike_{c}"] = d_music[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d_music["exclusiveness"] = d_music[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # 5) IV construction (on the DV-complete sample; model-specific listwise is handled later)
    # -------------------------
    d = d_music

    # Income per capita
    d["inc_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "inc_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["inc_pc"] = d["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic from ETHNIC (available in provided dataset):
    # In GSS, ETHNIC=1 commonly corresponds to "Mexican, Mexican-American, Chicano" and related Hispanic origins.
    # This is the only implementable, non-fabricated option from the supplied columns.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    d["conserv_prot"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["denom"].notna()
    d.loc[m_prot, "conserv_prot"] = d.loc[m_prot, "denom"].isin([1, 6, 7]).astype(float)

    # South
    d["south"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance scale: sum across 15 intolerance indicators.
    # Missing rule: allow partial completion if respondent answered at least 12 of 15 items,
    # then prorate to 15 and round to nearest integer.
    # This avoids excessive N-loss and matches typical scale-handling when split-ballot exists.
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
    n_ans = intoler.notna().sum(axis=1)
    sum_raw = intoler.sum(axis=1, min_count=1)

    d["polintol"] = np.nan
    # require at least 12 answered; prorate to 15; clamp to [0,15]
    m_ok = n_ans >= 12
    d.loc[m_ok, "polintol"] = (sum_raw.loc[m_ok] * (15.0 / n_ans.loc[m_ok])).round()
    d["polintol"] = d["polintol"].clip(lower=0, upper=15)

    # -------------------------
    # 6) Fit models with model-specific listwise deletion ONLY on variables in that model
    #    Report standardized betas (computed on estimation sample) + stars from raw p-values.
    # -------------------------
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "hispanic", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    def fit_model(data, yvar, xvars, name):
        cols = [yvar] + xvars
        dd = data.loc[:, cols].copy().dropna(how="any")

        # Handle any zero-variance predictors in the estimation sample
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
            if not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0:
                betas[v] = np.nan
            else:
                betas[v] = b * (sd_x / sd_y)

        tab = pd.DataFrame({
            "term": xvars,
            "beta_std": [betas.get(v, np.nan) for v in xvars],
            "p_raw": [model.pvalues.get(v, np.nan) for v in xvars],
            "included": [v in x_keep for v in xvars],
        })
        tab["sig"] = tab["p_raw"].map(stars)

        fit = {
            "model": name,
            "N": int(dd.shape[0]),
            "R2": float(model.rsquared),
            "Adj_R2": float(model.rsquared_adj),
            "const": float(model.params.get("const", np.nan)),
            "dropped_no_variance": ", ".join(dropped) if dropped else ""
        }
        return model, tab, fit, dd

    m1, tab1, fit1, d1 = fit_model(d, y, x_m1, "Model 1 (SES)")
    m2, tab2, fit2, d2 = fit_model(d, y, x_m2, "Model 2 (Demographic)")
    m3, tab3, fit3, d3 = fit_model(d, y, x_m3, "Model 3 (Political intolerance)")

    # -------------------------
    # 7) Table 1-style matrix: one row per predictor; cells = standardized beta + stars
    # -------------------------
    def table1_matrix(tabs, model_names, row_order):
        blocks = []
        for t, nm in zip(tabs, model_names):
            tt = t.copy()
            tt["model"] = nm
            blocks.append(tt)
        long = pd.concat(blocks, ignore_index=True)

        def fmt(beta, sig, included):
            if (not included) or pd.isna(beta):
                return ""
            return f"{beta:.3f}{sig}"

        long["cell"] = [fmt(b, s, inc) for b, s, inc in zip(long["beta_std"], long["sig"], long["included"])]
        wide = long.pivot(index="term", columns="model", values="cell")

        # enforce order
        idx = [r for r in row_order if r in wide.index]
        wide = wide.reindex(idx)
        return wide, long

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    row_order = x_m3
    table1, coef_long = table1_matrix([tab1, tab2, tab3], model_names, row_order)

    fitstats = pd.DataFrame([fit1, fit2, fit3])

    # Diagnostics: counts that help reconcile sample sizes
    n_dv_complete_music = int(d.shape[0])
    # Polintol non-missing among DV-complete
    n_polintol_nonmissing = int(d["polintol"].notna().sum())
    # Polintol complete (all 15) if needed for comparison
    n_polintol_complete15 = int(intoler.notna().all(axis=1).sum())

    # -------------------------
    # 8) Save outputs (human-readable)
    # -------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append("DV: musical exclusiveness = count (0–18) of 18 genres rated 4/5 (dislike / dislike very much).")
    lines.append("DV scoring: requires non-missing on all 18 music items (listwise across the 18).")
    lines.append("")
    lines.append("Models: OLS; coefficients shown are standardized betas (beta = b * SD(x)/SD(y)) computed on each model's estimation sample.")
    lines.append("Stars from raw OLS two-tailed p-values: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Sample counts (DV-complete base):")
    lines.append(f"  N with complete music module (DV available): {n_dv_complete_music}")
    lines.append(f"  N with polintol available (>=12/15 answered, prorated): {n_polintol_nonmissing}")
    lines.append(f"  N with complete 15/15 tolerance items (strict): {n_polintol_complete15}")
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fitstats[["model", "N", "R2", "Adj_R2", "const", "dropped_no_variance"]].to_string(index=False))
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
        f.write(fitstats[["model", "N", "R2", "Adj_R2", "const", "dropped_no_variance"]].to_string(index=False))
        f.write("\n")

    coef_long_out = coef_long.loc[:, ["model", "term", "included", "beta_std", "p_raw", "sig"]].copy()
    coef_long_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    table1.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)

    return {
        "fit_stats": fitstats,
        "table1_style_betas": table1,
        "coefficients_long": coef_long_out,
        "n_dv_complete_music": n_dv_complete_music,
        "n_polintol_nonmissing_prorated": n_polintol_nonmissing,
        "n_polintol_complete15_strict": n_polintol_complete15,
    }