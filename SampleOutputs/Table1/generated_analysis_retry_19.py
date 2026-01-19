def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def gss_na_to_nan(x):
        """
        Convert common GSS-style missing codes to NaN.
        Only replaces if those codes occur (safe for continuous vars).
        """
        x = to_num(x)
        na_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}
        return x.where(~x.isin(list(na_codes)), np.nan)

    def star_from_p(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def safe_sd(s):
        s = pd.to_numeric(s, errors="coerce")
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return np.nan
        return float(sd)

    def standardized_betas_from_unstd(y, X, params):
        """
        Compute standardized betas from unstandardized slopes:
          beta_j = b_j * sd(x_j) / sd(y)
        y: Series
        X: DataFrame WITHOUT constant
        params: Series from statsmodels results (includes 'const' and columns of X)
        """
        sd_y = safe_sd(y)
        out = {}
        for c in X.columns:
            b = params.get(c, np.nan)
            sd_x = safe_sd(X[c])
            if pd.isna(b) or pd.isna(sd_x) or pd.isna(sd_y):
                out[c] = np.nan
            else:
                out[c] = float(b) * (sd_x / sd_y)
        return out

    def fit_ols_table1_style(df, dv, x_cols, label_map, model_name):
        """
        Model-specific listwise deletion on dv + x_cols.
        Fit unstandardized OLS (for intercept), compute standardized betas for predictors.
        Return:
          - coef table with Constant (b) and predictors (beta, stars)
          - fit stats dict
          - model frame used
        """
        needed = [dv] + x_cols
        d = df[needed].copy()

        # model-specific listwise deletion
        d = d.dropna(axis=0, how="any").copy()

        # if empty, return shells
        if d.shape[0] == 0:
            rows = [{"term": "Constant", "value": ""}]
            for c in x_cols:
                rows.append({"term": label_map.get(c, c), "value": ""})
            tab = pd.DataFrame(rows)
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            return tab, fit, d

        y = d[dv].astype(float)
        X = d[x_cols].astype(float)

        # Drop zero-variance predictors (rare but can happen)
        kept = [c for c in x_cols if X[c].nunique(dropna=True) > 1]
        dropped = [c for c in x_cols if c not in kept]
        Xk = X[kept].copy()

        if Xk.shape[1] == 0:
            rows = [{"term": "Constant", "value": ""}]
            for c in x_cols:
                rows.append({"term": label_map.get(c, c), "value": ""})
            tab = pd.DataFrame(rows)
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            return tab, fit, d.iloc[0:0].copy()

        Xc = sm.add_constant(Xk, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(y, Xk, res.params)

        rows = []
        # Constant: unstandardized intercept, no stars (to match table style)
        const_b = res.params.get("const", np.nan)
        const_str = "" if pd.isna(const_b) else f"{float(const_b):.3f}"
        rows.append({"term": "Constant", "value": const_str})

        # Predictors: standardized beta + stars based on p-values from unstandardized model
        for c in x_cols:
            lab = label_map.get(c, c)
            if c in kept:
                beta = betas.get(c, np.nan)
                p = res.pvalues.get(c, np.nan)
                if pd.isna(beta):
                    rows.append({"term": lab, "value": ""})
                else:
                    rows.append({"term": lab, "value": f"{float(beta):.3f}{star_from_p(p)}"})
            else:
                # dropped for zero variance
                rows.append({"term": lab, "value": ""})

        tab = pd.DataFrame(rows)
        fit = {
            "model": model_name,
            "n": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "dropped_predictors": dropped,
        }
        return tab, fit, d

    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0-18), listwise across 18 items
    # disliked = 4 or 5; valid = 1..5; other/missing codes -> NaN
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: gss_na_to_nan(df[c]) for c in music_items})
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float)
    disliked = disliked.where(music.notna(), np.nan)

    # listwise across 18 items for DV construction
    df["num_genres_disliked"] = disliked.sum(axis=1, min_count=len(music_items))
    df.loc[music.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Count of 18 genres where response is 4 ('dislike') or 5 ('dislike very much').\n"
        "Valid responses are 1..5; GSS missing/DK/etc treated as missing.\n"
        "Listwise across all 18 items.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # IV construction
    # ----------------------------
    # SES
    df["educ_yrs"] = gss_na_to_nan(df.get("educ", np.nan))
    df.loc[df["educ_yrs"] == 0, "educ_yrs"] = np.nan

    df["prestg80_v"] = gss_na_to_nan(df.get("prestg80", np.nan))
    df.loc[df["prestg80_v"] == 0, "prestg80_v"] = np.nan

    df["realinc_v"] = gss_na_to_nan(df.get("realinc", np.nan))
    df.loc[df["realinc_v"] == 0, "realinc_v"] = np.nan

    df["hompop_v"] = gss_na_to_nan(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan

    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    df.loc[df["inc_pc"] <= 0, "inc_pc"] = np.nan

    # Demographics
    sex = gss_na_to_nan(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = gss_na_to_nan(df.get("age", np.nan))
    df.loc[df["age_v"] == 0, "age_v"] = np.nan

    race = gss_na_to_nan(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: best-available in this extract is "ethnic".
    # Do NOT create a mostly-missing dummy: create 0/1 for all non-missing ethnic values.
    # We treat known "not Hispanic" codes as 0, and Hispanic-related ancestry codes as 1.
    # If we cannot identify, default to 0 for non-missing to avoid collapsing N (paper has large N).
    hisp_codes = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17}  # common GSS ETHNIC categories for Hispanic origins
    eth = gss_na_to_nan(df.get("ethnic", np.nan))
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        # For non-missing eth, set hispanic=1 if code in hisp_codes else 0
        df["hispanic"] = np.where(eth.isna(), np.nan, eth.isin(list(hisp_codes)).astype(float))

    # Religion dummies
    relig = gss_na_to_nan(df.get("relig", np.nan))
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy based on RELIG and DENOM (limited extract)
    denom = gss_na_to_nan(df.get("denom", np.nan))
    # keep denom if it looks like standard small code; otherwise allow missing
    denom = denom.where((denom >= 0) & (denom <= 20), np.nan)
    is_prot = (relig == 1)
    # Conservative proxy: Baptist (often 1) + "other Protestant" (often 6)
    denom_cons = denom.isin([1, 6])
    # If not Protestant, cons_prot = 0 (not missing). If Protestant but denom missing, set missing.
    df["cons_prot"] = np.nan
    df.loc[relig.notna() & (~is_prot), "cons_prot"] = 0.0
    df.loc[relig.notna() & is_prot & denom.notna(), "cons_prot"] = (denom_cons[relig.notna() & is_prot & denom.notna()]).astype(float).values

    # Southern
    region = gss_na_to_nan(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale 0-15
    tol_map = {
        # Anti-religionist
        "spkath": ("spk", 2),
        "colath": ("col", 5),
        "libath": ("lib", 1),
        # Racist
        "spkrac": ("spk", 2),
        "colrac": ("col", 5),
        "librac": ("lib", 1),
        # Communist
        "spkcom": ("spk", 2),
        "colcom": ("colcom", 4),  # special coding in prompt
        "libcom": ("lib", 1),
        # Military-rule advocate
        "spkmil": ("spk", 2),
        "colmil": ("col", 5),
        "libmil": ("lib", 1),
        # Homosexual
        "spkhomo": ("spk", 2),
        "colhomo": ("col", 5),
        "libhomo": ("lib", 1),
    }
    tol_cols = list(tol_map.keys())
    missing_tol = [c for c in tol_cols if c not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance items: {missing_tol}")

    tol = pd.DataFrame({c: gss_na_to_nan(df[c]) for c in tol_cols})

    intolerant_indicators = []
    for c, (_kind, intolerant_code) in tol_map.items():
        x = tol[c]
        ind = np.where(x.isna(), np.nan, (x == intolerant_code).astype(float))
        intolerant_indicators.append(pd.Series(ind, index=df.index, name=c))

    intolerant_df = pd.concat(intolerant_indicators, axis=1)

    # Listwise across 15 items for the scale (respondents not asked will be missing and drop in Model 3)
    df["pol_intol"] = intolerant_df.sum(axis=1, min_count=len(tol_cols))
    df.loc[intolerant_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Missingness diagnostics (constructed variables)
    # ----------------------------
    diag_vars = [
        "num_genres_disliked",
        "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol"
    ]
    miss = []
    for v in diag_vars:
        if v in df.columns:
            nm = int(df[v].notna().sum())
            m = int(df[v].isna().sum())
            miss.append({"variable": v, "nonmissing": nm, "missing": m, "pct_missing": 100.0 * m / max(nm + m, 1)})
    miss_df = pd.DataFrame(miss).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", miss_df.to_string(index=False) + "\n")

    # ----------------------------
    # Fit the three models (Table 1)
    # ----------------------------
    labels = {
        "educ_yrs": "Education (years)",
        "inc_pc": "Household income per capita",
        "prestg80_v": "Occupational prestige",
        "female": "Female",
        "age_v": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "otherrace": "Other race",
        "cons_prot": "Conservative Protestant",
        "norelig": "No religion",
        "south": "Southern",
        "pol_intol": "Political intolerance (0–15)",
    }
    dv = "num_genres_disliked"

    m1_x = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2_x = m1_x + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    tab1, fit1, frame1 = fit_ols_table1_style(df, dv, m1_x, labels, "Model 1 (SES)")
    tab2, fit2, frame2 = fit_ols_table1_style(df, dv, m2_x, labels, "Model 2 (Demographic)")
    tab3, fit3, frame3 = fit_ols_table1_style(df, dv, m3_x, labels, "Model 3 (Political intolerance)")

    # ----------------------------
    # Write human-readable outputs (Table 1 style: β + stars; constant as b)
    # ----------------------------
    def format_block(tab, fit):
        lines = []
        lines.append(f"{fit['model']}")
        lines.append("=" * len(fit["model"]))
        lines.append(f"N = {fit['n']}")
        lines.append(f"R^2 = {fit['r2']:.3f}" if pd.notna(fit["r2"]) else "R^2 = NA")
        lines.append(f"Adj R^2 = {fit['adj_r2']:.3f}" if pd.notna(fit["adj_r2"]) else "Adj R^2 = NA")
        dropped = fit.get("dropped_predictors", [])
        if dropped:
            lines.append(f"Dropped (zero variance): {', '.join(dropped)}")
        lines.append("")
        lines.append("Coefficients (Table 1 style):")
        lines.append("- Constant shown as unstandardized intercept (b)")
        lines.append("- Predictors shown as standardized betas (β) with stars")
        lines.append("- Stars: * p<.05, ** p<.01, *** p<.001")
        lines.append("")
        lines.append(tab.to_string(index=False))
        lines.append("")
        return "\n".join(lines)

    write_text("./output/table1_model1_ses.txt", format_block(tab1, fit1))
    write_text("./output/table1_model2_demographic.txt", format_block(tab2, fit2))
    write_text("./output/table1_model3_political_intolerance.txt", format_block(tab3, fit3))

    # Compact combined summary
    fit_stats = pd.DataFrame([
        {"model": fit1["model"], "n": fit1["n"], "r2": fit1["r2"], "adj_r2": fit1["adj_r2"]},
        {"model": fit2["model"], "n": fit2["n"], "r2": fit2["r2"], "adj_r2": fit2["adj_r2"]},
        {"model": fit3["model"], "n": fit3["n"], "r2": fit3["r2"], "adj_r2": fit3["adj_r2"]},
    ])
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Return a dict of results for programmatic checking
    return {
        "fit_stats": fit_stats,
        "Model 1 (SES)": tab1,
        "Model 2 (Demographic)": tab2,
        "Model 3 (Political intolerance)": tab3,
        "missingness": miss_df,
        "n_model_frames": {
            "Model 1 (SES)": int(frame1.shape[0]),
            "Model 2 (Demographic)": int(frame2.shape[0]),
            "Model 3 (Political intolerance)": int(frame3.shape[0]),
        },
    }