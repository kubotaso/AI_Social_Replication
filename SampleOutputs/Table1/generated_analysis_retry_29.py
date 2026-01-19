def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA_CODES) | set(extra_na)
        return x.where(~x.isin(list(na)), np.nan)

    def sig_star(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def sd1(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas_from_unstd(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on the estimation sample
        sdy = sd1(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sd1(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_ols_and_table(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion on DV + predictors only
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors in this model's estimation sample
        kept, dropped = [], []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ",".join(dropped) if dropped else ""
        }

        # If no data or no predictors, return empty-like shells
        if len(frame) == 0 or len(kept) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_from_unstd(y, X, res.params)

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # constants not starred in Table 1 style
        })

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_style(tab):
        # Paper-like: constant unstandardized; predictors standardized betas + stars; no SEs shown
        out_vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out_vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out_vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out_vals})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: Musical exclusiveness (count of 18 genres disliked)
    # Dislike indicator: 1 if 4/5; 0 if 1/2/3; missing otherwise.
    # DV missing if ANY of the 18 music items is missing.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music_cols = [c for c in music_items if c not in df.columns]
    if missing_music_cols:
        raise ValueError(f"Missing required music columns: {missing_music_cols}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: across 18 music items, count responses 4/5 as disliked.\n"
        "Missing rule: if any of the 18 items is missing/DK/NA => DV set to missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1 mapping)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan)).where(lambda s: s.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race (GSS RACE: 1 white, 2 black, 3 other)
    race = clean_gss(df.get("race", np.nan)).where(lambda s: s.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC, but avoid creating lots of missing.
    # Strategy:
    # - If ETHNIC is missing, set hispanic=0 (so it doesn't annihilate N),
    #   because in many GSS extracts ETHNIC missing often reflects not asked/blank in extract.
    # - If ETHNIC is present with substantive values:
    #     * if binary {1,2}: 2=hispanic
    #     * else: treat code 1 as "not hispanic"; any other positive code as hispanic-origin (best-effort).
    df["hispanic"] = 0.0
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), 0.0, (eth == 2).astype(float))
        else:
            # best-effort for multi-category
            df["hispanic"] = np.where(eth.isna(), 0.0, ((eth >= 2) & (eth <= 99)).astype(float))

    # Religion dummies
    relig = clean_gss(df.get("relig", np.nan)).where(lambda s: s.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: approximation from RELIG==1 and DENOM in {1 Baptist, 6 other Protestant}
    # If Protestant but denom missing, treat as not conservative (0) to preserve sample as in many recodes.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per mapping instruction
    region = clean_gss(df.get("region", np.nan)).where(lambda s: s.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15), but do NOT require complete 15-item data.
    # Use a standard "proportional" scale when at least 12/15 items are answered:
    #   pol_intol = round( (sum intolerant / items_answered) * 15 )
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    missing_tol_cols = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol_cols:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol_cols}")

    tol_ind = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_ind[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    items_answered = tol_ind.notna().sum(axis=1).astype(float)
    sum_intol = tol_ind.sum(axis=1, skipna=True)

    df["pol_intol"] = np.nan
    enough = items_answered >= 12  # keeps those who were asked most of the battery; preserves N vs strict listwise
    df.loc[enough, "pol_intol"] = np.round((sum_intol[enough] / items_answered[enough]) * 15.0)
    df["pol_intol"] = df["pol_intol"].clip(lower=0, upper=15)

    # ----------------------------
    # Diagnostics: missingness
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss = int(df[v].isna().sum())
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # ----------------------------
    # Models (Table 1)
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

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_ols_and_table(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_ols_and_table(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_ols_and_table(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Full tables for auditing (contain b, beta, p, stars)
    model1_full = tab1.copy()
    model2_full = tab2.copy()
    model3_full = tab3.copy()

    # Paper-like display tables
    model1_table1style = table1_style(tab1)
    model2_table1style = table1_style(tab2)
    model3_table1style = table1_style(tab3)

    # ----------------------------
    # Write outputs
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n\n")
    write_text("./output/model1_full.txt", model1_full.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", model2_full.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", model3_full.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", model1_table1style.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", model2_table1style.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", model3_table1style.to_string(index=False) + "\n")

    # One combined human-readable summary
    summary_lines = []
    summary_lines.append("Table 1 replication-style output (computed from microdata)\n")
    summary_lines.append("Notes:")
    summary_lines.append("- Predictors shown as standardized OLS coefficients (beta) with stars.")
    summary_lines.append("- Constant shown as unstandardized intercept from the unstandardized model.")
    summary_lines.append("- Stars from two-tailed p-values: * p<.05, ** p<.01, *** p<.001.")
    summary_lines.append("- Standard errors are not displayed (Table 1 style).\n")

    summary_lines.append("Fit statistics:\n" + fit_stats.to_string(index=False) + "\n")
    summary_lines.append("Model 1 (SES) Table 1 style:\n" + model1_table1style.to_string(index=False) + "\n")
    summary_lines.append("Model 2 (Demographic) Table 1 style:\n" + model2_table1style.to_string(index=False) + "\n")
    summary_lines.append("Model 3 (Political intolerance) Table 1 style:\n" + model3_table1style.to_string(index=False) + "\n")

    write_text("./output/table1_summary.txt", "\n".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": model1_full,
        "model2_full": model2_full,
        "model3_full": model3_full,
        "model1_table1style": model1_table1style,
        "model2_table1style": model2_table1style,
        "model3_table1style": model3_table1style,
        "missingness": missingness,
    }