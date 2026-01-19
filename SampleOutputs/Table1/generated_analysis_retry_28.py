def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS missing-value codes across many variables. We treat them as missing everywhere.
    GSS_NA_CODES = {
        0, 7, 8, 9,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999
    }

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s, extra_na=()):
        """Convert to numeric; set common GSS NA codes + extra_na to NaN."""
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

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas_from_unstd(res, y, X):
        """
        Compute standardized betas from an unstandardized OLS fit:
        beta_j = b_j * SD(x_j) / SD(y), using SDs from the estimation sample.
        """
        sdy = sample_sd(y)
        betas = {}
        for c in X.columns:
            b = float(res.params.get(c, np.nan))
            sdx = sample_sd(X[c])
            betas[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return betas

    def safe_write(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_model(df, dv, xcols, model_name, labels):
        """
        Model-specific listwise deletion ONLY on dv + xcols.
        Do not drop predictors just because they are dummies; only drop if they are truly zero-variance.
        """
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Identify predictors with no variance in this model sample; keep them in output but not in design matrix.
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

        rows = []
        if len(frame) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(res, y, X)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        # Constant (unstandardized) + p (for reference), but do not star constant.
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""
        })

        # Preserve requested order; show NaNs if dropped due to zero variance.
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
        """Constant: unstandardized b. Predictors: standardized beta with stars."""
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    # ----------------------------
    # Read + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: number of music genres disliked (0–18)
    # 18 items; dislike if 4/5; DK/NA -> missing; if any item missing => DV missing.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music columns: {missing_music}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv_desc = df["num_genres_disliked"].describe()
    safe_write(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genres where response in {4=dislike, 5=dislike very much}.\n"
        "Missing rule: any missing/DK/NA on any of the 18 items => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics / group identities
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race (White reference)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC when available. Do NOT turn non-Hispanic into missing.
    # In many GSS extracts: ETHNIC is a multi-category origin measure; we treat code==1 as "not Hispanic",
    # and codes 2..(reasonable upper bound) as Hispanic-origin. This yields 0/1 with broad coverage.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Keep only positive codes; if extreme codes exist (e.g., 97/98/99) they are already NaN via clean_gss.
        # If eth==1 clearly indicates non-Hispanic in this extract, use it as 0; otherwise best-effort still.
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (best-effort with available variables):
    # Protestant (RELIG==1) and denomination in {Baptist(1), Other Protestant(6)}.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, treat as not conservative protestant (avoid dropping many cases).
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Region -> South (mapping instruction: REGION==3)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators; if any item missing => scale missing
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    missing_tol = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # keep plausible response codes; otherwise missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness (overall, not model-specific)
    # ----------------------------
    diag_vars = [
        "num_genres_disliked",
        "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss = int(df[v].isna().sum())
        tot = nonmiss + miss
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / tot * 100.0) if tot else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    safe_write("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

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

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save outputs (human-readable)
    safe_write("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    safe_write("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    safe_write("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    safe_write("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    safe_write("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    safe_write("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    safe_write("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # A compact combined summary text
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from microdata; predictors reported as standardized betas; constant unstandardized)")
    summary_lines.append("")
    summary_lines.append("FIT STATS")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("MODEL 1 (Table1-style)")
    summary_lines.append(t1.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("MODEL 2 (Table1-style)")
    summary_lines.append(t2.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("MODEL 3 (Table1-style)")
    summary_lines.append(t3.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("NOTE: Standard errors are not printed in Table 1-style outputs; stars are based on two-tailed p-values from OLS.")

    safe_write("./output/table1_summary.txt", "\n".join(summary_lines) + "\n")

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "missingness": missingness
    }