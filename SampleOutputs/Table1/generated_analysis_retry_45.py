def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Treat common GSS special codes as missing.
    # NOTE: In the provided extract, "ethnic" contains values like 97 that should be missing.
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

    def zscore(series):
        s = pd.to_numeric(series, errors="coerce").astype(float)
        mu = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def safe_write(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_table1_ols(df, dv, xcols, model_name, labels):
        """
        Table 1 wants:
          - predictors: standardized coefficients (beta), i.e., OLS on z(Y) ~ z(X)
          - constant: unstandardized intercept from OLS on raw Y ~ raw X
          - N, R2, adjR2 from the unstandardized model (standard practice)
        """
        # model-specific listwise deletion
        use = df[[dv] + xcols].copy()
        use = use.dropna(axis=0, how="any").copy()

        # drop any zero-variance predictors in this analytic sample
        kept = []
        dropped = []
        for c in xcols:
            if use[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(use)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ",".join(dropped) if dropped else ""
        }

        # Prepare output rows in original table order
        rows = []
        # If no data / no predictors, return shell
        if len(use) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return meta, tab, use

        # Unstandardized model (for constant, R2)
        y = use[dv].astype(float)
        X = use[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res_raw = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res_raw.rsquared)
        meta["adj_r2"] = float(res_raw.rsquared_adj)

        # Standardized model for betas (z(Y) ~ z(X))
        yz = zscore(y)
        Xz = pd.DataFrame({c: zscore(use[c]) for c in kept}, index=use.index)
        Xzc = sm.add_constant(Xz, has_constant="add")
        res_std = sm.OLS(yz, Xzc).fit()

        # Assemble rows: constant from raw model; betas+p from std model
        rows.append({
            "term": "Constant",
            "b": float(res_raw.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res_raw.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        })

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res_std.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": float(res_raw.params.get(c, np.nan)),
                    "beta": float(res_std.params.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        return meta, tab, use

    def table1_style(tab):
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

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
    # DV: number of music genres disliked (0-18)
    # Rule: 1 if response 4 or 5; 0 if 1-3; missing otherwise.
    # Listwise requirement for DV: if any of the 18 items is missing, DV missing.
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

    dv_desc = df["num_genres_disliked"].describe()
    safe_write(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Constructed as sum across 18 genres: 1 if response in {4,5}, 0 if in {1,2,3}; DK/NA->missing.\n"
        "Listwise on the 18 genre items: any missing item => DV missing.\n\n"
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

    # Female
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male, 2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race: black/other from race; missing only if race missing
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: from ETHNIC (available). Keep 0/1 with missing only if ETHNIC missing.
    # Use conservative binary mapping if present; else best-effort: code>=2 as hispanic.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If binary 1/2, treat 2 as Hispanic.
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best-effort: treat 1 as non-Hispanic; codes 2..9 as Hispanic-origin categories
            df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 9)).astype(float))

    # Religion: no religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: Protestant + denomination (approx, but deterministic and low-missing)
    denom = clean_gss(df.get("denom", np.nan))
    # Denom codes in GSS often: 1 Baptist, 2 Methodist, 3 Lutheran, 4 Presbyterian, 5 Episcopal, 6 Other, 7 No denom
    # Use a common "conservative" proxy: Baptist + Other Protestant + No denomination among Protestants
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6, 7])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, set conservative protestant to 0 (avoid unnecessary case loss)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South: REGION==3 per mapping instruction
    region = clean_gss(df.get("region", np.nan))
    # Keep 1..9 as plausible region codes; missing otherwise
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators; require complete on all 15 items
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

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Keep plausible small integer codes only; else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness and key intersections (to catch hidden N loss)
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
    safe_write("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Explicit intersections per model (should match model N if listwise deletion is correct)
    m1_vars = ["num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v"]
    m2_vars = m1_vars + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_vars = m2_vars + ["pol_intol"]

    inter_text = []
    for name, cols in [("Model 1", m1_vars), ("Model 2", m2_vars), ("Model 3", m3_vars)]:
        n_int = int(df[cols].dropna().shape[0])
        inter_text.append(f"{name} complete-case N on {cols}: {n_int}")
    safe_write("./output/table1_intersection_ns.txt", "\n".join(inter_text) + "\n")

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

    meta1, tab1, use1 = fit_table1_ols(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, use2 = fit_table1_ols(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, use3 = fit_table1_ols(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Save human-readable tables
    # ----------------------------
    safe_write("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    safe_write("./output/table1_model1_full.txt", tab1.to_string(index=False) + "\n")
    safe_write("./output/table1_model2_full.txt", tab2.to_string(index=False) + "\n")
    safe_write("./output/table1_model3_full.txt", tab3.to_string(index=False) + "\n")

    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    safe_write("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    safe_write("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    safe_write("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Additional sanity checks for dummy integrity within each model sample
    def dummy_check(use_df, cols, title):
        lines = [title]
        for c in cols:
            if c not in use_df.columns:
                continue
            v = use_df[c]
            lines.append(
                f"{c}: n={int(v.notna().sum())}, mean={float(v.mean()):.4f}, "
                f"min={float(v.min()):.4f}, max={float(v.max()):.4f}, unique={sorted(pd.unique(v))[:10]}"
            )
        return "\n".join(lines) + "\n"

    safe_write(
        "./output/table1_dummy_checks.txt",
        dummy_check(use2, ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"], "Model 2 dummy checks")
        + "\n"
        + dummy_check(use3, ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"], "Model 3 dummy checks")
    )

    # Return results as dict of DataFrames
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