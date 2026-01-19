def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Use common GSS missing codes. (Not all apply to every variable, but safe to treat as missing.)
    GSS_NA_CODES = {
        0, 7, 8, 9,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999
    }

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(series, extra_na=()):
        x = to_num(series)
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

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on the estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on variables in this model.
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors in this analytic sample (to prevent singular fits).
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
            "dropped_predictors": ",".join(dropped) if dropped else ""
        }

        # If nothing to fit, return empty shells
        if len(frame) == 0 or len(kept) == 0:
            terms = ["Constant"] + [labels.get(c, c) for c in xcols]
            tab = pd.DataFrame({"term": terms, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # no stars on constant in Table-1 style
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
        # Match paper presentation: Constant (unstandardized); predictors: standardized beta + stars.
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

    def df_to_text(df, index=False):
        return df.to_string(index=index) + "\n"

    # ----------------------------
    # Read + standardize column names
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower().strip() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: musical exclusiveness (# genres disliked), listwise across 18 items for DV construction
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

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male,2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race dummies: ensure nonmissing race yields 0/1 for each dummy.
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: prefer ETHNIC (available in provided variables).
    # Use binary coding if clearly binary; otherwise treat 2.. as Hispanic-origin (best effort).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        uniq = set(pd.unique(eth.dropna()))
        if uniq.issubset({1.0, 2.0}) and len(uniq) > 0:
            # 1=not hispanic, 2=hispanic
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # best effort: 1 = not hispanic; any other substantive code = hispanic
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: operationalized via RELIG==1 and DENOM in a conservative set.
    # Keep denom missing for Protestants as 0 (so we don't drop them unnecessarily).
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    # Conservative-ish denominations (best effort with DENOM codes available in many GSS extracts):
    # 1=Baptist, 6=Other Protestant, 7=No denom (often evangelical/non-denom), 8=Don't know (already NA)
    denom_cons = denom.isin([1, 6, 7])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South (REGION==3 per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15); require complete battery (listwise across 15 items)
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
        # Keep plausible substantive codes; otherwise missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Model specs
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

    # ----------------------------
    # Outputs: Table-1 style (beta+stars only), plus full diagnostics for debugging
    # ----------------------------
    tab1_t1 = table1_style(tab1)
    tab2_t1 = table1_style(tab2)
    tab3_t1 = table1_style(tab3)

    # DV descriptives (for sanity checks)
    dv_desc = df["num_genres_disliked"].describe()

    # Missingness diagnostics: report on key constructed vars
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    n_total = int(len(df))
    for v in diag_vars:
        if v not in df.columns:
            continue
        n_nonmiss = int(df[v].notna().sum())
        n_miss = int(df[v].isna().sum())
        miss_rows.append({
            "variable": v,
            "n_total": n_total,
            "nonmissing": n_nonmiss,
            "missing": n_miss,
            "pct_missing": (n_miss / n_total * 100.0) if n_total else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)

    # Also: within-model variance checks (to diagnose dropped predictors)
    def variance_check(frame, cols):
        rows = []
        for c in cols:
            if c not in frame.columns:
                continue
            x = frame[c]
            rows.append({
                "var": c,
                "n": int(x.notna().sum()),
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "sd": float(sample_sd(x)) if x.notna().any() else np.nan,
                "min": float(x.min()) if x.notna().any() else np.nan,
                "max": float(x.max()) if x.notna().any() else np.nan,
                "unique_nonmissing": int(x.nunique(dropna=True))
            })
        return pd.DataFrame(rows)

    vc_m2 = variance_check(frame2, m2)
    vc_m3 = variance_check(frame3, m3)

    # ----------------------------
    # Save to ./output as human-readable text
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", df_to_text(fit_stats, index=False))

    write_text("./output/table1_model1_full.txt", df_to_text(tab1, index=False))
    write_text("./output/table1_model2_full.txt", df_to_text(tab2, index=False))
    write_text("./output/table1_model3_full.txt", df_to_text(tab3, index=False))

    write_text("./output/table1_model1_table1style.txt", df_to_text(tab1_t1, index=False))
    write_text("./output/table1_model2_table1style.txt", df_to_text(tab2_t1, index=False))
    write_text("./output/table1_model3_table1style.txt", df_to_text(tab3_t1, index=False))

    write_text("./output/table1_dv_descriptives.txt", "DV: num_genres_disliked (0–18)\n" + dv_desc.to_string() + "\n")
    write_text("./output/table1_missingness.txt", df_to_text(missingness, index=False))
    write_text("./output/table1_variancecheck_model2_sample.txt", df_to_text(vc_m2, index=False))
    write_text("./output/table1_variancecheck_model3_sample.txt", df_to_text(vc_m3, index=False))

    # A compact single-file summary for quick review
    summary_txt = []
    summary_txt.append("Table 1 replication (computed from data; predictors shown as standardized betas with stars)\n")
    summary_txt.append("Fit stats:\n" + fit_stats.to_string(index=False) + "\n\n")
    summary_txt.append("Model 1 (Table-1 style):\n" + tab1_t1.to_string(index=False) + "\n\n")
    summary_txt.append("Model 2 (Table-1 style):\n" + tab2_t1.to_string(index=False) + "\n\n")
    summary_txt.append("Model 3 (Table-1 style):\n" + tab3_t1.to_string(index=False) + "\n\n")
    summary_txt.append("Note: Full tables with b/beta/p are saved separately; Table-1 style omits b/p and shows beta+stars only.\n")
    write_text("./output/table1_summary.txt", "".join(summary_txt))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": tab1_t1,
        "model2_table1style": tab2_t1,
        "model3_table1style": tab3_t1,
        "missingness": missingness,
        "dv_describe": dv_desc.to_frame(name="value")
    }