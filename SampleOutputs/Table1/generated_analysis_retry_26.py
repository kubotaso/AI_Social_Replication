def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    GSS_NA_CODES = {
        0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999
    }

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_gss(x, extra_na=()):
        s = to_num(x)
        na = set(GSS_NA_CODES) | set(extra_na)
        return s.where(~s.isin(list(na)), np.nan)

    def sd_sample(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        return np.nan if (pd.isna(v) or v <= 0) else float(np.sqrt(v))

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

    def standardized_betas_from_fit(y, X_no_const, params):
        # beta_j = b_j * sd(x_j) / sd(y), computed on the estimation sample
        sdy = sd_sample(y)
        betas = {}
        for c in X_no_const.columns:
            b = float(params.get(c, np.nan))
            sdx = sd_sample(X_no_const[c])
            betas[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return betas

    def fit_ols_table1(df, dv, xcols, model_name, term_labels):
        # model-specific listwise deletion ONLY on dv and xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # if any predictor has zero variance after deletion, drop it (avoid singular)
        kept = []
        dropped = []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        out_rows = []
        meta = {"model": model_name, "n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan, "dropped": dropped}

        if len(frame) == 0 or len(kept) == 0:
            # build empty-like table
            out_rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                out_rows.append({"term": term_labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(out_rows)
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_fit(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        # Table 1 style: constant unstandardized; predictors standardized betas + stars
        out_rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant (paper convention)
        })

        for c in xcols:
            lab = term_labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                out_rows.append({
                    "term": lab,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                out_rows.append({"term": lab, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(out_rows)
        return meta, tab, frame

    def to_table1_display(tab):
        # constant: show b (no stars); predictors: show beta + stars
        disp = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                disp.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                disp.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": disp})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read and restrict year
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column in the dataset.")
    df = df.loc[clean_gss(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0-18)
    # Disliked = 4 or 5; valid substantive = 1..5; DK etc => missing.
    # IMPORTANT: DV is missing if ANY of 18 items is missing (listwise across items).
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    miss_music = [c for c in music_items if c not in df.columns]
    if miss_music:
        raise ValueError(f"Missing required music items: {miss_music}")

    music = pd.DataFrame({c: clean_gss(df[c]) for c in music_items})
    # Only 1..5 are valid; everything else => missing
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv_desc = df["num_genres_disliked"].describe()
    dv_stats_txt = (
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4 ('dislike') or 5 ('dislike very much');\n"
        "valid responses are 1..5; other codes treated as missing; if any of 18 items missing => DV missing.\n\n"
        f"{dv_desc.to_string()}\n"
    )
    write_text("./output/table1_dv_descriptives.txt", dv_stats_txt)

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan), extra_na=())
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan), extra_na=())

    # Income per capita: REALINC / HOMPOP (do NOT drop <=0 unless clearly invalid)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan), extra_na=())
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan), extra_na=())
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan), extra_na=())
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan), extra_na=())
    # keep top-coded 89 as 89; just ensure nonpositive invalid are missing
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss(df.get("race", np.nan), extra_na=())
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=White,2=Black,3=Other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC as provided in this extract.
    # Make it non-missing whenever ETHNIC is present:
    # Common GSS coding: 1=not hispanic, 2=hispanic. If not binary, treat 1..9 as Hispanic-origin categories.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"], extra_na=())
        # binary yes/no pattern
        uniq = sorted(pd.unique(eth.dropna()))
        if len(uniq) > 0 and set(uniq).issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # fallback: classify 1..9 as Hispanic-origin codes; others => not Hispanic (0) when known
            # This avoids turning most respondents into NA, which collapses Model 2/3.
            df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 1) & (eth <= 9)).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan), extra_na=())
    # RELIG substantive: 1=Protestant,2=Catholic,3=Jewish,4=None,5=Other
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan), extra_na=())
    # Conservative Protestant approximation using RELIG+DENOM without inducing missingness:
    # Mark as 1 for Protestants in denominations commonly coded as Baptist or Other Protestant.
    # If Protestant but denom missing, treat as 0 (unknown/other) to preserve sample (paper did not show huge missingness here).
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # 1=Baptist, 6=Other Protestant (common GSS coding)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South
    region = clean_gss(df.get("region", np.nan), extra_na=())
    # REGION: 1=New England,2=Mid-Atl,3=E.N.C.,4=W.N.C.,5=South Atl,6=E.S.C.,7=W.S.C.,8=Mountain,9=Pacific in some codings;
    # In this extract mapping says REGION==3 is South. We'll follow the provided mapping instruction exactly.
    # But also allow common 1..9; code south=1 if region==3, else 0 for other known values.
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum of 15 binary "intolerant" items.
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    miss_tol = [v for v, _ in tol_items if v not in df.columns]
    if miss_tol:
        raise ValueError(f"Missing required political tolerance items: {miss_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for v, bad in tol_items:
        x = clean_gss(df[v], extra_na=())
        # retain only plausible substantive codes; if outside expected small integers, set missing
        # (keeps listwise behavior similar to GSS coding)
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[v] = np.where(x.isna(), np.nan, x.isin(list(bad)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    # listwise for index: require all 15 items non-missing
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Model definitions (Table 1)
    # ----------------------------
    term_labels = {
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
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    # ----------------------------
    # Fit models
    # ----------------------------
    meta1, tab1, frame1 = fit_ols_table1(df, dv, m1, "Model 1 (SES)", term_labels)
    meta2, tab2, frame2 = fit_ols_table1(df, dv, m2, "Model 2 (Demographic)", term_labels)
    meta3, tab3, frame3 = fit_ols_table1(df, dv, m3, "Model 3 (Political intolerance)", term_labels)

    fit_stats = pd.DataFrame([
        {"model": meta1["model"], "n": meta1["n"], "r2": meta1["r2"], "adj_r2": meta1["adj_r2"], "dropped": ", ".join(meta1["dropped"])},
        {"model": meta2["model"], "n": meta2["n"], "r2": meta2["r2"], "adj_r2": meta2["adj_r2"], "dropped": ", ".join(meta2["dropped"])},
        {"model": meta3["model"], "n": meta3["n"], "r2": meta3["r2"], "adj_r2": meta3["adj_r2"], "dropped": ", ".join(meta3["dropped"])},
    ])

    # ----------------------------
    # Diagnostics: missingness (1993 only)
    # ----------------------------
    diag_vars = [
        dv, "educ_yrs", "inc_pc", "prestg80_v", "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonm = int(df[v].notna().sum())
        mis = int(df[v].isna().sum())
        miss_rows.append({"variable": v, "nonmissing": nonm, "missing": mis, "pct_missing": (mis / max(1, len(df))) * 100.0})
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)

    # ----------------------------
    # Write outputs
    # ----------------------------
    # Full tables (b, beta, p, stars)
    def tab_to_text(tab):
        return tab.to_string(index=False, justify="left", col_space=2)

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False))
    write_text("./output/model1_full.txt", tab_to_text(tab1))
    write_text("./output/model2_full.txt", tab_to_text(tab2))
    write_text("./output/model3_full.txt", tab_to_text(tab3))

    # Table1-style display (constant b; predictors beta+stars)
    t1 = to_table1_display(tab1)
    t2 = to_table1_display(tab2)
    t3 = to_table1_display(tab3)
    write_text("./output/model1_table1_style.txt", t1.to_string(index=False))
    write_text("./output/model2_table1_style.txt", t2.to_string(index=False))
    write_text("./output/model3_table1_style.txt", t3.to_string(index=False))

    write_text("./output/missingness_1993.txt", missingness.to_string(index=False))

    # Extra: show estimation-sample DV mean/sd for each model (helps debug standardization/sample drift)
    def sample_desc(frame, label):
        s = frame[dv]
        return (
            f"{label}\n"
            f"  N={len(frame)}\n"
            f"  DV mean={float(s.mean()):.4f}  sd={float(s.std(ddof=1)):.4f}  min={float(s.min()):.1f}  max={float(s.max()):.1f}\n"
        )

    summary = []
    summary.append("Table 1 replication (computed from microdata; predictors reported as standardized betas; constant unstandardized)\n")
    summary.append(fit_stats.to_string(index=False))
    summary.append("\n\nEstimation-sample DV summaries:\n")
    summary.append(sample_desc(frame1, meta1["model"]))
    summary.append(sample_desc(frame2, meta2["model"]))
    summary.append(sample_desc(frame3, meta3["model"]))
    summary.append("\nNotes:\n- Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed).\n- Standard errors are not shown in the table-style outputs.\n")
    write_text("./output/summary.txt", "\n".join(summary))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1": t1,
        "model2_table1": t2,
        "model3_table1": t3,
        "missingness": missingness,
    }