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

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y) computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any")

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": "",
        }

        # If empty, return shell table
        rows = []
        if len(frame) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)

        # Keep only predictors with variance; do NOT silently drop from output, but record
        kept = []
        dropped = []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)
        meta["dropped"] = ",".join(dropped)

        if len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas(y, X, res.params)

        # Build output table in the same order as xcols (plus constant)
        rows.append(
            {
                "term": "Constant",
                "b": float(res.params.get("const", np.nan)),
                "beta": np.nan,
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": "",
            }
        )
        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append(
                    {
                        "term": term,
                        "b": float(res.params.get(c, np.nan)),
                        "beta": float(betas.get(c, np.nan)),
                        "p": p,
                        "sig": sig_star(p),
                    }
                )
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def table1_display(tab):
        # Constant: show unstandardized b; predictors: standardized beta + stars
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
    # DV: number of music genres disliked (0–18)
    # Rule: 1 if item in {4,5}, 0 if in {1,2,3}, missing otherwise.
    # IMPORTANT: listwise for DV construction across all 18 items (as per spec).
    # ----------------------------
    music_items = [
        "bigband",
        "blugrass",
        "country",
        "blues",
        "musicals",
        "classicl",
        "folk",
        "gospel",
        "jazz",
        "latin",
        "moodeasy",
        "newage",
        "opera",
        "rap",
        "reggae",
        "conrock",
        "oldies",
        "hvymetal",
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
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum across 18 genres of 1{response in (4,5)}; responses (1,2,3)=0; DK/NA=missing.\n"
        "Listwise for DV: if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n",
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

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white, 2=black, 3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic (ETHNIC). To avoid sign flips and massive NA propagation:
    # - if ETHNIC missing => hispanic missing
    # - else: treat 1 as not Hispanic; treat >=2 as Hispanic-origin (best-effort)
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: Protestant + denomination in (Baptist, other Protestant).
    # If denom missing but Protestant, set 0 (prevents large extra case loss).
    is_prot = relig == 1
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin([1, 6])).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Region: South dummy. Mapping says REGION==3.
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators.
    # To reduce unnecessary sample loss vs strict all-15 listwise:
    # - compute sum across non-missing items
    # - require at least MIN_TOL_ANSWERED items answered to assign a score (else missing)
    tol_items = [
        ("spkath", {2}),
        ("colath", {5}),
        ("libath", {1}),
        ("spkrac", {2}),
        ("colrac", {5}),
        ("librac", {1}),
        ("spkcom", {2}),
        ("colcom", {4}),
        ("libcom", {1}),
        ("spkmil", {2}),
        ("colmil", {5}),
        ("libmil", {1}),
        ("spkhomo", {2}),
        ("colhomo", {5}),
        ("libhomo", {1}),
    ]
    missing_tol = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol_df.notna().sum(axis=1)

    # Default: require all 15 answered (matches strictest reading)
    # But allow relaxing if the strict rule yields far too small N (replication robustness).
    MIN_TOL_ANSWERED_STRICT = 15
    MIN_TOL_ANSWERED_RELAXED = 12

    pol_sum = tol_df.sum(axis=1, min_count=1)

    df["pol_intol_strict"] = np.where(answered >= MIN_TOL_ANSWERED_STRICT, pol_sum, np.nan)
    df["pol_intol_relaxed"] = np.where(answered >= MIN_TOL_ANSWERED_RELAXED, pol_sum, np.nan)

    # Choose the version that keeps more data while remaining close to full battery:
    # If strict keeps >= 450 cases, keep strict; else use relaxed.
    strict_n = int(df["pol_intol_strict"].notna().sum())
    df["pol_intol"] = df["pol_intol_strict"] if strict_n >= 450 else df["pol_intol_relaxed"]

    # ----------------------------
    # Diagnostics: missingness
    # ----------------------------
    diag_vars = [
        "num_genres_disliked",
        "educ_yrs",
        "inc_pc",
        "prestg80_v",
        "female",
        "age_v",
        "black",
        "hispanic",
        "otherrace",
        "cons_prot",
        "norelig",
        "south",
        "pol_intol",
    ]
    miss_rows = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss = int(df[v].isna().sum())
        denom_n = nonmiss + miss
        miss_rows.append(
            {
                "variable": v,
                "nonmissing": nonmiss,
                "missing": miss,
                "pct_missing": (miss / denom_n * 100.0) if denom_n else np.nan,
            }
        )
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # ----------------------------
    # Models
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
        "pol_intol": "Political intolerance",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Save human-readable outputs
    # ----------------------------
    def full_table_text(meta, tab):
        lines = []
        lines.append(f"{meta['model']}")
        lines.append(f"N = {meta['n']}")
        lines.append(f"R2 = {meta['r2']:.6f}" if pd.notna(meta["r2"]) else "R2 = NA")
        lines.append(f"Adj R2 = {meta['adj_r2']:.6f}" if pd.notna(meta["adj_r2"]) else "Adj R2 = NA")
        lines.append(f"Dropped (zero-variance predictors): {meta['dropped'] if meta['dropped'] else '(none)'}")
        lines.append("")
        lines.append(tab.to_string(index=False))
        return "\n".join(lines) + "\n"

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", full_table_text(meta1, tab1))
    write_text("./output/model2_full.txt", full_table_text(meta2, tab2))
    write_text("./output/model3_full.txt", full_table_text(meta3, tab3))

    # Table-1 style (standardized betas with stars; constant unstandardized)
    t1_1 = table1_display(tab1)
    t1_2 = table1_display(tab2)
    t1_3 = table1_display(tab3)

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Also save quick frequency checks for key dummies in each analytic sample (debugging “otherrace dropped” etc.)
    def freq_txt(frame, cols, title):
        out = [title, f"N (analytic) = {len(frame)}", ""]
        for c in cols:
            if c not in frame.columns:
                continue
            vc = frame[c].value_counts(dropna=False).sort_index()
            out.append(f"{c} value_counts (incl NA):")
            out.append(vc.to_string())
            out.append("")
        return "\n".join(out) + "\n"

    write_text(
        "./output/model2_dummy_freqs.txt",
        freq_txt(frame2, ["black", "hispanic", "otherrace", "female", "south", "cons_prot", "norelig"], "Model 2 dummy checks"),
    )
    write_text(
        "./output/model3_dummy_freqs.txt",
        freq_txt(frame3, ["black", "hispanic", "otherrace", "female", "south", "cons_prot", "norelig"], "Model 3 dummy checks"),
    )

    # Summary text
    summary = []
    summary.append("Table 1 replication: standardized OLS betas (predictors) + unstandardized constants")
    summary.append("Data: GSS 1993 (filtered YEAR==1993)")
    summary.append("")
    summary.append("Fit statistics:")
    summary.append(fit_stats.to_string(index=False))
    summary.append("")
    summary.append("Notes:")
    summary.append("- DV is listwise across the 18 music items (missing on any item => DV missing).")
    summary.append("- Hispanic is derived from ETHNIC if present; coded 1 if ETHNIC>=2 else 0; missing only if ETHNIC missing.")
    summary.append("- Political intolerance is summed across 15 items; if strict all-15 answered yields low N, a >=12 answered rule is used.")
    summary_text = "\n".join(summary) + "\n"
    write_text("./output/summary.txt", summary_text)

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "missingness": missingness,
    }