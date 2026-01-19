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
        0, 7, 8, 9,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999
    }

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
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def safe_nunique(s):
        return int(pd.Series(s).dropna().nunique())

    def fit_model(df, dv, xcols, model_name, labels):
        # Strict model-specific listwise deletion ONLY on dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        dropped = []
        kept = []
        for c in xcols:
            if safe_nunique(frame[c]) <= 1:
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

        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
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

        return meta, pd.DataFrame(rows), frame

    def table1_display(tab):
        # Constant: show unstandardized b; predictors: show standardized beta + stars
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

    def freq_table(series, name):
        s = series.copy()
        return pd.DataFrame({name: s.value_counts(dropna=False)})

    # ----------------------------
    # Read data and restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected 'year' column.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0–18)
    # - dislike indicators: 1 if 4 or 5; 0 if 1-3; missing otherwise
    # - listwise for DV construction: if ANY of 18 items missing => DV missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing music columns: {missing_music}")

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
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race and Hispanic ethnicity:
    # Ensure 0/1 dummies (not NA) for all non-missing underlying values.
    # Make dummies mutually exclusive with a White non-Hispanic reference:
    # - If Hispanic==1, set black=0 and otherrace=0 (ethnicity overrides race categories).
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    black = np.where(race.isna(), np.nan, (race == 2).astype(float))
    otherrace = np.where(race.isna(), np.nan, (race == 3).astype(float))

    hisp = np.nan * np.ones(len(df), dtype=float)
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Treat 1 as "not Hispanic" and 2 as "Hispanic" if binary.
        uniq = set(pd.unique(eth.dropna()))
        if len(uniq) > 0 and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best-effort: code 1 as non-Hispanic, any other positive code as Hispanic
            hisp = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 99)).astype(float))

    df["hispanic"] = hisp
    df["black"] = black
    df["otherrace"] = otherrace

    # Ethnicity override to make categories closer to "Black / Hispanic / Other race" with white non-Hispanic reference
    mask_hisp = (df["hispanic"] == 1) & df["hispanic"].notna()
    df.loc[mask_hisp, "black"] = 0.0
    df.loc[mask_hisp, "otherrace"] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)

    # Conservative Protestant proxy (best-effort with available vars)
    # Keep it simple but avoid inducing missingness: if Protestant and denom missing -> 0.
    denom_cons = denom.isin([1, 6])  # Baptist or Other Protestant (proxy)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern (per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance index (0–15)
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    missing_tol = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing tolerance columns: {missing_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Keep only plausible substantive codes; otherwise missing.
        # SPK*: 1/2; LIB*: 1/2/3; COL*: often 4/5 with other values; safest keep 1..6
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Save diagnostics
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4/5; DK/NA treated as missing; if any of 18 items missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        s = df[v] if v in df.columns else pd.Series(dtype=float)
        nonmiss = int(s.notna().sum())
        miss = int(s.isna().sum())
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Key frequency tables to catch dummy coding problems
    ft = []
    ft.append(freq_table(race, "race_raw"))
    if "ethnic" in df.columns:
        ft.append(freq_table(clean_gss(df["ethnic"]), "ethnic_raw"))
    ft.append(freq_table(df["black"], "black_dummy"))
    ft.append(freq_table(df["hispanic"], "hispanic_dummy"))
    ft.append(freq_table(df["otherrace"], "otherrace_dummy"))
    write_text("./output/table1_race_ethnicity_checks.txt", "\n\n".join([t.to_string() for t in ft]) + "\n")

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
        "pol_intol": "Political intolerance (0–15)",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save full regression tables (b, beta, p, stars) for debugging
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n\n")
    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    # Save Table 1 style output (constants unstd, predictors standardized betas)
    t1 = table1_display(tab1)
    t2 = table1_display(tab2)
    t3 = table1_display(tab3)
    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from raw data)\n")
    summary_lines.append("Fit statistics:\n" + fit_stats.to_string(index=False))
    summary_lines.append("\n\nModel 1 (Table 1 style):\n" + t1.to_string(index=False))
    summary_lines.append("\n\nModel 2 (Table 1 style):\n" + t2.to_string(index=False))
    summary_lines.append("\n\nModel 3 (Table 1 style):\n" + t3.to_string(index=False))
    summary_text = "\n".join(summary_lines) + "\n"
    write_text("./output/table1_summary.txt", summary_text)

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "missingness": missingness,
        "dv_descriptives": dv_desc.to_frame(name="value"),
        "analytic_samples": {
            "model1": frame1,
            "model2": frame2,
            "model3": frame3
        }
    }