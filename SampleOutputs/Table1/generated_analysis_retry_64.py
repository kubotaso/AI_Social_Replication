def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Use conservative "special missing" codes. (Do NOT treat valid small positive category codes as missing.)
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
        # beta_j = b_j * SD(x_j) / SD(y), all computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-wise listwise deletion ONLY on dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors in this analytic sample (avoid singular fits)
        kept, dropped = [], []
        for c in xcols:
            nun = frame[c].nunique(dropna=True)
            if nun <= 1:
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
            tab = pd.DataFrame(rows)
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        # Plain OLS (paper reports OLS; no robust/design correction implied by Table 1)
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

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_display(tab):
        # Constant: unstandardized b; predictors: standardized beta + stars
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].astype(str).values, "Table1": out})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def describe_series(s):
        s = pd.to_numeric(s, errors="coerce")
        return {
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "sd": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan
        }

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
    # - Each item: 1..5 substantive; disliked if 4 or 5; DK/NA -> missing
    # - DV missing if ANY of the 18 items missing (listwise across items)
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
    # SES predictors
    # ----------------------------
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Income per capita: REALINC / HOMPOP
    # Fixes:
    # - do not inadvertently treat valid values as missing
    # - require HOMPOP > 0
    # - optional log transform is NOT used (paper says "income per capita", not logged)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # ----------------------------
    # Demographic / group identity controls
    # ----------------------------
    # Female dummy from SEX (1=male, 2=female)
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race dummies from RACE (1=white, 2=black, 3=other)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: use ETHNIC if present; treat only true NA codes as missing.
    # Important: do NOT generate missing for valid non-Hispanic responses.
    # Heuristic: if ETHNIC is binary {1,2} => 2 is Hispanic.
    # Otherwise: treat code==1 as not Hispanic, codes >=2 as Hispanic origin.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        uniq = set(pd.unique(eth.dropna()))
        if len(uniq) > 0 and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))

    # Religion: No religion dummy from RELIG (4=none)
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: Protestant (RELIG==1) and DENOM is "Baptist" or "other Protestant" (approx)
    # Crucial: do NOT create missingness if DENOM missing; set to 0 for Protestants with missing denom.
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = relig == 1
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 (per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance scale (0–15)
    # - intolerant responses per mapping instruction
    # - IMPORTANT: treat only the final pol_intol as needed for Model 3 listwise deletion,
    #   but within scale construction require complete 15 items (consistent with strict count scale).
    # ----------------------------
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
        # Keep plausible small integers only; anything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics
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

    # Descriptives for key constructs
    dv_desc = describe_series(df["num_genres_disliked"])
    pol_desc = describe_series(df["pol_intol"])
    inc_desc = describe_series(df["inc_pc"])
    write_text(
        "./output/table1_key_descriptives.txt",
        "Key descriptives (1993 subset)\n\n"
        f"DV num_genres_disliked: {dv_desc}\n"
        f"pol_intol (0-15):       {pol_desc}\n"
        f"inc_pc (REALINC/HOMPOP): {inc_desc}\n"
    )

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
        "pol_intol": "Political intolerance",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table1-style panels
    t1 = table1_display(tab1).rename(columns={"Table1": "Model 1"})
    t2 = table1_display(tab2).rename(columns={"Table1": "Model 2"})
    t3 = table1_display(tab3).rename(columns={"Table1": "Model 3"})

    # Merge by term to avoid NaN term misalignment bugs
    table1_panel = t1.merge(t2, on="term", how="outer").merge(t3, on="term", how="outer")

    # Ensure stable term ordering: constant first, then in model 3 order
    desired_order = ["Constant"] + [labels.get(c, c) for c in m3]
    desired_order = list(dict.fromkeys(desired_order))  # unique, preserving order
    order_map = {t: i for i, t in enumerate(desired_order)}
    table1_panel["__ord"] = table1_panel["term"].map(order_map).fillna(10_000).astype(int)
    table1_panel = table1_panel.sort_values(["__ord", "term"]).drop(columns="__ord").reset_index(drop=True)

    # ----------------------------
    # Save outputs (human-readable)
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text(
        "./output/table1_model1_full.txt",
        tab1[["term", "b", "beta", "p", "sig"]].to_string(index=False) + "\n"
    )
    write_text(
        "./output/table1_model2_full.txt",
        tab2[["term", "b", "beta", "p", "sig"]].to_string(index=False) + "\n"
    )
    write_text(
        "./output/table1_model3_full.txt",
        tab3[["term", "b", "beta", "p", "sig"]].to_string(index=False) + "\n"
    )

    write_text("./output/table1_panel.txt", table1_panel.to_string(index=False) + "\n")

    # Also write quick sample size checks
    sample_sizes = pd.DataFrame([
        {"model": "Model 1 (SES)", "n": len(frame1)},
        {"model": "Model 2 (Demographic)", "n": len(frame2)},
        {"model": "Model 3 (Political intolerance)", "n": len(frame3)},
    ])
    write_text("./output/table1_sample_sizes.txt", sample_sizes.to_string(index=False) + "\n")

    # Frequency checks for key dummies in each model sample (helps debug dummy drops / sign flips)
    def freq_in_sample(frame, col):
        if col not in frame.columns:
            return None
        s = frame[col]
        return {
            "n": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "sum": float(s.sum()) if s.notna().any() else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan
        }

    freq_report_lines = []
    for model_name, fr in [("Model 1", frame1), ("Model 2", frame2), ("Model 3", frame3)]:
        freq_report_lines.append(f"{model_name} dummy checks:")
        for col in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
            freq_report_lines.append(f"  {col}: {freq_in_sample(fr, col)}")
        freq_report_lines.append("")
    write_text("./output/table1_dummy_checks.txt", "\n".join(freq_report_lines))

    # Return structured results
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "table1_panel": table1_panel,
        "missingness": missingness
    }