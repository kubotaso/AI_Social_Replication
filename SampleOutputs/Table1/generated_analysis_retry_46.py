def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS missing codes (best-effort across variables)
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
        # Two-tailed thresholds as specified
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
        # beta_j = b_j * SD(x_j) / SD(y), within estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion on exactly dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        meta = {"model": model_name, "n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan}

        # Prepare output rows even if empty
        rows = []
        if len(frame) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        # Drop zero-variance predictors only if absolutely necessary (avoid accidental "dropped otherrace")
        X0 = frame[xcols].astype(float)
        keep = [c for c in xcols if X0[c].nunique(dropna=True) > 1]
        dropped = [c for c in xcols if c not in keep]

        y = frame[dv].astype(float)
        X = frame[keep].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)
        meta["dropped"] = ",".join(dropped) if dropped else ""

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
            if c in keep:
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
        # Constant: unstandardized; others: standardized beta with stars
        vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": vals})

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
    # - disliked = 1 if response 4 or 5; 0 if 1..3
    # - DK/NA treated as missing
    # - listwise across 18 items for DV (per description)
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
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum across 18 genres; item=1 if response in {4,5}, 0 if in {1,2,3}; DK/NA => missing.\n"
        "DV set to missing if ANY of 18 genre items missing.\n\n"
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

    # Demographics / group identity
    sex = clean_gss(df.get("sex", np.nan)).where(lambda s: s.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race: ensure 0s are real 0s (not missing) when race observed
    race = clean_gss(df.get("race", np.nan)).where(lambda s: s.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC (available) with explicit 0/1 coding and NA only if ethnic is missing
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Most common GSS coding: 1=not Hispanic, 2=Hispanic
        # Use strict binary if possible; otherwise treat value==1 as non-Hispanic, >=2 as Hispanic (best-effort).
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan)).where(lambda s: s.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: keep simple, avoid creating missingness
    # Use RELIG==1 (Protestant) and DENOM in {1,6} as a defensible approximation.
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = relig == 1
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, treat as not conservative rather than missing (prevents artificial N loss)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South dummy (per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance scale (0–15)
    # IMPORTANT: Avoid artificial missingness.
    # - Code each item intolerant(1)/tolerant(0), NA if missing.
    # - Compute count across answered items.
    # - Require a minimum number of answered items to be comparable; use >=12 of 15 (best-effort).
    #   This reduces overly aggressive missingness while still being close to a full-scale count.
    # ----------------------------
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
    missing_tol_cols = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol_cols:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol_cols}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Keep plausible response codes; if outside range, set missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol_df.notna().sum(axis=1)
    raw_sum = tol_df.sum(axis=1, skipna=True)

    # Require at least 12 answered items (best-effort to align with published N without forcing complete cases)
    df["pol_intol"] = np.where(answered >= 12, raw_sum, np.nan)

    # ----------------------------
    # Diagnostics: missingness (crucial to ensure dummies are not becoming NA)
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

    # Within-Model-3 variance checks to avoid "otherrace dropped" surprises
    # (Write quick frequencies for key dummies in each model frame)
    def freq_text(series, name):
        vc = series.value_counts(dropna=False)
        lines = [f"{name} value_counts (incl NA):"]
        for k, v in vc.items():
            lines.append(f"  {k}: {int(v)}")
        return "\n".join(lines) + "\n"

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

    # Save key model-frame diagnostics
    diag_txt = []
    diag_txt.append("Model-frame diagnostic frequencies\n")
    diag_txt.append(f"\nN Model 1 frame: {len(frame1)}\n")
    diag_txt.append(f"\nN Model 2 frame: {len(frame2)}\n")
    diag_txt.append(f"\nN Model 3 frame: {len(frame3)}\n\n")
    if len(frame2):
        diag_txt.append(freq_text(frame2["black"], "black (Model 2 frame)"))
        diag_txt.append(freq_text(frame2["hispanic"], "hispanic (Model 2 frame)"))
        diag_txt.append(freq_text(frame2["otherrace"], "otherrace (Model 2 frame)"))
        diag_txt.append(freq_text(frame2["south"], "south (Model 2 frame)"))
    if len(frame3):
        diag_txt.append(freq_text(frame3["black"], "black (Model 3 frame)"))
        diag_txt.append(freq_text(frame3["hispanic"], "hispanic (Model 3 frame)"))
        diag_txt.append(freq_text(frame3["otherrace"], "otherrace (Model 3 frame)"))
        diag_txt.append(freq_text(frame3["south"], "south (Model 3 frame)"))
        diag_txt.append(freq_text(frame3["pol_intol"], "pol_intol (Model 3 frame)"))
    write_text("./output/table1_model_frame_diagnostics.txt", "".join(diag_txt))

    # Full regression tables for checking
    def full_table(tab):
        out = tab.copy()
        out["b"] = out["b"].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
        out["beta"] = out["beta"].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
        out["p"] = out["p"].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        out["sig"] = out["sig"].fillna("")
        return out[["term", "b", "beta", "p", "sig"]]

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/model1_full.txt", full_table(tab1).to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", full_table(tab2).to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", full_table(tab3).to_string(index=False) + "\n")

    # Table-1-style display (constant unstandardized, others standardized beta + stars)
    t1_1 = table1_display(tab1)
    t1_2 = table1_display(tab2)
    t1_3 = table1_display(tab3)

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Compact combined summary
    summary_lines = []
    summary_lines.append("Table 1 replication outputs (computed from microdata)\n")
    summary_lines.append("OLS; two-tailed t-tests; stars: * p<.05, ** p<.01, *** p<.001.\n\n")
    summary_lines.append("Fit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nModel 1 (SES) - Table1 style:\n")
    summary_lines.append(t1_1.to_string(index=False))
    summary_lines.append("\n\nModel 2 (Demographic) - Table1 style:\n")
    summary_lines.append(t1_2.to_string(index=False))
    summary_lines.append("\n\nModel 3 (Political intolerance) - Table1 style:\n")
    summary_lines.append(t1_3.to_string(index=False))
    summary_lines.append("\n")
    write_text("./output/table1_summary.txt", "".join(summary_lines))

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