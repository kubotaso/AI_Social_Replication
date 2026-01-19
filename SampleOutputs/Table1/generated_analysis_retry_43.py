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

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0) else b * (sdx / sdy)
        return out

    def fit_ols(df, dv, xcols, model_name, labels, add_intercept=True):
        # Model-specific listwise deletion ONLY on dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors within this model sample
        kept, dropped = [], []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {"model": model_name, "n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan, "dropped": ",".join(dropped)}

        # If no sample or no predictors, return shells
        rows = []
        if len(frame) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame, None

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)

        Xc = sm.add_constant(X, has_constant="add") if add_intercept else X
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas(y, X, res.params)

        # Build full table (with p-values internally), but we'll display Table1-style without p-values
        if add_intercept:
            rows.append(
                {
                    "term": "Constant",
                    "b": float(res.params.get("const", np.nan)),
                    "beta": np.nan,
                    "p": float(res.pvalues.get("const", np.nan)),
                    "sig": "",
                }
            )
        else:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

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

        return meta, pd.DataFrame(rows), frame, res

    def table1_style(tab):
        # Constant: unstandardized b. Predictors: standardized beta with stars. No p-values/SEs in display.
        vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": vals})

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
    # Dependent variable: number of music genres disliked (0–18)
    # Rule: disliked = 1 if item is 4/5, else 0 if 1/2/3; DK/NA -> missing.
    # Listwise requirement for DV: if ANY of the 18 items missing -> DV missing.
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
        "Construction: sum across 18 genres of I(response in {4,5}); 'don't know'/NA -> missing; if any genre missing -> DV missing.\n\n"
        + dv_desc.to_string()
        + "\n",
    )

    # ----------------------------
    # Predictors (Table 1)
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

    # Race/ethnicity: enforce mutually exclusive categories with White non-Hispanic as implicit reference.
    # Use RACE (1 white,2 black,3 other) and ETHNIC if present.
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)

    # Hispanic indicator from ETHNIC (best-effort; keep missing small by coding unknown as 0)
    df["hispanic"] = 0.0
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Common case: 1=not Hispanic, 2=Hispanic
        if set(pd.unique(eth.dropna())).issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), 0.0, (eth == 2).astype(float))
        else:
            # Best-effort: treat 1 as non-Hispanic; any other valid positive code as Hispanic-origin
            df["hispanic"] = np.where(eth.isna(), 0.0, ((eth >= 2) & (eth <= 99)).astype(float))
    else:
        df["hispanic"] = 0.0

    # Mutually exclusive race dummies
    # If Hispanic==1, set Black/Other to 0 to avoid overlap (matches typical Table-1 dummy setup).
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    df.loc[df["hispanic"] == 1.0, "black"] = 0.0
    df.loc[df["hispanic"] == 1.0, "otherrace"] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: Protestant + denomination proxy. Keep missing denom from deleting cases by setting 0 for Protestants with missing denom.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # Baptist / Other Protestant (best-effort)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per mapping instruction (residence in South)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum 15 intolerant indicators.
    # Important: Do NOT require all 15 items present; allow partial completion and scale to 0–15:
    #   pol_intol = round(15 * mean(intolerant_items), 6) for those with >= 1 answered item.
    # This avoids artificial attrition from item nonresponse while preserving a 0–15 metric.
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

    tol_mat = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Keep plausible small integers; everything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_mat[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol_mat.notna().sum(axis=1)
    mean_intol = tol_mat.mean(axis=1, skipna=True)
    df["pol_intol"] = np.where(answered >= 1, 15.0 * mean_intol, np.nan)

    # ----------------------------
    # Diagnostics: missingness + key frequencies
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
        miss_rows.append(
            {
                "variable": v,
                "nonmissing": nonmiss,
                "missing": miss,
                "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan,
            }
        )
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Frequency checks within year (helps catch dummy bugs)
    freq_txt = []
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        if v in df.columns:
            freq_txt.append(f"\n{v} value counts (incl NA):\n{df[v].value_counts(dropna=False).to_string()}\n")
    write_text("./output/table1_key_frequencies.txt", "".join(freq_txt))

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

    meta1, tab1, frame1, res1 = fit_ols(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2, res2 = fit_ols(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3, res3 = fit_ols(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save full tables (includes p-values for debugging; Table1 display excludes them)
    def save_model_outputs(model_key, meta, tab, frame):
        write_text(f"./output/{model_key}_fit.txt", pd.DataFrame([meta]).to_string(index=False) + "\n")

        # Full coefficients table
        full = tab.copy()
        # Keep readable rounding
        for col in ["b", "beta", "p"]:
            if col in full.columns:
                full[col] = pd.to_numeric(full[col], errors="coerce")
        write_text(f"./output/{model_key}_coefficients_full.txt", full.to_string(index=False) + "\n")

        # Table 1 style
        t1 = table1_style(tab)
        write_text(f"./output/{model_key}_table1style.txt", t1.to_string(index=False) + "\n")

        # Sample descriptives (quick check)
        desc = frame.describe(include="all").T
        write_text(f"./output/{model_key}_sample_descriptives.txt", desc.to_string() + "\n")

    save_model_outputs("model1", meta1, tab1, frame1)
    save_model_outputs("model2", meta2, tab2, frame2)
    save_model_outputs("model3", meta3, tab3, frame3)

    # Combined summary file (human readable)
    summary_lines = []
    summary_lines.append("Table 1 replication (GSS 1993) — summary\n")
    summary_lines.append("\nFit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nModel 1 (Table1-style):\n")
    summary_lines.append(table1_style(tab1).to_string(index=False))
    summary_lines.append("\n\nModel 2 (Table1-style):\n")
    summary_lines.append(table1_style(tab2).to_string(index=False))
    summary_lines.append("\n\nModel 3 (Table1-style):\n")
    summary_lines.append(table1_style(tab3).to_string(index=False))
    summary_lines.append("\n")
    write_text("./output/table1_summary.txt", "".join(summary_lines))

    # Return a structured object
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": table1_style(tab1),
        "model2_table1style": table1_style(tab2),
        "model3_table1style": table1_style(tab3),
        "missingness": missingness,
    }