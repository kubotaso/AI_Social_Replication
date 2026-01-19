def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # GSS-style missing codes commonly used across items
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

    def standardized_betas_from_unstd(y, X, params, dummy_cols=None):
        """
        beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample.
        Paper reports standardized coefficients; it is standard to standardize all predictors,
        including dummies, unless explicitly stated otherwise. We therefore standardize ALL X.
        """
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_ols_table1(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on dv + xcols (do not pre-drop globally)
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors in this analytic sample
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

        # Build output skeleton even if empty
        rows = []
        if len(frame) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_from_unstd(y, X, res.params)

        # Constant (unstandardized)
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""
        })

        # Predictors (both b and beta for diagnostics; Table1 display uses beta)
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

    def to_table1_style(tab):
        # Predictors: standardized beta + stars; Constant: unstandardized b (no stars)
        out_vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out_vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out_vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].astype(str).values, "Table1": out_vals})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read data + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: count of genres disliked (0-18)
    # Rule: disliked if response in {4,5}; like/neutral {1,2,3}; DK/NA => missing.
    # DV is missing if ANY of 18 items is missing.
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
        # Keep only valid substantive 1..5; else missing
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
        "Construction: sum across 18 genre ratings; disliked={4,5}; like/neutral={1,2,3}; DK/NA=>missing;\n"
        "DV missing if ANY genre item missing (listwise across the 18 items).\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors per mapping
    # ----------------------------
    # Education years
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))

    # Occupational prestige
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Income per capita: REALINC / HOMPOP (both cleaned)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Female dummy from SEX (1=male, 2=female)
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race dummies from RACE (1=white,2=black,3=other)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy from ETHNIC (best effort)
    # Keep: 1=not hispanic, 2=hispanic if binary; otherwise treat 1 as non-hisp, >=2 as hispanic-origin.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))

    # Religion dummies from RELIG and DENOM
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: approximation using RELIG==1 and DENOM in {1 (Baptist), 6 (Other Protestant)}.
    # If Protestant and denom missing, set to 0 to avoid dropping cases.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern dummy: REGION==3 (as provided mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance: sum of 15 intolerant responses
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
        # Keep plausible range for these items (varies); keep 1..6; else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness (in 1993 only)
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
        tot = nonmiss + miss
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / tot * 100.0) if tot else np.nan
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

    meta1, tab1, frame1 = fit_ols_table1(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_ols_table1(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_ols_table1(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table-1-style panels
    t1 = to_table1_style(tab1).rename(columns={"Table1": "Model 1 (SES)"})
    t2 = to_table1_style(tab2).rename(columns={"Table1": "Model 2 (Demographic)"})
    t3 = to_table1_style(tab3).rename(columns={"Table1": "Model 3 (Political intolerance)"})

    # Merge by term (avoid row-order bugs / NaN term issues)
    table1_panel = t1.merge(t2, on="term", how="outer").merge(t3, on="term", how="outer")

    # Keep a consistent order: constant first, then model 1 vars, then additions in order
    desired_order = ["Constant"] + [labels[c] for c in m1] + [labels[c] for c in m2[len(m1):]] + [labels[c] for c in m3[len(m2):]]
    order_map = {name: i for i, name in enumerate(desired_order)}
    table1_panel["__ord"] = table1_panel["term"].map(order_map).fillna(10_000).astype(int)
    table1_panel = table1_panel.sort_values(["__ord", "term"]).drop(columns="__ord").reset_index(drop=True)

    # ----------------------------
    # Write human-readable outputs
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text(
        "./output/table1_model1_full.txt",
        "Model 1 (SES) full results (diagnostic): b (unstd), beta (std), p, stars\n\n"
        + tab1.to_string(index=False)
        + "\n"
    )
    write_text(
        "./output/table1_model2_full.txt",
        "Model 2 (Demographic) full results (diagnostic): b (unstd), beta (std), p, stars\n\n"
        + tab2.to_string(index=False)
        + "\n"
    )
    write_text(
        "./output/table1_model3_full.txt",
        "Model 3 (Political intolerance) full results (diagnostic): b (unstd), beta (std), p, stars\n\n"
        + tab3.to_string(index=False)
        + "\n"
    )

    write_text(
        "./output/table1_panel.txt",
        "Table 1-style output:\n"
        "- Constant is unstandardized (b)\n"
        "- Predictors are standardized coefficients (beta) with two-tailed stars (* p<.05, ** p<.01, *** p<.001)\n\n"
        + table1_panel.to_string(index=False)
        + "\n"
    )

    # Also write a concise overall summary
    summary_lines = []
    summary_lines.append("Replication summary (computed from provided microdata)\n")
    summary_lines.append("DV: Number of music genres disliked (0–18), constructed from 18 genre items.\n")
    summary_lines.append("Models: OLS with Table-1-style reporting (predictors standardized betas; constant unstandardized).\n")
    summary_lines.append("\nFit stats:\n" + fit_stats.to_string(index=False) + "\n")
    write_text("./output/summary.txt", "\n".join(summary_lines))

    # Return results for programmatic inspection
    return {
        "fit_stats": fit_stats,
        "table1_panel": table1_panel,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "missingness": missingness,
        "frames_n": pd.DataFrame({
            "model": ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"],
            "n": [len(frame1), len(frame2), len(frame3)]
        })
    }