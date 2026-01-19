def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # GSS-style missing codes (best-effort; applied conservatively)
    GSS_NA = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_codes(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA) | set(extra_na)
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
        return np.nan if (pd.isna(v) or v <= 0) else float(np.sqrt(v))

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y) computed on the estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_ols(df, dv, xcols, model_name, labels):
        # Listwise deletion per model only (dv + xcols)
        frame = df[[dv] + xcols].copy().dropna(axis=0, how="any")
        meta = {"model": model_name, "n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan}

        # If empty, return placeholders
        if len(frame) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        # Drop any zero-variance predictors in estimation sample (keep table rows but mark as missing)
        kept, dropped = [], []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)
        meta["dropped"] = ",".join(dropped) if dropped else ""

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas(y, X, res.params)

        rows = []
        rows.append(
            {"term": "Constant", "b": float(res.params.get("const", np.nan)), "beta": np.nan,
             "p": float(res.pvalues.get("const", np.nan)), "sig": ""}  # no stars on constant
        )

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append(
                    {"term": term, "b": float(res.params.get(c, np.nan)), "beta": float(betas.get(c, np.nan)),
                     "p": p, "sig": sig_star(p)}
                )
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def to_table1_style(tab):
        # Table 1 style: unstandardized constant; standardized betas + stars for predictors
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def outer_merge_tables(tables, names):
        # Full outer join on term, columns = each model name
        out = None
        for t, nm in zip(tables, names):
            tt = t.copy()
            tt = tt.rename(columns={"Table1": nm})
            if out is None:
                out = tt
            else:
                out = out.merge(tt, on="term", how="outer")
        return out

    # ----------------------------
    # Read + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")
    df["year_v"] = clean_codes(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: # of music genres disliked (0-18)
    # dislike/dislike very much = 4/5; like/neutral (1-3) = 0; DK/etc -> missing
    # listwise across 18 items for DV construction
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
        x = clean_codes(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum over 18 genres of 1{response in (4,5)}; DK/NA missing; if any genre missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_codes(df.get("educ", np.nan))
    df["prestg80_v"] = clean_codes(df.get("prestg80", np.nan))

    # Household income per capita: REALINC / HOMPOP (as instructed)
    df["realinc_v"] = clean_codes(df.get("realinc", np.nan))
    df["hompop_v"] = clean_codes(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_codes(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_codes(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race dummies (only missing if race missing)
    race = clean_codes(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white 2=black 3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic (ETHNIC) - do NOT make White cases missing; only missing if ETHNIC missing
    # Common GSS coding: 1=not hispanic, 2=hispanic. If more categories: treat 1 as not hispanic; >=2 as hispanic-origin.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_codes(df["ethnic"])
        # Keep as missing only when eth is missing; otherwise force 0/1
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))

    # Religion
    relig = clean_codes(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant (approx): Protestant + denom in (Baptist=1, Other Protestant=6)
    denom = clean_codes(df.get("denom", np.nan))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, treat as not conservative (avoid unnecessary case loss)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern (REGION == 3 per mapping instruction)
    region = clean_codes(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum of 15 intolerant indicators; missing if any item missing
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
        x = clean_codes(df[c])
        # Keep plausible integer codes; otherwise missing
        x = x.where(x.isin([1, 2, 3, 4, 5, 6]), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Missingness diagnostics
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
        miss_rows.append(
            {"variable": v, "nonmissing": nonmiss, "missing": miss,
             "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan}
        )
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Quick value-count diagnostics for key dummies (helps catch coding/NA bugs)
    def vc(s):
        return s.value_counts(dropna=False).to_string()

    write_text(
        "./output/table1_dummy_diagnostics.txt",
        "Value counts (including NA) for key indicators:\n\n"
        f"female:\n{vc(df['female'])}\n\n"
        f"black:\n{vc(df['black'])}\n\n"
        f"hispanic:\n{vc(df['hispanic'])}\n\n"
        f"otherrace:\n{vc(df['otherrace'])}\n\n"
        f"south:\n{vc(df['south'])}\n\n"
        f"cons_prot:\n{vc(df['cons_prot'])}\n\n"
        f"norelig:\n{vc(df['norelig'])}\n"
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
        "pol_intol": "Political intolerance (0–15)",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_ols(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_ols(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_ols(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save full diagnostic regression tables (includes b, beta, p) - clearly marked as replication diagnostics
    write_text("./output/table1_model1_full_replication_diagnostics.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_full_replication_diagnostics.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_full_replication_diagnostics.txt", tab3.to_string(index=False) + "\n")

    # Save Table 1 style (no SE columns; constants unstd; predictors standardized betas + stars)
    t1 = to_table1_style(tab1)
    t2 = to_table1_style(tab2)
    t3 = to_table1_style(tab3)

    write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined table with full outer join so Model 2/3-only variables appear
    combined = outer_merge_tables([t1, t2, t3], ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"])
    # Put terms in a sensible order
    term_order = [
        "Constant",
        "Education (years)",
        "Household income per capita",
        "Occupational prestige",
        "Female",
        "Age",
        "Black",
        "Hispanic",
        "Other race",
        "Conservative Protestant",
        "No religion",
        "Southern",
        "Political intolerance (0–15)",
    ]
    combined["__ord"] = combined["term"].map({t: i for i, t in enumerate(term_order)})
    combined = combined.sort_values(["__ord", "term"], na_position="last").drop(columns="__ord")
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Fit stats summary
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Also return objects for programmatic inspection
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined": combined,
        "missingness": missingness,
        "n_model_frames": pd.DataFrame({
            "model": ["Model 1", "Model 2", "Model 3"],
            "n": [len(frame1), len(frame2), len(frame3)]
        })
    }