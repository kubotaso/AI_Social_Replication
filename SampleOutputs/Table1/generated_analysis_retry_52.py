def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Keep NA handling conservative to avoid falsely recoding valid categories to missing.
    # Many GSS extracts already store true missing as blank/NaN; only recode common explicit missing codes.
    GSS_NA_CODES = {0, 8, 9, 98, 99, 998, 999, 9998, 9999, 997, 9997}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s):
        x = to_num(s)
        return x.where(~x.isin(list(GSS_NA_CODES)), np.nan)

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

    def sd_sample(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        return float(np.sqrt(v)) if pd.notna(v) and v > 0 else np.nan

    def standardized_betas(y, X, params):
        sdy = sd_sample(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sd_sample(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion on dv + xcols only
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        meta = {"model": model_name, "n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan, "dropped": ""}

        # If no data, return shell
        if len(frame) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        # Drop any zero-variance predictors in this specific analytic sample (rare, but guard anyway)
        kept, dropped = [], []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)
        meta["dropped"] = ",".join(dropped) if dropped else ""

        if len(kept) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
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

        rows = [{
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        }]

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

    def table1_style(tab):
        # Constant: unstandardized b; Predictors: standardized beta + stars
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
    # - For each of 18 items: disliked=1 if 4/5, 0 if 1/2/3, else missing
    # - DV is missing if ANY of 18 items missing
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
        x = clean_gss(df[c])
        # only accept substantive 1..5, other values => missing
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum of 18 genre items coded 1 if response in {4,5}, 0 if in {1,2,3}.\n"
        "Missing rule: if any of the 18 items missing/invalid => DV missing.\n\n"
        + df["num_genres_disliked"].describe().to_string()
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

    # Sex
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[(df["age_v"] <= 0) | (df["age_v"] > 99), "age_v"] = np.nan  # keep 89 top-code as 89; guard absurds

    # Race/Ethnicity (mutually exclusive: Black, Hispanic, Other; ref=White non-Hispanic)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    # Use 'ethnic' to define Hispanic origin (best effort for this extract).
    # In this file, ethnic appears to be a coded origin variable with many categories;
    # treat code==1 as "not Hispanic" and code>=2 as "Hispanic/other origin".
    # Then create mutually exclusive categories by letting Hispanic override race.
    hisp = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Keep positive codes; treat others as missing
        eth = eth.where((eth >= 1) & (eth <= 99), np.nan)
        # If binary 1/2: 2=Hispanic
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Otherwise: 1 = not Hispanic; 2..* = Hispanic-origin categories (best effort for this extract)
            hisp = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))
    df["hispanic"] = hisp

    # Mutually exclusive race dummies:
    # Hispanic (any race) is separated; among non-Hispanic, use race to identify Black/Other.
    df["black"] = np.nan
    df["otherrace"] = np.nan
    # For cases with known hispanic and race:
    mask_known = pd.notna(df["hispanic"]) & pd.notna(race)
    df.loc[mask_known, "black"] = ((df.loc[mask_known, "hispanic"] == 0) & (race[mask_known] == 2)).astype(float)
    df.loc[mask_known, "otherrace"] = ((df.loc[mask_known, "hispanic"] == 0) & (race[mask_known] == 3)).astype(float)
    # For cases missing hispanic but known race: still code black/other from race (avoid needless attrition);
    # keep hispanic missing so those cases drop only when hispanic is used in a model.
    mask_hisp_missing = pd.isna(df["hispanic"]) & pd.notna(race)
    df.loc[mask_hisp_missing, "black"] = (race[mask_hisp_missing] == 2).astype(float)
    df.loc[mask_hisp_missing, "otherrace"] = (race[mask_hisp_missing] == 3).astype(float)

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (approximation consistent with using RELIG+DENOM):
    # Mark as conservative if Protestant and denomination is Baptist or Other Protestant.
    is_prot = relig == 1
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, treat as not conservative rather than missing (to reduce spurious attrition)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South
    region = clean_gss(df.get("region", np.nan))
    # In this extract, region codes appear 1..9; follow instruction: REGION==3 is "south"
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant responses; missing if any missing
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
        x = clean_gss(df[c])
        # Keep plausible small integers; other values missing
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

    # Quick frequency checks for dummies (helps catch the previous 'otherrace all zeros' issue)
    freq_txt = []
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        vc = df[v].value_counts(dropna=False).sort_index()
        freq_txt.append(f"\n{v} value_counts(dropna=False):\n{vc.to_string()}\n")
    write_text("./output/table1_dummy_frequencies.txt", "\n".join(freq_txt))

    # ----------------------------
    # Models (OLS; standardized betas reported like Table 1)
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

    # Save human-readable summaries (Table 1 style: betas + stars; constant in raw units)
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    def model_block(meta, tab_full, tab_t1):
        lines = []
        lines.append(f"{meta['model']}")
        lines.append(f"N = {meta['n']}")
        lines.append(f"R^2 = {meta['r2']:.6f}" if pd.notna(meta["r2"]) else "R^2 = NA")
        lines.append(f"Adj R^2 = {meta['adj_r2']:.6f}" if pd.notna(meta["adj_r2"]) else "Adj R^2 = NA")
        if meta.get("dropped"):
            lines.append(f"Dropped (zero variance): {meta['dropped']}")
        lines.append("\nTable-1-style (Constant unstd; Predictors standardized betas):")
        lines.append(tab_t1.to_string(index=False))
        lines.append("\nFull (for debugging: b, beta, p):")
        lines.append(tab_full.to_string(index=False))
        return "\n".join(lines) + "\n"

    summary_text = []
    summary_text.append("Table 1 replication (computed from provided GSS 1993 extract)\n")
    summary_text.append("Notes:\n- OLS fit on model-specific complete cases (listwise per model).\n- Reported betas are standardized coefficients computed from the fitted unstandardized model.\n- Stars based on two-tailed p-values: * p<.05, ** p<.01, *** p<.001.\n")
    summary_text.append("\nFit statistics:\n" + fit_stats.to_string(index=False) + "\n")
    summary_text.append(model_block(meta1, tab1, t1))
    summary_text.append(model_block(meta2, tab2, t2))
    summary_text.append(model_block(meta3, tab3, t3))

    write_text("./output/table1_summary.txt", "\n".join(summary_text))
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Return a convenient bundle
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "missingness": missingness,
        "n_model_frames": pd.DataFrame({
            "model": ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"],
            "n_complete_cases": [len(frame1), len(frame2), len(frame3)]
        })
    }