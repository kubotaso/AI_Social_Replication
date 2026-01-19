def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS special codes; keep conservative and dataset-safe.
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
        # beta_j = b_j * SD(x_j) / SD(y), computed on the estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
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
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def freq_table(series, name, max_levels=30):
        s = series.copy()
        s = pd.Series(s)
        counts = s.value_counts(dropna=False).head(max_levels)
        out = [f"Frequency: {name}", counts.to_string(), ""]
        return "\n".join(out)

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
    # Dependent variable: # of genres disliked (0-18)
    # DV set missing if ANY of 18 items missing.
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
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + df["num_genres_disliked"].describe().to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Income per capita: REALINC / HOMPOP (as instructed)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan)).where(lambda s: s.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: ensure Hispanic is NOT encoded inside race.
    # Build mutually-exclusive categories with White non-Hispanic as reference:
    # black = (race==2 and not hispanic)
    # otherrace = (race==3 and not hispanic)
    # hispanic = (ethnic indicates hispanic), regardless of race; if hispanic==1 => black/otherrace forced to 0
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)

    # Hispanic from ETHNIC
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Most extracts: 1=not hispanic, 2=hispanic. Treat anything else as missing (avoid best-effort overcoding).
        df["hispanic"] = np.where(
            eth.isna(), np.nan,
            np.where(eth == 2, 1.0, np.where(eth == 1, 0.0, np.nan))
        )

    # Race dummies among non-Hispanic only; if Hispanic missing, keep race dummies based on race alone.
    hisp = df["hispanic"]
    nonhisp = (hisp == 0)
    is_hisp = (hisp == 1)

    black_raw = np.where(race.isna(), np.nan, (race == 2).astype(float))
    other_raw = np.where(race.isna(), np.nan, (race == 3).astype(float))

    df["black"] = black_raw.copy()
    df["otherrace"] = other_raw.copy()

    # Enforce mutual exclusivity when Hispanic is known to be 1
    df.loc[is_hisp.fillna(False), "black"] = 0.0
    df.loc[is_hisp.fillna(False), "otherrace"] = 0.0

    # If Hispanic is known to be 0, keep race coding as-is (black/other are meaningful vs white ref)
    # If Hispanic is missing, do not force; leave black/otherrace based on race.

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)

    # Conservative Protestant: keep simple and avoid introducing missingness:
    # set 1 for prot with denom codes typically conservative/evangelical; set 0 for other prot;
    # if denom missing but prot known -> 0 (do not drop).
    cons_codes = {1, 6}  # Baptist, Other Protestant (best-available in this extract)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin(list(cons_codes))).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern dummy: REGION==3 (per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15): sum of 15 intolerant indicators.
    # IMPORTANT: keep the response ranges flexible by column (avoid wrongly dropping valid codes),
    # and only treat explicit GSS NA codes as missing (already handled by clean_gss).
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
        # Do NOT impose tight numeric range filters; just require it be finite.
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics (raw distributions to catch coding collapse)
    # ----------------------------
    diag_text = []
    diag_text.append(f"N rows after YEAR==1993: {len(df)}\n")

    if "race" in df.columns:
        diag_text.append(freq_table(clean_gss(df["race"]), "race (raw cleaned)"))
    if "ethnic" in df.columns:
        diag_text.append(freq_table(clean_gss(df["ethnic"]), "ethnic (raw cleaned)"))
    diag_text.append(freq_table(df["hispanic"], "hispanic (constructed 0/1/NA)"))
    diag_text.append(freq_table(df["black"], "black (constructed)"))
    diag_text.append(freq_table(df["otherrace"], "otherrace (constructed)"))
    diag_text.append(freq_table(df["south"], "south (constructed)"))

    write_text("./output/table1_diagnostics_freqs.txt", "\n".join(diag_text) + "\n")

    # Missingness summary for analysis vars
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        nonmiss = int(df[v].notna().sum()) if v in df.columns else 0
        miss = int(df[v].isna().sum()) if v in df.columns else 0
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # ----------------------------
    # Models (Table 1)
    # IMPORTANT: model-specific listwise deletion happens inside fit_model.
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

    # Save human-readable outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    t1 = table1_display(tab1)
    t2 = table1_display(tab2)
    t3 = table1_display(tab3)

    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined Table 1-style view (terms aligned by label order)
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

    def to_map(table1style):
        return dict(zip(table1style["term"].tolist(), table1style["Table1"].tolist()))

    m1_map, m2_map, m3_map = to_map(t1), to_map(t2), to_map(t3)

    combined = pd.DataFrame({
        "term": term_order,
        "Model 1 (SES)": [m1_map.get(t, "") for t in term_order],
        "Model 2 (Demographic)": [m2_map.get(t, "") for t in term_order],
        "Model 3 (Political intolerance)": [m3_map.get(t, "") for t in term_order],
    })

    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Stepwise N audit (to find which variable collapses N)
    def stepwise_n(df, dv, steps):
        rows = []
        base = [dv]
        for name, cols in steps:
            needed = base + cols
            n = int(df[needed].dropna().shape[0])
            rows.append({"step": name, "vars_added": ",".join(cols), "n_complete": n})
        return pd.DataFrame(rows)

    n_audit = stepwise_n(
        df,
        "num_genres_disliked",
        steps=[
            ("M1 vars", m1),
            ("M2 added vars", ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]),
            ("M3 added vars", ["pol_intol"]),
        ],
    )
    write_text("./output/table1_stepwise_n_audit.txt", n_audit.to_string(index=False) + "\n")

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
        "stepwise_n_audit": n_audit,
    }