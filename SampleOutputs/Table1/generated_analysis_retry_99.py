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

    def zscore(series):
        s = pd.to_numeric(series, errors="coerce")
        m = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - m) / sd

    def fit_ols_table1(df, dv, xcols, model_name, labels):
        """
        Fit OLS on raw y to get intercept in raw units.
        Compute standardized betas by refitting OLS on z(y) with z(X) (NO intercept),
        which yields standardized coefficients directly (dummy betas are also standardized).
        """
        # Model-specific listwise deletion ONLY on variables in this model
        frame = df[[dv] + xcols].copy().dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors in this analytic sample (but keep in table as NA)
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

        # Prepare output skeleton
        rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
        for c in xcols:
            rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        if len(frame) == 0 or len(kept) == 0:
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)

        # Raw model (for intercept, R2, p-values)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        # Standardized betas: regress z(y) on z(X) with no intercept
        zy = zscore(y)
        zX = pd.DataFrame({c: zscore(X[c]) for c in kept}, index=X.index)
        # If any zscore produced NA due to zero sd (shouldn't for kept), drop rows just in case
        zframe = pd.concat([zy.rename("zy"), zX], axis=1).dropna(axis=0, how="any")
        if len(zframe) > 0:
            zres = sm.OLS(zframe["zy"].values, zframe[kept].values).fit()
            beta_map = dict(zip(kept, zres.params))
        else:
            beta_map = {}

        # Fill rows
        out_rows = []
        out_rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # no stars on constant (as table convention)
        })

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                out_rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(beta_map.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                out_rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(out_rows)
        return meta, tab, frame

    def table1_display(tab):
        # Constant: unstandardized intercept; predictors: standardized beta + stars. No SE/p in this display.
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

    def describe_series(s):
        s = pd.to_numeric(s, errors="coerce")
        return pd.Series({
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "sd": float(s.std(ddof=1)) if s.notna().sum() >= 2 else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan
        })

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
    # Rule: for each of 18 items, disliked=1 if {4,5}, 0 if {1,2,3}, missing otherwise.
    # DV missing if ANY of 18 items missing.
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

    dv_stats = describe_series(df["num_genres_disliked"])
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum across 18 items of (response in {4,5}); DK/NA treated as missing; "
        "DV set missing if any of the 18 items missing.\n\n"
        + dv_stats.to_string()
        + "\n"
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

    # Race (GSS RACE: 1=White, 2=Black, 3=Other)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic (use ETHNIC if available): typical 1=not hispanic, 2=hispanic.
    # IMPORTANT: ensure 0/1 for all non-missing ETHNIC values; do not accidentally create large missingness.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Most consistent for this extract: treat 1=not Hispanic, 2=Hispanic; else fallback to 0/1 by (eth==2)
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: best-effort using RELIG==1 (Prot) and DENOM:
    # 1 Baptist, 2 Methodist, 3 Lutheran, 4 Presbyterian, 5 Episcopalian, 6 Other, 7 None
    # Conservative often includes Baptist + Other Protestant; keep as [1,6].
    is_prot = (relig == 1)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin([1, 6])).astype(float))
    # If Protestant but denom missing, treat as not conservative Protestant (avoid dropping Protestants)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South (mapping instruction says REGION==3 is South)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15): sum of 15 intolerant responses; missing if ANY item missing
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
        # Keep plausible codes (small integers); other values missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + key distributions
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
        miss_rows.append({"variable": v, **describe_series(df[v]).to_dict()})
    missingness = pd.DataFrame(miss_rows)
    missingness["pct_missing"] = (missingness["missing"] / (missingness["n"] + missingness["missing"]) * 100.0).where(
        (missingness["n"] + missingness["missing"]) > 0, np.nan
    )
    missingness = missingness.sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Also save basic frequency checks for key dummies (within year=1993, before model-wise deletion)
    freq_lines = []
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        if v in df.columns:
            vc = df[v].value_counts(dropna=False).sort_index()
            freq_lines.append(f"{v} value counts (incl NA):\n{vc.to_string()}\n")
    write_text("./output/table1_dummy_frequencies.txt", "\n".join(freq_lines))

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

    meta1, tab1, frame1 = fit_ols_table1(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_ols_table1(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_ols_table1(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save full regression tables (includes b, beta, p, sig)
    write_text("./output/table1_model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", tab3.to_string(index=False) + "\n")

    # Save Table-1-style displays (no p-values)
    t1 = table1_display(tab1)
    t2 = table1_display(tab2)
    t3 = table1_display(tab3)

    write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Save fit stats summary
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Save a combined summary file for convenience
    combined = []
    combined.append("Table 1 replication (computed from data)\n")
    combined.append("FIT STATS:\n" + fit_stats.to_string(index=False) + "\n")
    combined.append("\nMODEL 1 (Table-1 style):\n" + t1.to_string(index=False) + "\n")
    combined.append("\nMODEL 2 (Table-1 style):\n" + t2.to_string(index=False) + "\n")
    combined.append("\nMODEL 3 (Table-1 style):\n" + t3.to_string(index=False) + "\n")
    write_text("./output/table1_summary.txt", "\n".join(combined))

    # Return structured outputs
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "missingness": missingness,
        "model_frames": {
            "model1_frame": frame1,
            "model2_frame": frame2,
            "model3_frame": frame3
        }
    }