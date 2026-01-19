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

    def standardized_betas_from_unstd(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion ONLY on dv + xcols
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
            "dropped": ",".join(dropped) if dropped else "",
        }

        if len(frame) == 0 or len(kept) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
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

        rows = []
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

    def table1_style(tab):
        # Constant: unstandardized b; predictors: standardized beta + stars
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

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
    # Disliked indicator: 1 if response in {4,5}, 0 if in {1,2,3}, else missing
    # DV missing if ANY of 18 items missing (listwise across items)
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

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum across 18 genres of (response==4 or 5); any missing/DK/NA on any genre => DV missing.\n\n"
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
    sex = clean_gss(df.get("sex", np.nan)).where(lambda s: s.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: enforce mutually exclusive categories with White non-Hispanic as implicit reference.
    # Use ETHNIC as Hispanic-origin when present:
    # - if ETHNIC is binary {1,2}: 2 => Hispanic
    # - else if categorical: treat 1 => not Hispanic, any other positive code => Hispanic
    race = clean_gss(df.get("race", np.nan)).where(lambda s: s.isin([1, 2, 3]), np.nan)
    eth = clean_gss(df.get("ethnic", np.nan)) if "ethnic" in df.columns else pd.Series(np.nan, index=df.index)

    hisp = pd.Series(np.nan, index=df.index, dtype=float)
    if "ethnic" in df.columns:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            hisp = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))

    df["hispanic"] = hisp

    # Mutually exclusive dummies:
    # - Hispanic: Hispanic origin (hispanic==1) regardless of race
    # - Black: non-Hispanic and race==2
    # - Other race: non-Hispanic and race==3
    # - White ref: non-Hispanic and race==1 (omitted)
    #
    # Handling of missing:
    # - If Hispanic-origin missing OR race missing, set dummies missing (so model deletion matches explicit listwise).
    # This is conservative, but avoids overlap/collinearity and fixes sign issues caused by overlap.
    valid_re = race.notna() & df["hispanic"].notna()
    df["black"] = np.where(valid_re, ((df["hispanic"] == 0) & (race == 2)).astype(float), np.nan)
    df["otherrace"] = np.where(valid_re, ((df["hispanic"] == 0) & (race == 3)).astype(float), np.nan)
    df["hispanic"] = np.where(valid_re, (df["hispanic"] == 1).astype(float), np.nan)

    # Religion
    relig = clean_gss(df.get("relig", np.nan)).where(lambda s: s.isin([1, 2, 3, 4, 5]), np.nan)
    denom = clean_gss(df.get("denom", np.nan))
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: Protestant (RELIG==1) and DENOM in {1 Baptist, 6 Other Protestant}.
    # Keep denom-missing Protestants as 0 to avoid dropping many cases.
    is_prot = relig == 1
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Region (South): mapping instruction says REGION==3 is South
    region = clean_gss(df.get("region", np.nan))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant responses; missing if any item missing
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
        # keep plausible range for these items; else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Model specs
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
    # Diagnostics: variable missingness (overall) + key frequencies within each model
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

    def freq_in_sample(frame, col):
        if col not in frame.columns:
            return ""
        s = frame[col]
        vc = s.value_counts(dropna=False).sort_index()
        return vc.to_string()

    sample_diag_txt = []
    for name, frame in [("Model 1 (SES)", frame1), ("Model 2 (Demographic)", frame2), ("Model 3 (Political intolerance)", frame3)]:
        sample_diag_txt.append(f"{name} analytic N = {len(frame)}\n")
        if len(frame) > 0:
            for col in ["black", "hispanic", "otherrace", "south", "female"]:
                if col in frame.columns:
                    sample_diag_txt.append(f"{col} frequency (incl. NA):\n{freq_in_sample(frame, col)}\n")
        sample_diag_txt.append("-" * 60 + "\n")
    write_text("./output/table1_model_sample_diagnostics.txt", "".join(sample_diag_txt))

    # ----------------------------
    # Save tables (human-readable)
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Full regression tables (b, beta, p, stars) for debugging
    write_text("./output/table1_model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", tab3.to_string(index=False) + "\n")

    # Table-1 style output: constant + standardized betas
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)
    write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined Table 1 (side-by-side)
    all_terms = pd.DataFrame({"term": tab3["term"].values})
    combined = all_terms.merge(t1, on="term", how="left", suffixes=("", "_m1"))
    combined = combined.rename(columns={"Table1": "Model 1 (SES)"})
    combined = combined.merge(t2, on="term", how="left")
    combined = combined.rename(columns={"Table1": "Model 2 (Demographic)"})
    combined = combined.merge(t3, on="term", how="left")
    combined = combined.rename(columns={"Table1": "Model 3 (Political intolerance)"})
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Summary text
    summary_lines = []
    summary_lines.append("Table 1 replication outputs (computed from data; standardized betas for predictors)\n")
    summary_lines.append(f"Rows in input after YEAR==1993 filter: {len(df)}\n\n")
    summary_lines.append("Fit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False) + "\n\n")
    summary_lines.append("Notes:\n")
    summary_lines.append("- Predictors reported as standardized coefficients (beta) with stars; constants are unstandardized.\n")
    summary_lines.append("- Each model uses model-specific listwise deletion on DV + that model's predictors.\n")
    summary_lines.append("- Race/ethnicity dummies are mutually exclusive with White non-Hispanic as reference.\n")
    summary_lines.append("- Output files written under ./output\n")
    write_text("./output/table1_summary.txt", "".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined_table1style": combined,
        "missingness": missingness,
    }