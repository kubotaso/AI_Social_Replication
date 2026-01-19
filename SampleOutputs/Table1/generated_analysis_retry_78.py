def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Use a conservative (but not over-aggressive) NA set.
    # Avoid treating "0" as missing globally because some GSS recodes legitimately use 0/1.
    GSS_NA = {7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean(s, extra_na=()):
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

    def sd_sample(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.sqrt(x.var(ddof=1))) if x.notna().sum() > 1 else np.nan

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on the estimation sample
        sdy = sd_sample(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sd_sample(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0) else b * (sdx / sdy)
        return out

    def fit_ols_table1(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion on ONLY dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # If any predictor is constant in the estimation sample, keep it but report NaN (paper has it, but
        # our estimation shouldn't silently remove rows). statsmodels will drop collinear columns; we handle it.
        X = frame[xcols].astype(float)
        y = frame[dv].astype(float)

        # Build design matrix and fit
        Xc = sm.add_constant(X, has_constant="add")

        # If singular, statsmodels can still fit but will drop columns internally; capture safely.
        res = sm.OLS(y, Xc, missing="drop").fit()

        params = res.params.copy()
        pvals = res.pvalues.copy()

        # Standardized betas computed on estimation sample (using original X columns)
        betas = standardized_betas(y, X, params)

        # Assemble full table (constant unstd, predictors with b/beta/p/star)
        rows = []
        rows.append({
            "term": "Constant",
            "b": float(params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(pvals.get("const", np.nan)),
            "sig": ""
        })

        for c in xcols:
            p = float(pvals.get(c, np.nan)) if c in pvals.index else np.nan
            rows.append({
                "term": labels.get(c, c),
                "b": float(params.get(c, np.nan)),
                "beta": float(betas.get(c, np.nan)),
                "p": p,
                "sig": sig_star(p)
            })

        full = pd.DataFrame(rows)

        # Table1-style: predictors show standardized beta + stars; constant shows unstandardized b
        disp_vals = []
        for _, r in full.iterrows():
            if r["term"] == "Constant":
                disp_vals.append("" if pd.isna(r["b"]) else f"{r['b']:.3f}")
            else:
                disp_vals.append("" if pd.isna(r["beta"]) else f"{r['beta']:.3f}{r['sig']}")
        table1 = pd.DataFrame({"term": full["term"], "Table1": disp_vals})

        meta = pd.DataFrame([{
            "model": model_name,
            "n": int(res.nobs) if hasattr(res, "nobs") else int(frame.shape[0]),
            "r2": float(res.rsquared) if hasattr(res, "rsquared") else np.nan,
            "adj_r2": float(res.rsquared_adj) if hasattr(res, "rsquared_adj") else np.nan
        }])

        return meta, full, table1, frame

    def write_txt(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: Number of music genres disliked (0–18), listwise across 18 items for DV construction
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
        x = clean(df[c])
        # keep only substantive 1..5
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean(df["educ"]) if "educ" in df.columns else np.nan
    df["prestg80_v"] = clean(df["prestg80"]) if "prestg80" in df.columns else np.nan

    df["realinc_v"] = clean(df["realinc"]) if "realinc" in df.columns else np.nan
    df["hompop_v"] = clean(df["hompop"]) if "hompop" in df.columns else np.nan
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean(df["sex"]) if "sex" in df.columns else np.nan
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    age = clean(df["age"]) if "age" in df.columns else np.nan
    # Keep topcode 89 as 89; drop nonpositive
    df["age_v"] = age.where(age > 0, np.nan)

    race = clean(df["race"]) if "race" in df.columns else np.nan
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy (use 'ethnic' in this extract). In GSS, ETHNIC often: 1=Hispanic, 2=Not Hispanic.
    # The prior code reversed/overgeneralized; fix by using majority mapping:
    # - If values are {1,2}: assume 1=Hispanic, 2=Not Hispanic (common in GSS extracts).
    # - Otherwise, if >2 categories: treat explicit 1 as Hispanic, 2 as not, others as missing (unknown).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean(df["ethnic"])
        vals = sorted(pd.unique(eth.dropna()))
        if set(vals).issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))
        else:
            # Best-effort: keep only 1/2 if present, else missing (avoid the earlier ">=2 means hispanic" bug)
            df["hispanic"] = np.where(eth.isna(), np.nan, np.where(eth == 1, 1.0, np.where(eth == 2, 0.0, np.nan)))

    # Religion
    relig = clean(df["relig"]) if "relig" in df.columns else np.nan
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean(df["denom"]) if "denom" in df.columns else np.nan
    # Conservative Protestant: approximation using available denom codes.
    # Keep conservative=1 for Protestants with denom in {1 Baptist, 6 Other Protestant}; else 0 among Protestants.
    is_prot = relig == 1
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Region: mapping instruction says REGION==3 is South; do that.
    region = clean(df["region"]) if "region" in df.columns else np.nan
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15), listwise across 15 items for scale construction
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

    tol = pd.DataFrame(index=df.index)
    for c, intolerant in tol_items:
        x = clean(df[c])
        # keep plausible codes; treat others as missing
        x = x.where(x.isin([1, 2, 4, 5]), np.nan)
        tol[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant)).astype(float))

    df["pol_intol"] = tol.sum(axis=1)
    df.loc[tol.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics (quick checks)
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss_n = int(df[v].isna().sum())
        miss.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss_n,
            "pct_missing": (miss_n / (nonmiss + miss_n) * 100.0) if (nonmiss + miss_n) else np.nan
        })
    missingness = pd.DataFrame(miss).sort_values("pct_missing", ascending=False)

    # simple distributions for key dummies (on all 1993 cases)
    def freq01(series):
        s = series.dropna()
        return pd.Series({
            "n": int(s.shape[0]),
            "mean": float(s.mean()) if s.shape[0] else np.nan,
            "p1": float((s == 1).mean()) if s.shape[0] else np.nan,
            "p0": float((s == 0).mean()) if s.shape[0] else np.nan
        })

    dummy_checks = pd.DataFrame({
        "female": freq01(df["female"]),
        "black": freq01(df["black"]),
        "hispanic": freq01(df["hispanic"]),
        "otherrace": freq01(df["otherrace"]),
        "south": freq01(df["south"])
    }).T

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

    meta1, full1, t1_1, frame1 = fit_ols_table1(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, full2, t1_2, frame2 = fit_ols_table1(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, full3, t1_3, frame3 = fit_ols_table1(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.concat([meta1, meta2, meta3], ignore_index=True)

    # ----------------------------
    # Save human-readable outputs
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_txt(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    write_txt("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")
    write_txt("./output/table1_dummy_checks.txt", dummy_checks.to_string() + "\n")

    write_txt("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_txt("./output/model1_full.txt", full1.to_string(index=False) + "\n")
    write_txt("./output/model2_full.txt", full2.to_string(index=False) + "\n")
    write_txt("./output/model3_full.txt", full3.to_string(index=False) + "\n")

    write_txt("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_txt("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_txt("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Combined Table 1-style panel
    combined = t1_1.merge(t1_2, on="term", how="outer", suffixes=(" (M1)", " (M2)"))
    combined = combined.merge(t1_3, on="term", how="outer")
    combined = combined.rename(columns={"Table1": "Table1 (M3)"})
    combined = combined[["term", "Table1 (M1)", "Table1 (M2)", "Table1 (M3)"]]
    write_txt("./output/table1_combined_panel.txt", combined.to_string(index=False) + "\n")

    # Short summary
    summary_lines = []
    summary_lines.append("Table 1 replication (GSS 1993) — computed from raw data\n")
    summary_lines.append("Fit statistics:\n" + fit_stats.to_string(index=False) + "\n\n")
    summary_lines.append("Dummy checks (means; should be plausible and non-constant):\n" + dummy_checks.to_string() + "\n\n")
    summary_lines.append("Model 1 (Table 1 style):\n" + t1_1.to_string(index=False) + "\n\n")
    summary_lines.append("Model 2 (Table 1 style):\n" + t1_2.to_string(index=False) + "\n\n")
    summary_lines.append("Model 3 (Table 1 style):\n" + t1_3.to_string(index=False) + "\n")
    write_txt("./output/table1_summary.txt", "".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "table1_panel": combined,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "model1_full": full1,
        "model2_full": full2,
        "model3_full": full3,
        "missingness": missingness,
        "dummy_checks": dummy_checks,
        "n_model1_frame": pd.DataFrame({"n": [len(frame1)]}),
        "n_model2_frame": pd.DataFrame({"n": [len(frame2)]}),
        "n_model3_frame": pd.DataFrame({"n": [len(frame3)]}),
    }