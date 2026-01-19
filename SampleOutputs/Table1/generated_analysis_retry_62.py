def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # IMPORTANT: Use a conservative NA scheme. The earlier version over-blanked variables
    # (especially ETHNIC and tolerance items), collapsing N.
    GSS_NA_CODES = {0, 8, 9, 98, 99, 998, 999, 9998, 9999}

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
        # beta_j = b_j * SD(x_j) / SD(y) computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels, weights=None):
        # Model-specific listwise deletion ONLY on dv + xcols (+ weights if provided)
        cols = [dv] + xcols
        if weights is not None:
            cols = cols + [weights]
        frame = df[cols].copy().dropna(axis=0, how="any")

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ""
        }

        rows = []
        if len(frame) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)

        # Drop only predictors that are truly zero-variance in THIS model's analytic sample
        kept = [c for c in xcols if X[c].nunique(dropna=True) > 1]
        dropped = [c for c in xcols if c not in kept]
        meta["dropped"] = ",".join(dropped) if dropped else ""

        Xk = X[kept].copy()
        Xc = sm.add_constant(Xk, has_constant="add")

        if weights is None:
            res = sm.OLS(y, Xc).fit()
        else:
            w = frame[weights].astype(float)
            res = sm.WLS(y, Xc, weights=w).fit()

        betas = standardized_betas(y, Xk, res.params)

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

    def table1_style(tab):
        # Constant: unstandardized b; predictors: standardized beta + stars; no SEs shown
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

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: 18 genre ratings; disliked=1 if response is 4/5, else 0 if 1/2/3; DK/NA missing.\n"
        "Listwise for DV: if any of 18 genre items missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # IVs / covariates (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Female
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race dummies (reference = White)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white, 2=black, 3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy from ETHNIC (do NOT over-treat values as missing)
    # Many files use 1=not hispanic, 2=hispanic. In the sample, ETHNIC has values like 29, 97 etc.
    # Treat only explicit missing codes as NA; otherwise: 1 => non-Hispanic; anything not 1 => Hispanic-origin.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If binary 1/2: use it directly.
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best-effort consistent with typical GSS ETHNIC coding:
            # 1 = "not hispanic" (often labeled); all other non-missing codes indicate some Hispanic origin category.
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion dummies
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (approx): Protestant + (Baptist OR Other Protestant)
    # Keep denom missing among Protestants as 0 (avoid unnecessary attrition).
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern dummy (REGION==3 per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    # Do not over-clean REGION: keep any positive codes; only mapping uses ==3 for South.
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15)
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
        # Tolerance items are small integers; keep 1..6, else missing
        x = x.where(x.between(1, 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + quick distributions
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

    # Value counts for key dummies (helps catch the "Other race dropped" problem)
    counts_txt = []
    for v in ["black", "hispanic", "otherrace", "female", "south", "cons_prot", "norelig"]:
        if v in df.columns:
            vc = df[v].value_counts(dropna=False).sort_index()
            counts_txt.append(f"\n{v} value counts (incl NA):\n{vc.to_string()}\n")
    write_text("./output/table1_dummy_value_counts.txt", "\n".join(counts_txt))

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

    # No weights variable is available in the provided dataset; run unweighted OLS.
    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels, weights=None)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels, weights=None)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels, weights=None)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table 1 style outputs (no SEs; standardized betas; unstandardized constant)
    t1_1 = table1_style(tab1)
    t1_2 = table1_style(tab2)
    t1_3 = table1_style(tab3)

    # Ensure all terms appear in each model table (no merge loss)
    all_terms = pd.DataFrame({"term": ["Constant"] + [labels.get(c, c) for c in m3]})
    merged = all_terms.merge(t1_1.rename(columns={"Table1": "Model 1"}), on="term", how="left")
    merged = merged.merge(t1_2.rename(columns={"Table1": "Model 2"}), on="term", how="left")
    merged = merged.merge(t1_3.rename(columns={"Table1": "Model 3"}), on="term", how="left")
    merged = merged.fillna("")

    # ----------------------------
    # Save human-readable text outputs
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

    write_text("./output/table1_main_table.txt", merged.to_string(index=False) + "\n")

    # Also save a compact summary file
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from microdata)\n")
    summary_lines.append("Main table reports: standardized beta (β) with stars; constant is unstandardized.\n")
    summary_lines.append("Stars computed from OLS p-values (two-tailed): *<.05, **<.01, ***<.001.\n\n")
    summary_lines.append("Fit statistics:\n" + fit_stats.to_string(index=False) + "\n\n")
    summary_lines.append("Main Table:\n" + merged.to_string(index=False) + "\n")
    write_text("./output/table1_summary.txt", "".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "table1": merged,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "frames": {
            "model1_frame": frame1,
            "model2_frame": frame2,
            "model3_frame": frame3,
        },
        "missingness": missingness,
    }