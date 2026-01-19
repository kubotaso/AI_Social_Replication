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

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas_from_b(y, X, b_params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(b_params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion on dv + xcols ONLY
        frame = df[[dv] + xcols].copy().dropna(axis=0, how="any")

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
        }

        # Empty guard
        rows = []
        if len(frame) == 0:
            rows.append({"term": "Constant", "b0": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b0": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)

        # Drop zero-variance predictors in this analytic sample (should not happen if coding is correct)
        kept = [c for c in xcols if X[c].nunique(dropna=True) > 1]
        dropped = [c for c in xcols if c not in kept]
        meta["dropped"] = ",".join(dropped) if dropped else ""

        Xk = X[kept].copy()
        Xc = sm.add_constant(Xk, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_from_b(y, Xk, res.params)

        # Constant row: unstandardized
        rows.append(
            {
                "term": "Constant",
                "b0": float(res.params.get("const", np.nan)),
                "beta": np.nan,
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": "",  # do not star the constant
            }
        )

        # Predictor rows: report standardized beta + stars (p-values from unstandardized OLS)
        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append(
                    {
                        "term": term,
                        "b0": np.nan,
                        "beta": float(betas.get(c, np.nan)),
                        "p": p,
                        "sig": sig_star(p),
                    }
                )
            else:
                rows.append({"term": term, "b0": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_style(tab):
        # Constant: show b0, predictors: show beta + sig
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b0"]) else f"{float(r['b0']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
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
    # Disliked = response 4 or 5 on each item, 1..3 = not disliked, other/DK/NA = missing
    # DV requires complete data on all 18 items (listwise across items)
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    miss_music = [c for c in music_items if c not in df.columns]
    if miss_music:
        raise ValueError(f"Missing required music columns: {miss_music}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # ----------------------------
    # SES predictors
    # ----------------------------
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # ----------------------------
    # Demographics / group identity controls
    # ----------------------------
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1 male, 2 female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race dummies (White reference)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy from ETHNIC (best effort; keep missing only when ETHNIC missing)
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If ETHNIC is a binary not-hispanic/hispanic
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best-effort: treat 1 as not Hispanic; any other positive code as Hispanic origin
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion dummies
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant proxy: Protestant and denomination in Baptist or "other Protestant"
    is_prot = (relig == 1)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin([1, 6])).astype(float))
    # If Protestant but denom missing, code 0 to avoid unnecessary deletion in Model 2/3
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern dummy (mapping instruction: REGION==3)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance (0–15): allow partial completion to reduce excessive missingness
    # Paper battery asked of ~2/3 sample; strict 15/15 listwise is usually too strict.
    # Rule here: compute sum if respondent answered at least MIN_ANSWERED items; else missing.
    # ----------------------------
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    miss_tol = [c for c, _ in tol_items if c not in df.columns]
    if miss_tol:
        raise ValueError(f"Missing required political tolerance columns: {miss_tol}")

    tol_ind = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Keep plausible codes 1..6; else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_ind[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol_ind.notna().sum(axis=1)
    MIN_ANSWERED = 12  # pragmatic partial-completion rule to better align with reported N
    df["pol_intol"] = tol_ind.sum(axis=1, skipna=True)
    df.loc[answered < MIN_ANSWERED, "pol_intol"] = np.nan

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

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n",
    )
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

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table 1 style outputs
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    # Save human-readable outputs (no SEs; p-values retained only in "full" debug tables)
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Full debug tables (b0 for constant, beta/p for predictors)
    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    # Simple sample-size diagnostics per model
    sample_diag = pd.DataFrame(
        [
            {"model": meta1["model"], "n": meta1["n"], "dv_nonmissing": int(df["num_genres_disliked"].notna().sum())},
            {"model": meta2["model"], "n": meta2["n"], "dv_nonmissing": int(df["num_genres_disliked"].notna().sum())},
            {"model": meta3["model"], "n": meta3["n"], "dv_nonmissing": int(df["num_genres_disliked"].notna().sum())},
        ]
    )
    write_text("./output/table1_sample_diag.txt", sample_diag.to_string(index=False) + "\n")

    # Return results as a dict of DataFrames
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "missingness": missingness,
    }