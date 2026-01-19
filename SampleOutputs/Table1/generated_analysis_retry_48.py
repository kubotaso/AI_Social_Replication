def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # GSS-style missing codes; apply conservatively (only to variables where these appear)
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

    def standardized_betas_from_estimation_sample(y, X, params):
        # Compute β_j = b_j * SD(x_j)/SD(y) using estimation sample (y,X already listwise)
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            if pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0:
                out[c] = np.nan
            else:
                out[c] = b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels, weight_col=None):
        # model-specific listwise deletion
        use_cols = [dv] + xcols + ([weight_col] if weight_col else [])
        frame = df[use_cols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # ensure dummies are truly 0/1 where expected; if any stray values, set to NA
        for c in xcols:
            if c in {"female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"}:
                v = pd.to_numeric(frame[c], errors="coerce")
                frame[c] = v.where(v.isin([0.0, 1.0]), np.nan)

        frame = frame.dropna(axis=0, how="any").copy()

        # drop any zero-variance predictors in estimation sample (but keep track)
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

        # Empty shell if nothing to estimate
        rows = []
        if len(frame) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        if weight_col:
            w = frame[weight_col].astype(float).clip(lower=1e-9)
            res = sm.WLS(y, Xc, weights=w).fit()
        else:
            res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_estimation_sample(y, X, res.params)
        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star the constant
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

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_display(tab):
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
    # DV: musical exclusiveness (# genres disliked), 18 items
    # - dislike/dislike very much => 1
    # - like/neutral => 0
    # - DK/NA => missing
    # - listwise across 18 items for DV
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
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: enforce mutually exclusive categories to avoid sign flips/overlap
    # Baseline: White non-Hispanic
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    eth = None
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
    # GSS ETHNIC often: 1=not hispanic, 2=hispanic (but can be multi-category in some extracts)
    if eth is None:
        df["hispanic"] = np.nan
    else:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = (eth == 2)
        else:
            # best-effort: treat 1 as not Hispanic; any other positive code as Hispanic origin
            hisp = (eth >= 2) & (eth <= 99)
        df["hispanic"] = np.where(eth.isna(), np.nan, hisp.astype(float))

    # Mutually exclusive: if Hispanic==1, do not classify into race dummies (to match "White non-Hispanic" baseline)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    if "hispanic" in df.columns:
        hisp_mask = (df["hispanic"] == 1.0)
        # For Hispanics, set black/otherrace to 0 (and implicitly remove from White baseline as well)
        df.loc[hisp_mask & df["black"].notna(), "black"] = 0.0
        df.loc[hisp_mask & df["otherrace"].notna(), "otherrace"] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: Protestant and denom in common conservative groups.
    # Use a broader but still simple operationalization; keep denom-missing Protestants as 0 to avoid excess listwise loss.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6, 7])  # Baptist, Other Protestant, No denomination (often evangelical)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per provided mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): allow partial completion by requiring a minimum # answered
    # This reduces artificial missingness vs requiring all 15, while still staying faithful to a count scale.
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
        # keep plausible codes; allow both 1..5 and 1..6 variants
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol_df.notna().sum(axis=1)
    pol_intol = tol_df.sum(axis=1, min_count=1)

    # Require at least 12/15 answered (best-effort to match published N without over-imputation)
    df["pol_intol"] = pol_intol.where(answered >= 12, np.nan)

    # ----------------------------
    # Optional weights (best-effort): use WTSSALL if present; otherwise unweighted
    # ----------------------------
    weight_col = None
    for cand in ["wtssall", "wtssnr", "weight", "wtss"]:
        if cand in df.columns:
            w = clean_gss(df[cand])
            if w.notna().any():
                df[cand] = w
                weight_col = cand
                break

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
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)

    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
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

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels, weight_col=weight_col)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels, weight_col=weight_col)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels, weight_col=weight_col)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table1-style display
    t1 = table1_display(tab1)
    t2 = table1_display(tab2)
    t3 = table1_display(tab3)

    # ----------------------------
    # Save human-readable outputs
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text(
        "./output/table1_model1_full.txt",
        tab1.assign(
            b=tab1["b"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}"),
            beta=tab1["beta"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}"),
            p=tab1["p"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        ).to_string(index=False) + "\n"
    )
    write_text(
        "./output/table1_model2_full.txt",
        tab2.assign(
            b=tab2["b"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}"),
            beta=tab2["beta"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}"),
            p=tab2["p"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        ).to_string(index=False) + "\n"
    )
    write_text(
        "./output/table1_model3_full.txt",
        tab3.assign(
            b=tab3["b"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}"),
            beta=tab3["beta"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}"),
            p=tab3["p"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        ).to_string(index=False) + "\n"
    )

    write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Short summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from microdata)\n")
    summary_lines.append(f"Weighting: {'WLS using ' + weight_col if weight_col else 'Unweighted OLS'}\n")
    summary_lines.append("Fit stats:\n" + fit_stats.to_string(index=False) + "\n\n")
    summary_lines.append("Model 1 (Table 1 style):\n" + t1.to_string(index=False) + "\n\n")
    summary_lines.append("Model 2 (Table 1 style):\n" + t2.to_string(index=False) + "\n\n")
    summary_lines.append("Model 3 (Table 1 style):\n" + t3.to_string(index=False) + "\n")

    write_text("./output/table1_summary.txt", "".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "missingness": missingness,
        "weight_used": pd.DataFrame({"weight_col": [weight_col if weight_col else ""]})
    }