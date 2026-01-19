def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS-style missing codes (varies by variable; we'll also bound-check by valid ranges)
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

    def standardized_betas_from_fit(y, X, params):
        # beta_j = b_j * SD(X_j) / SD(Y), computed on the estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on dv + xcols (key to avoid Model 2 inheriting Model 3 missingness)
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
        }

        # If no data, return shells
        rows = []
        if len(frame) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)

        # Drop zero-variance predictors within THIS model sample only
        kept = [c for c in xcols if X[c].nunique(dropna=True) > 1]
        dropped = [c for c in xcols if c not in kept]
        meta["dropped"] = ",".join(dropped) if dropped else ""

        if len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        Xk = X[kept]
        Xc = sm.add_constant(Xk, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_from_fit(y, Xk, res.params)

        # Build results table: constant (b), predictors (b, beta, p, stars) computed from our model
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # don't star constant (paper doesn't)
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

    def to_table1_style(tab):
        # Mimic Table 1 display: constant as unstandardized b; predictors as standardized beta with computed stars.
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].astype(str).values, "Table1": out})

    def merge_table1_panels(panels):
        # panels: list of (model_label, table1_df)
        # Ensure term is preserved and merged by term (avoid NaN term rows).
        merged = None
        for model_label, tdf in panels:
            t = tdf.copy()
            t = t.rename(columns={"Table1": model_label})
            if merged is None:
                merged = t
            else:
                merged = merged.merge(t, on="term", how="outer")
        # Keep "Constant" first if present
        if merged is None:
            return pd.DataFrame()
        merged["__order__"] = np.where(merged["term"] == "Constant", -1, 0)
        merged = merged.sort_values(["__order__", "term"]).drop(columns="__order__")
        return merged

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
    # Rule: for each item, disliked=1 if 4/5; 0 if 1/2/3; DK/NA missing.
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

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    # Bound-check plausible education in years (avoid propagating weird codes)
    df.loc[(df["educ_yrs"] < 0) | (df["educ_yrs"] > 20), "educ_yrs"] = np.nan

    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))
    df.loc[(df["prestg80_v"] < 0) | (df["prestg80_v"] > 100), "prestg80_v"] = np.nan

    # Income per capita: REALINC / HOMPOP (as instructed)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df.loc[df["realinc_v"] <= 0, "realinc_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[(df["age_v"] <= 0) | (df["age_v"] > 89), "age_v"] = np.nan  # 89 includes 89+

    # Race
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if present; treat code 1 as non-Hispanic and 2 as Hispanic when possible.
    # IMPORTANT: do not create missingness by treating valid "non-Hispanic" codes as NA.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If it's binary 1/2, use that directly.
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Conservative fallback: interpret 1 as "not hispanic"; any other positive code as "hispanic/Spanish origin"
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Best-effort conservative Protestant: Protestant + (Baptist or other Protestant)
    # Keep denom missing among Protestants as 0 to avoid unnecessary listwise deletion.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: follow mapping instruction REGION==3
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15)
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
        # Keep plausible codes only; otherwise missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics (descriptives + missingness)
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    pol_desc = df["pol_intol"].describe()
    write_text(
        "./output/table1_pol_intol_descriptives.txt",
        "Political intolerance scale (0–15)\n"
        "Construction: sum of 15 intolerant indicators; item DK/NA treated as missing; if any item missing => scale missing.\n\n"
        + pol_desc.to_string()
        + "\n"
    )

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

    # Write "full" tables (computed b, beta, p; p-values are computed here and not in the paper's Table 1)
    def tab_to_text(tab):
        t = tab.copy()
        # Keep a stable column order
        t = t[["term", "b", "beta", "p", "sig"]]
        return t.to_string(index=False)

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/model1_full.txt", tab_to_text(tab1) + "\n")
    write_text("./output/model2_full.txt", tab_to_text(tab2) + "\n")
    write_text("./output/model3_full.txt", tab_to_text(tab3) + "\n")

    # Table 1 style (constant b; predictors beta + computed stars)
    t1_1 = to_table1_style(tab1)
    t1_2 = to_table1_style(tab2)
    t1_3 = to_table1_style(tab3)

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    table1_panel = merge_table1_panels([
        ("Model 1 (SES)", t1_1),
        ("Model 2 (Demographic)", t1_2),
        ("Model 3 (Political intolerance)", t1_3),
    ])
    write_text("./output/table1_panel.txt", table1_panel.to_string(index=False) + "\n")

    # Brief summary
    summary_lines = []
    summary_lines.append("Replication outputs for GSS 1993 Table 1-style models (computed from provided microdata).")
    summary_lines.append("Note: Table 1 in the paper does not report SEs; stars here are computed from OLS p-values (two-tailed).")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Panel (constant is unstandardized b; predictors are standardized beta):")
    summary_lines.append(table1_panel.to_string(index=False))
    summary_text = "\n".join(summary_lines) + "\n"
    write_text("./output/summary.txt", summary_text)

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "table1_panel": table1_panel,
        "missingness": missingness,
        "n_model1_frame": int(len(frame1)),
        "n_model2_frame": int(len(frame2)),
        "n_model3_frame": int(len(frame3)),
    }