def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative "standard" GSS missing codes; do NOT treat 1/2/3/... as missing.
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
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
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
            "dropped": ",".join(dropped) if dropped else ""
        }

        rows = []
        if len(frame) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        # Plain OLS (paper reports OLS). If weights are required, they must exist in the dataset; not assumed here.
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

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_display(tab):
        # Constant: show unstandardized b; predictors: show standardized beta + stars
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

    def describe_series(s):
        s = pd.to_numeric(s, errors="coerce")
        d = s.describe()
        extra = pd.Series({
            "missing": int(s.isna().sum()),
            "pct_missing": float(s.isna().mean() * 100.0),
            "min": d.get("min", np.nan),
            "max": d.get("max", np.nan),
            "mean": d.get("mean", np.nan),
            "std": d.get("std", np.nan),
        })
        return extra

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
    # Rule: indicator=1 if response 4 or 5; 0 if 1/2/3; missing otherwise.
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
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Female
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male, 2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race and Hispanic (mutually exclusive categories)
    # Use RACE for White/Black/Other; use ETHNIC for Hispanic-origin. Hispanic overrides race.
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    eth = clean_gss(df.get("ethnic", np.nan)) if "ethnic" in df.columns else pd.Series(np.nan, index=df.index)

    # Best-effort decode of ETHNIC:
    # - If only {1,2}: assume 1=not Hispanic, 2=Hispanic
    # - Else: treat 1 as not Hispanic; 2..9 (and 10..99 if present) as Hispanic-origin categories.
    hisp = pd.Series(np.nan, index=df.index, dtype="float64")
    eth_valid = eth.where(~eth.isin(list(GSS_NA_CODES)), np.nan)

    uniq_eth = set(pd.unique(eth_valid.dropna()))
    if len(uniq_eth) > 0 and uniq_eth.issubset({1.0, 2.0}):
        hisp = np.where(eth_valid.isna(), np.nan, (eth_valid == 2).astype(float))
        hisp = pd.Series(hisp, index=df.index, dtype="float64")
    else:
        # Common multi-category coding
        hisp = np.where(eth_valid.isna(), np.nan, (eth_valid >= 2).astype(float))
        hisp = pd.Series(hisp, index=df.index, dtype="float64")

    df["hispanic"] = hisp

    # Build mutually exclusive race/ethnicity categories:
    # - If hispanic==1 => Hispanic
    # - Else (hispanic==0) => use RACE (White/Black/Other)
    # - If hispanic missing OR race missing (when needed) => missing on these dummies
    need_race = (df["hispanic"] == 0)
    cat_missing = df["hispanic"].isna() | (need_race & race.isna())

    df["black"] = np.where(cat_missing, np.nan, ((df["hispanic"] == 0) & (race == 2)).astype(float))
    df["otherrace"] = np.where(cat_missing, np.nan, ((df["hispanic"] == 0) & (race == 3)).astype(float))
    # Note: White non-Hispanic is reference (hispanic=0, race=1) => black=0, otherrace=0, hispanic=0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: best-effort proxy using RELIG==1 and DENOM in {1 Baptist, 6 Other Protestant}
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # Do not force Protestants with missing denom to missing; treat as not conservative Protestant
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern (REGION==3 per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15): sum of 15 intolerant indicators; missing if ANY item missing
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
        # keep plausible range; otherwise missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics outputs
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    diag = pd.DataFrame({"variable": diag_vars})
    diag = diag.assign(**{
        "missing": [int(df[v].isna().sum()) if v in df.columns else np.nan for v in diag_vars],
        "nonmissing": [int(df[v].notna().sum()) if v in df.columns else np.nan for v in diag_vars],
        "pct_missing": [float(df[v].isna().mean() * 100.0) if v in df.columns else np.nan for v in diag_vars],
    })
    write_text("./output/table1_missingness.txt", diag.sort_values("pct_missing", ascending=False).to_string(index=False) + "\n")

    # DV descriptives
    dv_desc = describe_series(df["num_genres_disliked"])
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum across 18 genres of 1{response in [4,5]}; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # Category counts within potential model samples (helps debug N collapse)
    def value_counts_block(sample_mask, name):
        out = [f"=== {name} ===", f"N rows in sample mask: {int(sample_mask.sum())}"]
        for v in ["hispanic", "black", "otherrace", "female", "south"]:
            if v in df.columns:
                vc = df.loc[sample_mask, v].value_counts(dropna=False).sort_index()
                out.append(f"\n{v} value_counts(dropna=False):\n{vc.to_string()}")
        return "\n".join(out) + "\n"

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

    # Save estimation-sample composition diagnostics
    write_text("./output/table1_sample_composition_model1.txt", value_counts_block(df.index.isin(frame1.index), "Model 1 estimation sample"))
    write_text("./output/table1_sample_composition_model2.txt", value_counts_block(df.index.isin(frame2.index), "Model 2 estimation sample"))
    write_text("./output/table1_sample_composition_model3.txt", value_counts_block(df.index.isin(frame3.index), "Model 3 estimation sample"))

    # Full tables (b, beta, p) for debugging
    def full_table_text(tab, title):
        show = tab.copy()
        show["b"] = show["b"].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
        show["beta"] = show["beta"].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
        show["p"] = show["p"].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        cols = ["term", "b", "beta", "p", "sig"]
        return f"{title}\n{show[cols].to_string(index=False)}\n"

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/table1_model1_full.txt", full_table_text(tab1, "Model 1 (SES): full output"))
    write_text("./output/table1_model2_full.txt", full_table_text(tab2, "Model 2 (Demographic): full output"))
    write_text("./output/table1_model3_full.txt", full_table_text(tab3, "Model 3 (Political intolerance): full output"))

    # Table-1 style (constant=b, predictors=beta+stars)
    t1 = table1_display(tab1).rename(columns={"Table1": "Model 1"})
    t2 = table1_display(tab2).rename(columns={"Table1": "Model 2"})
    t3 = table1_display(tab3).rename(columns={"Table1": "Model 3"})

    # Combine across all terms (outer join on term)
    table1 = t1.merge(t2, on="term", how="outer").merge(t3, on="term", how="outer")

    # Keep a stable, Table-1-like order
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
    table1["__ord"] = table1["term"].map({t: i for i, t in enumerate(term_order)})
    table1 = table1.sort_values(by="__ord", kind="stable").drop(columns="__ord")

    write_text("./output/table1_replication_table.txt", table1.to_string(index=False) + "\n")

    # A compact human-readable summary
    summary_lines = []
    summary_lines.append("Replication outputs for Table 1 (GSS 1993)\n")
    summary_lines.append("DV: Number of music genres disliked (0–18); DV missing if any of 18 genre items missing.\n")
    summary_lines.append("Predictors reported as standardized betas (beta) with stars; constants are unstandardized.\n\n")
    summary_lines.append("Fit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nTable 1-style coefficients (betas + stars; constant unstandardized):\n")
    summary_lines.append(table1.to_string(index=False))
    summary_lines.append("\n")
    write_text("./output/table1_summary.txt", "\n".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "table1": table1,
        "estimation_sample_sizes": pd.DataFrame({
            "model": ["Model 1", "Model 2", "Model 3"],
            "n": [len(frame1), len(frame2), len(frame3)]
        })
    }