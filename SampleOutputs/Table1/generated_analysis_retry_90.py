def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    GSS_NA_CODES = {
        0, 7, 8, 9,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999
    }

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
        # Model-specific listwise deletion ONLY on dv + xcols
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
            "dropped_predictors": ",".join(dropped) if dropped else ""
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
        # Constant: unstandardized b; predictors: standardized beta + stars
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
    # Dependent variable: number of music genres disliked (0–18)
    # Rule: disliked = 1 if {4,5}; 0 if {1,2,3}; DK/NA -> missing; if any of 18 missing => DV missing
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

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
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
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male, 2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: make mutually exclusive categories with WHITE NON-HISPANIC as reference.
    # GSS 'race': 1 white, 2 black, 3 other
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)

    # Hispanic origin: use ETHNIC if available; ensure non-Hispanic is coded 0 (not missing).
    # Most common: 1=not hispanic, 2=hispanic. If multi-category, treat 1 as not-hispanic; 2.. as hispanic.
    hisp = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If binary 1/2:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best effort: 1=not hispanic, any other positive substantive code => hispanic
            hisp = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 95)).astype(float))
    df["hispanic"] = hisp

    # Mutually exclusive race dummies:
    # - black: race==2 and not hispanic
    # - otherrace: race==3 and not hispanic
    # - hispanic: hispanic==1 regardless of race (and implies black/otherrace set to 0)
    df["black"] = np.nan
    df["otherrace"] = np.nan
    if "hispanic" in df.columns:
        # if hispanic missing, we cannot construct mutually-exclusive dummies reliably -> set to missing
        h = df["hispanic"]
        df["black"] = np.where(race.isna() | h.isna(), np.nan, ((race == 2) & (h == 0)).astype(float))
        df["otherrace"] = np.where(race.isna() | h.isna(), np.nan, ((race == 3) & (h == 0)).astype(float))
        # keep hispanic as 0/1, but leave missing if truly unknown
        df["hispanic"] = np.where(h.isna(), np.nan, (h == 1).astype(float))
    else:
        df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
        df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: keep simple, avoid creating extra missingness.
    # For Protestants (RELIG==1), mark conservative if DENOM indicates Baptist or "other Protestant".
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # 1 Baptist, 6 other
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant and denom missing, treat as not conservative (0) to prevent unnecessary listwise deletion.
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern (mapping instruction says REGION==3 is South)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15): 5 groups x 3 contexts
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
        # Keep small integers; everything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    # IMPORTANT: follow additive count; keep listwise requirement within the scale by default
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

    # Pol intolerance range check
    pol_desc = df["pol_intol"].describe()
    write_text("./output/table1_pol_intol_descriptives.txt", pol_desc.to_string() + "\n")

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

    # Save full regression tables (b, beta, p, stars)
    def fmt_full(tab):
        t = tab.copy()
        for col in ["b", "beta", "p"]:
            if col in t.columns:
                t[col] = pd.to_numeric(t[col], errors="coerce")
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        return t[["term", "b", "beta", "p", "sig"]]

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", fmt_full(tab1).to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", fmt_full(tab2).to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", fmt_full(tab3).to_string(index=False) + "\n")

    # Save Table-1 style (constant b, predictors standardized beta+stars)
    t1_1 = table1_display(tab1)
    t1_2 = table1_display(tab2)
    t1_3 = table1_display(tab3)

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Combined "Table 1"-like view
    all_terms = pd.DataFrame({"term": tab3["term"]})
    combined = all_terms.merge(t1_1, on="term", how="left", suffixes=("", "_m1"))
    combined = combined.rename(columns={"Table1": "Model 1"})
    combined = combined.merge(t1_2.rename(columns={"Table1": "Model 2"}), on="term", how="left")
    combined = combined.merge(t1_3.rename(columns={"Table1": "Model 3"}), on="term", how="left")

    write_text("./output/table1_combined.txt", combined.to_string(index=False) + "\n")

    # Short human-readable summary
    summary_lines = []
    summary_lines.append("Replicated Table 1-style OLS models (computed from provided GSS 1993 extract)")
    summary_lines.append("")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Notes:")
    summary_lines.append("- Predictors shown as standardized betas (β); constants shown as unstandardized intercepts.")
    summary_lines.append("- Stars computed from two-tailed OLS p-values: * p<.05, ** p<.01, *** p<.001.")
    summary_lines.append("- Model-specific listwise deletion is applied per model.")
    summary_lines.append("")
    summary_lines.append("See ./output/model*_full.txt and ./output/model*_table1style.txt for details.")
    write_text("./output/summary.txt", "\n".join(summary_lines) + "\n")

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "table1_combined": combined,
        "missingness": missingness,
    }