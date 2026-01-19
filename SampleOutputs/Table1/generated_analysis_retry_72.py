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

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels, weight_col=None):
        # model-specific listwise deletion ONLY on dv + xcols (+ weight if used)
        cols = [dv] + list(xcols)
        if weight_col is not None:
            cols = cols + [weight_col]
        frame = df[cols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

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

        # Empty shell if can't estimate
        if len(frame) == 0 or len(kept) == 0:
            tab = pd.DataFrame(
                [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
                + [{"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""} for c in xcols]
            )
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        if weight_col is None:
            res = sm.OLS(y, Xc).fit()
        else:
            w = frame[weight_col].astype(float)
            # Guard against nonpositive weights
            w = w.where(np.isfinite(w) & (w > 0), np.nan)
            ok = w.notna()
            y = y.loc[ok]
            Xc = Xc.loc[ok]
            X = X.loc[ok]
            frame = frame.loc[ok]
            meta["n"] = int(len(frame))
            if len(frame) == 0:
                tab = pd.DataFrame(
                    [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
                    + [{"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""} for c in xcols]
                )
                return meta, tab, frame
            res = sm.WLS(y, Xc, weights=w).fit()

        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows = []
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

    def table1_style(tab):
        # Constant: unstandardized b; predictors: standardized beta + stars
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

    def fmt_df(df_):
        if df_ is None:
            return ""
        with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 200):
            return df_.to_string(index=False)

    # ----------------------------
    # Read + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0–18), listwise across 18 items
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
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    write_text(
        "./output/dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Count across 18 genres: 1 if response in {4,5}, 0 if {1,2,3}; DK/NA -> missing; any missing among 18 -> DV missing.\n\n"
        + df["num_genres_disliked"].describe().to_string()
        + "\n"
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

    # Demographics: female, age
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: mutually exclusive dummies with White non-Hispanic reference
    # Use ETHNIC as Hispanic origin when available: 1=not Hispanic, 2=Hispanic (common in GSS extracts)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    eth = None
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Try to interpret 1/2 coding; if not, use best-effort: 1=not Hispanic, >=2 indicates Hispanic origin
        if set(pd.unique(eth.dropna())).issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # best-effort: code 1 as non-Hispanic, 2.. as Hispanic-origin (retain NA if missing)
            hisp = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))
        df["hispanic_any"] = hisp
    else:
        df["hispanic_any"] = np.nan

    # Build mutually exclusive: if Hispanic==1 => Hispanic category regardless of race (common practice)
    # Else if race==2 => Black; else if race==3 => Other; else (race==1) => White reference.
    # Missing if Hispanic is missing OR race missing (to avoid fabricating categories).
    hisp = df["hispanic_any"]
    df["black"] = np.nan
    df["hispanic"] = np.nan
    df["otherrace"] = np.nan

    known = hisp.notna() & race.notna()
    df.loc[known, "hispanic"] = (hisp.loc[known] == 1).astype(float)
    df.loc[known, "black"] = ((hisp.loc[known] == 0) & (race.loc[known] == 2)).astype(float)
    df.loc[known, "otherrace"] = ((hisp.loc[known] == 0) & (race.loc[known] == 3)).astype(float)
    # White reference is implicit: (hisp==0 & race==1) -> all three dummies 0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (best-effort within available fields):
    # Protestant (RELIG==1) AND DENOM in {1 Baptist, 2 Methodist, 3 Lutheran, 4 Presbyterian, 5 Episcopal, 6 Other}
    # In many GSS recodes, "conservative" is often approximated by Baptist + Other Protestant;
    # we implement that conventional approximation while keeping others as 0.
    is_prot = (relig == 1)
    denom = denom.where(denom.isin([1, 2, 3, 4, 5, 6, 7]), np.nan)  # 7 = no denom
    df["cons_prot"] = np.where(relig.isna(), np.nan, 0.0)
    df.loc[is_prot & denom.notna(), "cons_prot"] = denom.isin([1, 6]).astype(float)
    # If Protestant but denom missing, treat as 0 (so we don't drop Protestants just due to denom missing)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators; require all 15 present
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

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Missingness diagnostics (overall)
    # ----------------------------
    diag_vars = [
        "num_genres_disliked",
        "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v",
        "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol"
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
    write_text("./output/missingness_overall.txt", fmt_df(missingness) + "\n")

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

    # No weights variable provided in available variables; keep unweighted OLS (paper may be unweighted).
    weight_col = None

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels, weight_col=weight_col)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels, weight_col=weight_col)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels, weight_col=weight_col)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Model-specific sample composition diagnostics (means of dummies + key vars)
    def sample_profile(frame, cols):
        out = []
        for c in cols:
            if c not in frame.columns:
                continue
            s = frame[c]
            out.append({
                "var": c,
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "sd": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
                "n_nonmissing": int(s.notna().sum())
            })
        return pd.DataFrame(out)

    prof_cols = ["num_genres_disliked"] + m3
    prof1 = sample_profile(frame1, prof_cols)
    prof2 = sample_profile(frame2, prof_cols)
    prof3 = sample_profile(frame3, prof_cols)

    # ----------------------------
    # Save outputs (human-readable)
    # ----------------------------
    write_text("./output/fit_stats.txt", fmt_df(fit_stats) + "\n")

    write_text("./output/model1_full.txt", fmt_df(tab1) + "\n")
    write_text("./output/model2_full.txt", fmt_df(tab2) + "\n")
    write_text("./output/model3_full.txt", fmt_df(tab3) + "\n")

    write_text("./output/model1_table1style.txt", fmt_df(table1_style(tab1)) + "\n")
    write_text("./output/model2_table1style.txt", fmt_df(table1_style(tab2)) + "\n")
    write_text("./output/model3_table1style.txt", fmt_df(table1_style(tab3)) + "\n")

    write_text("./output/model1_sample_profile.txt", fmt_df(prof1) + "\n")
    write_text("./output/model2_sample_profile.txt", fmt_df(prof2) + "\n")
    write_text("./output/model3_sample_profile.txt", fmt_df(prof3) + "\n")

    # Combined "Table 1"-like panel
    panel = pd.DataFrame({"term": table1_style(tab1)["term"]})
    panel["Model 1 (SES)"] = table1_style(tab1)["Table1"].values
    panel = panel.merge(table1_style(tab2), on="term", how="left", suffixes=("", "_m2"))
    panel.rename(columns={"Table1": "Model 2 (Demographic)"}, inplace=True)
    panel = panel.merge(table1_style(tab3), on="term", how="left", suffixes=("", "_m3"))
    panel.rename(columns={"Table1": "Model 3 (Political intolerance)"}, inplace=True)

    write_text("./output/table1_panel.txt", fmt_df(panel) + "\n")

    # Return as dict of DataFrames for programmatic use
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": table1_style(tab1),
        "model2_table1style": table1_style(tab2),
        "model3_table1style": table1_style(tab3),
        "table1_panel": panel,
        "missingness_overall": missingness,
        "model1_sample_profile": prof1,
        "model2_sample_profile": prof2,
        "model3_sample_profile": prof3,
    }