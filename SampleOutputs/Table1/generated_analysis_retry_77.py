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

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sample_sd(x, w=None):
        x = pd.to_numeric(x, errors="coerce").astype(float)
        if w is None:
            v = x.var(ddof=1)
            return np.nan if pd.isna(v) or v <= 0 else float(np.sqrt(v))
        w = pd.to_numeric(w, errors="coerce").astype(float)
        m = (~x.isna()) & (~w.isna()) & (w > 0)
        x = x[m].to_numpy()
        w = w[m].to_numpy()
        if x.size < 2:
            return np.nan
        wsum = w.sum()
        if wsum <= 0:
            return np.nan
        mu = (w * x).sum() / wsum
        # reliability-weighted (frequency weights style) variance with Bessel-ish correction
        # effective df = sum(w) - 1
        denom = wsum - 1.0
        if denom <= 0:
            return np.nan
        v = (w * (x - mu) ** 2).sum() / denom
        return np.nan if (not np.isfinite(v) or v <= 0) else float(np.sqrt(v))

    def standardized_betas(y, X, params, w=None):
        sdy = sample_sd(y, w=w)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c], w=w)
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels, wcol=None):
        # Model-specific listwise deletion only on dv + xcols (+ weight if used)
        cols = [dv] + list(xcols)
        if wcol is not None and wcol in df.columns:
            cols = cols + [wcol]

        frame = df[cols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors in THIS analytic sample
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

        # Shell on empty
        if len(frame) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        if wcol is not None and wcol in frame.columns:
            w = frame[wcol].astype(float)
            res = sm.WLS(y, Xc, weights=w).fit()
            betas = standardized_betas(y, X, res.params, w=w)
        else:
            res = sm.OLS(y, Xc).fit()
            betas = standardized_betas(y, X, res.params, w=None)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows = [{
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        }]

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
        # Constant: unstandardized b. Predictors: standardized beta + stars. No SE/p-values in display.
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def describe_dummies(frame, cols):
        rows = []
        for c in cols:
            if c not in frame.columns:
                continue
            s = frame[c]
            rows.append({
                "var": c,
                "n": int(s.notna().sum()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "sd": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan
            })
        return pd.DataFrame(rows)

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
    # - 18 items; disliked if 4/5; 1-3 = not disliked; anything else missing
    # - DV missing if ANY of 18 items missing (listwise over items)
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

    # Income per capita: REALINC / HOMPOP (no logging; treat NA codes as missing)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df.loc[df["realinc_v"] <= 0, "realinc_v"] = np.nan  # nonpositive treated as missing
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # ----------------------------
    # Demographic / group identity predictors
    # ----------------------------
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1 male, 2 female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC (1=not hispanic, 2=hispanic). Any other codes -> missing (avoid reverse coding).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        df["hispanic"] = np.where(eth.isna(), np.nan,
                                  np.where(eth == 2, 1.0,
                                           np.where(eth == 1, 0.0, np.nan)))

    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: Protestant (RELIG==1) and DENOM in {1 Baptist, 2 Methodist, 3 Lutheran, 4 Presbyterian, 5 Episcopalian, 6 Other}
    # Here, use a conservative-leaning approximation: {1 Baptist, 6 Other} only; denom missing among Protestants -> 0 (retain cases).
    is_prot = (relig == 1)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin([1, 6])).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    # Mapping instruction: South == 3
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance scale (0–15): sum of 15 intolerant indicators; complete-case required across 15 items
    # ----------------------------
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1})
    ]
    miss_tol = [c for c, _ in tol_items if c not in df.columns]
    if miss_tol:
        raise ValueError(f"Missing required political tolerance columns: {miss_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # keep typical codes 1..6 only; everything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Labels
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
        "pol_intol": "Political intolerance (0–15)"
    }

    # ----------------------------
    # Diagnostics: distributions + missingness + key dummy means (overall)
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
        "Rule: sum across 18 genre items; disliked if response in {4,5}; 1-3 = not disliked; DK/NA missing.\n"
        "If any of 18 items missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Models (unweighted, OLS), standardized betas for predictors; constant unstandardized
    # ----------------------------
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels, wcol=None)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels, wcol=None)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels, wcol=None)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table1-style panels
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    # Save full coefficient tables (b, beta, p, sig) for debugging; Table1-style for comparison
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Model-sample composition checks (means of dummies, etc.)
    sample_checks = {
        "model1_sample_desc": describe_dummies(frame1, ["educ_yrs", "inc_pc", "prestg80_v"]),
        "model2_sample_desc": describe_dummies(frame2, ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]),
        "model3_sample_desc": describe_dummies(frame3, ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south", "pol_intol"])
    }
    write_text("./output/model_sample_composition.txt",
               "Model 1 sample:\n" + sample_checks["model1_sample_desc"].to_string(index=False) + "\n\n"
               "Model 2 sample:\n" + sample_checks["model2_sample_desc"].to_string(index=False) + "\n\n"
               "Model 3 sample:\n" + sample_checks["model3_sample_desc"].to_string(index=False) + "\n")

    # Combined Table 1-like output (side-by-side)
    merged = t1.rename(columns={"Table1": "Model 1"}).merge(
        t2.rename(columns={"Table1": "Model 2"}), on="term", how="outer"
    ).merge(
        t3.rename(columns={"Table1": "Model 3"}), on="term", how="outer"
    )
    # Put terms in canonical order
    term_order = ["Constant"] + [labels[c] for c in m3]
    merged["__ord"] = merged["term"].map({t: i for i, t in enumerate(term_order)})
    merged = merged.sort_values(["__ord", "term"]).drop(columns="__ord")
    write_text("./output/table1_combined.txt", merged.to_string(index=False) + "\n")

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined": merged,
        "missingness": missingness,
        "model_samples": sample_checks
    }