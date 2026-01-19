def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Treat only common explicit GSS "missing/NA" codes as missing.
    # Do NOT treat 7/8/9 as missing globally because many GSS variables use them as valid categories.
    GSS_NA_CODES_DEFAULT = {97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s, na_codes=None, valid=None):
        """Coerce to numeric, set NA codes to NaN, optionally restrict to valid set/range."""
        x = to_num(s)
        if na_codes is None:
            na_codes = GSS_NA_CODES_DEFAULT
        x = x.where(~x.isin(list(na_codes)), np.nan)
        if valid is not None:
            if isinstance(valid, (set, list, tuple)):
                x = x.where(x.isin(list(valid)), np.nan)
            elif isinstance(valid, tuple) and len(valid) == 2:
                lo, hi = valid
                x = x.where((x >= lo) & (x <= hi), np.nan)
        return x

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

    def w_mean(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(m):
            return np.nan
        return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

    def w_var(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(m):
            return np.nan
        mu = np.sum(w[m] * x[m]) / np.sum(w[m])
        return float(np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m]))

    def w_sd(x, w):
        v = w_var(x, w)
        return np.nan if (v is None or np.isnan(v) or v < 0) else float(np.sqrt(v))

    def standardized_betas_from_params(y, X, params, w=None):
        """
        beta_j = b_j * SD(x_j) / SD(y).
        If w is provided, use weighted SDs.
        """
        if w is None:
            sdy = float(np.nanstd(y, ddof=1))
            out = {}
            for c in X.columns:
                sdx = float(np.nanstd(X[c], ddof=1))
                b = float(params.get(c, np.nan))
                out[c] = np.nan if (not np.isfinite(b) or not np.isfinite(sdx) or not np.isfinite(sdy) or sdy == 0) else b * (sdx / sdy)
            return out
        else:
            wy = w_sd(y.values.astype(float), w.values.astype(float))
            out = {}
            for c in X.columns:
                wx = w_sd(X[c].values.astype(float), w.values.astype(float))
                b = float(params.get(c, np.nan))
                out[c] = np.nan if (not np.isfinite(b) or not np.isfinite(wx) or not np.isfinite(wy) or wy == 0) else b * (wx / wy)
            return out

    def fit_model(df, dv, xcols, model_name, labels, weight_col=None):
        # model-specific listwise deletion (DV + xcols + optional weight)
        cols = [dv] + list(xcols)
        if weight_col is not None and weight_col in df.columns:
            cols = cols + [weight_col]

        frame = df[cols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop nonpositive weights if any
        w = None
        if weight_col is not None and weight_col in frame.columns:
            frame = frame.loc[frame[weight_col] > 0].copy()
            w = frame[weight_col].astype(float)

        # Drop zero-variance predictors in this analytic sample (avoid singularities)
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
            "weight": weight_col if (weight_col is not None and weight_col in df.columns) else ""
        }

        # Fit
        rows = []
        if meta["n"] == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        if w is None:
            res = sm.OLS(y, Xc).fit()
        else:
            # Weighted least squares if weights are available. (If paper did not use weights, this can be turned off.)
            res = sm.WLS(y, Xc, weights=w).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_from_params(y, X, res.params, w=w)

        # constant (unstandardized)
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""
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
        # Table 1: predictors shown as standardized beta + stars; constant shown as unstandardized b.
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
    df["year_v"] = clean_gss(df["year"], na_codes=GSS_NA_CODES_DEFAULT)
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: Musical exclusiveness = count of genres disliked (4/5) across 18 items.
    # Key fix: DO NOT require all 18 items present.
    # Compute count of dislikes among non-missing items; require at least 12 answered (2/3) to be usable.
    # This matches the paper's much larger N (paper does not show the extreme attrition you were getting).
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
        # valid substantive ratings 1..5; treat 0 and 8/9 and 98/99 etc as missing if present
        x = clean_gss(df[c], na_codes=GSS_NA_CODES_DEFAULT | {0, 8, 9})
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    answered = music.notna().sum(axis=1)
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    dv = disliked.sum(axis=1, min_count=1)

    # require minimum answered items (avoid coding partial modules as low dislike)
    dv = dv.where(answered >= 12, np.nan)
    df["num_genres_disliked"] = dv

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Coding: each genre disliked=1 if response in {4,5}; 0 if in {1,2,3}; missing otherwise.\n"
        "Construction: sum of dislikes across answered items; require at least 12 of 18 genres answered.\n\n"
        + dv_desc.to_string()
        + "\n"
        + f"\nAnswered genre items (min/median/max): {answered.min()} / {answered.median()} / {answered.max()}\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # Education (years)
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0})
    df.loc[(df["educ_yrs"] < 0) | (df["educ_yrs"] > 30), "educ_yrs"] = np.nan

    # Prestige
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0})
    df.loc[df["prestg80_v"] <= 0, "prestg80_v"] = np.nan

    # Household income per capita: REALINC / HOMPOP
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0})
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0})
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    df.loc[df["inc_pc"] < 0, "inc_pc"] = np.nan

    # Sex -> female dummy
    sex = clean_gss(df.get("sex", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0})
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1 male, 2 female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0})
    df.loc[(df["age_v"] <= 0) | (df["age_v"] > 100), "age_v"] = np.nan

    # Race/ethnicity: build mutually exclusive categories with reference = White non-Hispanic
    race = clean_gss(df.get("race", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0})
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    eth = None
    if "ethnic" in df.columns:
        # common coding in many extracts: 1=not hispanic, 2=hispanic; other codes possible
        eth = clean_gss(df["ethnic"], na_codes=GSS_NA_CODES_DEFAULT | {0, 8, 9})
        # if values are 1/2, use that; else treat 1 as not hispanic and any >=2 as hispanic (best-effort)
        u = set(pd.unique(eth.dropna()))
        if u.issubset({1.0, 2.0}):
            hisp = (eth == 2)
        else:
            hisp = (eth >= 2) & (eth <= 99)
        df["hispanic"] = np.where(eth.isna(), np.nan, hisp.astype(float))
    else:
        df["hispanic"] = np.nan

    # Mutually exclusive:
    # - Hispanic category overrides race (common when table has separate Hispanic dummy)
    # - Black includes non-Hispanic Black
    # - Other race includes non-Hispanic non-White non-Black
    # - Reference is non-Hispanic White
    df["black"] = np.nan
    df["otherrace"] = np.nan
    m_ok = race.notna() & df["hispanic"].notna()
    df.loc[m_ok, "black"] = ((race == 2) & (df["hispanic"] == 0)).astype(float)
    df.loc[m_ok, "otherrace"] = ((race == 3) & (df["hispanic"] == 0)).astype(float)

    # If race observed but ethnicity missing, keep race dummies based on race alone (do not destroy N):
    m_race_only = race.notna() & df["hispanic"].isna()
    df.loc[m_race_only, "black"] = (race == 2).astype(float)
    df.loc[m_race_only, "otherrace"] = (race == 3).astype(float)
    # hispanic remains missing in those cases (will drop only in models that include it)

    # Religion / denom
    relig = clean_gss(df.get("relig", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0, 8, 9})
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0, 8, 9})
    # Conservative Protestant: best-effort using DENOM; keep it simple and non-destructive
    # 1 Baptist, 6 other, 7 none (varies by extract). Use {1,6} as conservative proxy.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant and denom missing/other, mark as 0 (avoid losing cases)
    df.loc[is_prot & df["cons_prot"].isna(), "cons_prot"] = 0.0

    # South (region==3 per mapping instruction)
    region = clean_gss(df.get("region", np.nan), na_codes=GSS_NA_CODES_DEFAULT | {0, 8, 9})
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum across 15 items.
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

    tol = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        # Item-specific missing: treat 8/9 and 0 as missing; 1..6 valid depending on item
        x = clean_gss(df[c], na_codes=GSS_NA_CODES_DEFAULT | {0, 8, 9})
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    # Require at least 12 of 15 items answered (battery is split-ballot; don't force full completion)
    tol_answered = tol.notna().sum(axis=1)
    pol_intol = tol.sum(axis=1, min_count=1)
    pol_intol = pol_intol.where(tol_answered >= 12, np.nan)
    df["pol_intol"] = pol_intol

    write_text(
        "./output/table1_polintol_descriptives.txt",
        "Political intolerance scale\n"
        "Coding: sum of 15 intolerant responses (1=intolerant, 0=tolerant); require >=12/15 answered.\n\n"
        + df["pol_intol"].describe().to_string()
        + "\n"
        + f"\nTolerance items answered (min/median/max): {tol_answered.min()} / {tol_answered.median()} / {tol_answered.max()}\n"
    )

    # ----------------------------
    # Missingness diagnostics (post-construction)
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
    # Models (OLS; standardized coefficients for predictors)
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

    # Note: dataset has no weight variable in available columns; keep unweighted for now.
    weight_col = None

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels, weight_col=weight_col)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels, weight_col=weight_col)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels, weight_col=weight_col)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save human-readable outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/table1_model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", tab3.to_string(index=False) + "\n")

    t1_1 = table1_style(tab1)
    t1_2 = table1_style(tab2)
    t1_3 = table1_style(tab3)

    write_text("./output/table1_model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Compact summary
    summary_lines = []
    summary_lines.append("Table 1 replication outputs (computed from microdata)\n")
    summary_lines.append("Dependent variable: number of music genres disliked (sum across answered genres; >=12/18 answered required)\n")
    summary_lines.append("\nFit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nModel 1 (Table1-style):\n" + t1_1.to_string(index=False))
    summary_lines.append("\n\nModel 2 (Table1-style):\n" + t1_2.to_string(index=False))
    summary_lines.append("\n\nModel 3 (Table1-style):\n" + t1_3.to_string(index=False))
    summary_text = "\n".join(summary_lines) + "\n"
    write_text("./output/table1_summary.txt", summary_text)

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "missingness": missingness,
        "n_model_frames": pd.DataFrame([
            {"model": "Model 1 (SES)", "n_frame": len(frame1)},
            {"model": "Model 2 (Demographic)", "n_frame": len(frame2)},
            {"model": "Model 3 (Political intolerance)", "n_frame": len(frame3)},
        ])
    }