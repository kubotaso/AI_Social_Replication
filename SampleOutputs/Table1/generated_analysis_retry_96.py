def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS missing codes across many variables; keep conservative.
    GSS_NA = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(series, extra_na=()):
        x = to_num(series)
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

    def sd1(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas_from_unstd(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
        sdy = sd1(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sd1(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def safe_ols(y, X, weights=None):
        # Returns fitted results; uses WLS if weights provided; else OLS.
        Xc = sm.add_constant(X, has_constant="add")
        if weights is None:
            return sm.OLS(y, Xc).fit()
        else:
            w = pd.to_numeric(weights, errors="coerce").astype(float)
            return sm.WLS(y, Xc, weights=w).fit()

    def fit_model(df, dv, xcols, model_name, labels, weights=None):
        # Model-specific listwise deletion ONLY on dv + xcols (+weights if given)
        cols = [dv] + list(xcols)
        if weights is not None:
            cols = cols + [weights]

        frame = df[cols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Ensure predictors are float
        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)

        # Drop any zero-variance predictors in this analytic sample (but keep them as rows in output)
        kept, dropped = [], []
        for c in xcols:
            if X[c].nunique(dropna=True) <= 1:
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
        if len(frame) == 0:
            # empty model shell
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        # Fit (on kept predictors only)
        if len(kept) == 0:
            # intercept-only model
            Xk = pd.DataFrame(index=frame.index)
            res = safe_ols(y, Xk, weights=frame[weights] if weights is not None else None)
            meta["r2"] = float(res.rsquared) if hasattr(res, "rsquared") else np.nan
            meta["adj_r2"] = float(res.rsquared_adj) if hasattr(res, "rsquared_adj") else np.nan
            rows.append({"term": "Constant", "b": float(res.params.get("const", np.nan)), "beta": np.nan,
                         "p": float(res.pvalues.get("const", np.nan)), "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        Xk = X[kept]
        res = safe_ols(y, Xk, weights=frame[weights] if weights is not None else None)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_from_unstd(y, Xk, res.params)

        # Constant (unstandardized)
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""
        })

        # Predictors in requested order; fill NA for dropped/unused
        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),  # includes dummies standardized by SD(0/1)
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

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

    def df_to_text(df, title=None, index=False):
        s = ""
        if title:
            s += title.rstrip() + "\n"
        s += df.to_string(index=index) + "\n"
        return s

    # ----------------------------
    # Read + normalize columns
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    # Restrict to GSS 1993
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0–18)
    # - 18 items; disliked if 4 or 5
    # - DK/NA treated missing
    # - DV missing if ANY of 18 items missing
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

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df["educ"]) if "educ" in df.columns else np.nan
    df["prestg80_v"] = clean_gss(df["prestg80"]) if "prestg80" in df.columns else np.nan

    # Income per capita: REALINC / HOMPOP
    df["realinc_v"] = clean_gss(df["realinc"]) if "realinc" in df.columns else np.nan
    df["hompop_v"] = clean_gss(df["hompop"]) if "hompop" in df.columns else np.nan
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df["sex"]) if "sex" in df.columns else np.nan
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(pd.isna(sex), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df["age"]) if "age" in df.columns else np.nan
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity:
    # - Use RACE for Black & Other race.
    # - Use ETHNIC for Hispanic; to avoid huge N loss when ETHNIC missing, default missing ETHNIC to 0 (non-Hispanic).
    #   (This is a pragmatic choice to prevent structural-missing collapse in extracts; adjust if your codebook differs.)
    race = clean_gss(df["race"]) if "race" in df.columns else np.nan
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=White, 2=Black, 3=Other
    df["black"] = np.where(pd.isna(race), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(pd.isna(race), np.nan, (race == 3).astype(float))

    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Most common: 1=not Hispanic, 2=Hispanic. If missing, treat as 0 to avoid structural missingness.
        eth2 = eth.where(eth.isin([1, 2]), np.nan)
        hisp = np.where(pd.isna(eth2), 0.0, (eth2 == 2).astype(float))
        # But if race is missing too, keep hispanic missing (unknown person)
        hisp = np.where(pd.isna(race), np.nan, hisp)
        df["hispanic"] = hisp
    else:
        # If no ETHNIC variable, set to 0 (assume non-Hispanic) but keep missing where race missing
        df["hispanic"] = np.where(pd.isna(race), np.nan, 0.0)

    # Religion: No religion
    relig = clean_gss(df["relig"]) if "relig" in df.columns else np.nan
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(pd.isna(relig), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: RELIG==1 and DENOM in conservative set.
    # With this extract, denom appears coded with small integers; keep common conservative categories.
    # If Protestant but denom missing, set 0 (avoid unnecessary drops).
    denom = clean_gss(df["denom"]) if "denom" in df.columns else np.nan
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6, 7])  # Baptist, other Protestant, no denom (often evangelical/sectarian in some codings)
    df["cons_prot"] = np.where(pd.isna(relig), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & pd.isna(denom) & relig.notna(), "cons_prot"] = 0.0

    # Southern: mapping instruction says REGION==3 is South
    region = clean_gss(df["region"]) if "region" in df.columns else np.nan
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(pd.isna(region), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators, require all 15 items observed
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
        # Keep plausible small integers; other values missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(pd.isna(x), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum across 18 genres of (response in {4,5}); any missing genre => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # Missingness for analysis vars
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
        tot = nonmiss + miss
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / tot * 100.0) if tot else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Frequencies for dummies (sanity check)
    freq_lines = []
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        if v in df.columns:
            vc = df[v].value_counts(dropna=False).sort_index()
            freq_lines.append(f"{v} value counts (incl NA):\n{vc.to_string()}\n")
    write_text("./output/table1_dummy_frequencies.txt", "\n".join(freq_lines))

    # ----------------------------
    # Models (Table 1): unweighted unless a weight variable exists and user adds it
    # The prompt doesn't provide a GSS weight variable in Available Variables, so we stay unweighted.
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

    # Full regression tables
    def format_full(tab):
        t = tab.copy()
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        t["sig"] = t["sig"].fillna("")
        return t

    tab1_full = format_full(tab1)
    tab2_full = format_full(tab2)
    tab3_full = format_full(tab3)

    write_text("./output/table1_fit_stats.txt", df_to_text(fit_stats, title="Fit statistics", index=False))
    write_text("./output/model1_full.txt", df_to_text(tab1_full, title="Model 1 (SES): unstandardized b, standardized beta, p, stars", index=False))
    write_text("./output/model2_full.txt", df_to_text(tab2_full, title="Model 2 (Demographic): unstandardized b, standardized beta, p, stars", index=False))
    write_text("./output/model3_full.txt", df_to_text(tab3_full, title="Model 3 (Political intolerance): unstandardized b, standardized beta, p, stars", index=False))

    # Table 1 style (betas + stars; constant unstd)
    t1_1 = table1_style(tab1)
    t1_2 = table1_style(tab2)
    t1_3 = table1_style(tab3)

    write_text("./output/model1_table1style.txt", df_to_text(t1_1, title="Model 1 (SES): Table-1 style (Constant=b, Predictors=beta+stars)", index=False))
    write_text("./output/model2_table1style.txt", df_to_text(t1_2, title="Model 2 (Demographic): Table-1 style (Constant=b, Predictors=beta+stars)", index=False))
    write_text("./output/model3_table1style.txt", df_to_text(t1_3, title="Model 3 (Political intolerance): Table-1 style (Constant=b, Predictors=beta+stars)", index=False))

    # Combine into one Table 1-like display using UNION of terms (not intersection)
    # Keep the same row order as Model 3 (it includes all terms), ensuring terms appear once.
    order_terms = tab3["term"].tolist()
    # In case a term appears in M1/M2 but missing in M3 (shouldn't), append:
    for term in tab2["term"].tolist() + tab1["term"].tolist():
        if term not in order_terms:
            order_terms.append(term)

    def to_term_map(tstyle_df):
        return dict(zip(tstyle_df["term"].tolist(), tstyle_df["Table1"].tolist()))

    m1_map, m2_map, m3_map = to_term_map(t1_1), to_term_map(t1_2), to_term_map(t1_3)
    combined_rows = []
    for term in order_terms:
        combined_rows.append({
            "term": term,
            "Model 1 (SES)": m1_map.get(term, ""),
            "Model 2 (Demographic)": m2_map.get(term, ""),
            "Model 3 (Political intolerance)": m3_map.get(term, "")
        })
    table1_combined = pd.DataFrame(combined_rows)
    write_text("./output/table1_combined.txt", df_to_text(table1_combined, title="Combined Table 1-style coefficients", index=False))

    # Return objects for programmatic inspection
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "table1_combined": table1_combined,
        "missingness": missingness,
        "n_model_frames": pd.DataFrame({
            "model": ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"],
            "n": [len(frame1), len(frame2), len(frame3)]
        })
    }