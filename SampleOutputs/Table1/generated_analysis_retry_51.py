def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative "special codes" set; we also allow per-variable tightening below.
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s, valid=None, extra_na=()):
        """Convert to numeric, set GSS missing codes to NaN, optionally enforce valid set/range."""
        x = to_num(s)
        na = set(GSS_NA_CODES) | set(extra_na)
        x = x.where(~x.isin(list(na)), np.nan)
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

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y) on estimation sample
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

        # Drop any zero-variance predictors in this analytic sample (but keep in output as NaN)
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
            return meta, pd.DataFrame(rows), frame

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
            "sig": ""  # never star constant in Table 1 style
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

    def table1_display(tab):
        # Constant: show unstandardized b; predictors: show standardized beta + stars
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
    # Dependent variable: number of music genres disliked (0–18)
    # Rule: count across 18 items where response in {4,5}; DK/NA missing;
    # DV missing if ANY item missing (listwise for the scale).
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
        x = clean_gss(df[c], valid={1, 2, 3, 4, 5})
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum over 18 music items of I(response in {4,5}); any missing item => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # SES predictors
    # ----------------------------
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan), valid=(0, 50))
    df.loc[df["educ_yrs"] <= 0, "educ_yrs"] = np.nan  # guard against 0 being miscoded substantive

    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan), valid=(0, 100))

    # Income per capita: REALINC / HOMPOP (best available in provided variables)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan), valid=(0, 1e9))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan), valid=(1, 50))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    df.loc[df["inc_pc"] < 0, "inc_pc"] = np.nan

    # ----------------------------
    # Demographics / group identity
    # ----------------------------
    sex = clean_gss(df.get("sex", np.nan), valid={1, 2})
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan), valid=(18, 89))  # GSS age is typically adult; retain 89 top-code
    # (Do not force 89 to missing; treat as 89)

    # Race
    race = clean_gss(df.get("race", np.nan), valid={1, 2, 3})  # 1=White,2=Black,3=Other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace_raw"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC. In this extract ETHNIC appears numeric with many categories;
    # A robust approach: treat codes 1..9/1..99 as substantive and set missing codes to NaN.
    # Operationalize Hispanic-origin as (ETHNIC != 29 and ETHNIC != 97 etc.) is unknown; instead:
    # - If ETHNIC is binary {1,2}, treat 2 as Hispanic.
    # - Else use a conservative scheme: codes >= 20 commonly denote Hispanic origin in GSS ETHNIC;
    #   in the sample preview we see values like 29, 21, 97, 8. We'll treat 20–29 as Hispanic.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best-effort for this dataset: treat 20-29 as Hispanic-origin (matches preview codes 21,29).
            df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 20) & (eth <= 29)).astype(float))

    # Make race/ethnicity mutually exclusive with White non-Hispanic as reference:
    # - black: from RACE==2
    # - hispanic: from ETHNIC (can overlap race); paper likely uses mutually exclusive categories,
    #   so set black=0 and otherrace=0 for Hispanics, and define "other race" as non-Black, non-White, non-Hispanic.
    hisp = df["hispanic"]
    black = df["black"]
    oth_raw = df["otherrace_raw"]

    # If hispanic is missing, keep others as-is (to avoid extra missingness propagation).
    # If hispanic==1, force black=0 and otherrace=0 (Hispanic category overrides).
    df.loc[hisp == 1, "black"] = 0.0
    df.loc[hisp == 1, "otherrace_raw"] = 0.0

    # Other race: RACE==3 and not Hispanic
    df["otherrace"] = df["otherrace_raw"]
    df.loc[hisp == 1, "otherrace"] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan), valid={1, 2, 3, 4, 5})  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: best-effort with available variables only.
    # Use common GSS DENOM groupings where 1=Baptist, 2=Methodist, 3=Lutheran, 4=Presbyterian,
    # 5=Episcopal, 6=Other, 7=None. Conservative prot approximated as Protestant & (Baptist or Other)
    # This is not perfect but avoids inducing huge missingness.
    is_prot = (relig == 1)
    denom_valid = denom.where(denom.isin([1, 2, 3, 4, 5, 6, 7]), np.nan)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_valid.isin([1, 6])).astype(float))
    # If Protestant but denom missing/invalid, set to 0 to avoid dropping many cases
    df.loc[is_prot & denom_valid.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per provided mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance (0–15): sum of 15 "intolerant" indicators
    # Rule: item-level missing -> scale missing if ANY item missing (listwise for the scale)
    # ----------------------------
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
        # Keep plausible small integers; all else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + quick sanity checks for dummies
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

    # Dummy frequencies (helps catch reversed coding quickly)
    freq_lines = []
    for v in ["black", "hispanic", "otherrace", "female", "south", "cons_prot", "norelig"]:
        if v in df.columns:
            s = df[v]
            freq_lines.append(f"{v}: nonmissing={int(s.notna().sum())}, mean={float(s.mean(skipna=True)) if s.notna().any() else np.nan}")
            if s.notna().any():
                vc = s.value_counts(dropna=True).sort_index()
                freq_lines.append(vc.to_string())
            freq_lines.append("")
    write_text("./output/table1_dummy_frequencies.txt", "\n".join(freq_lines).strip() + "\n")

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

    # ----------------------------
    # Save human-readable outputs
    # ----------------------------
    def full_table_text(tab, title):
        cols = ["term", "b", "beta", "p", "sig"]
        t = tab[cols].copy()
        # Pretty numeric formatting
        for c in ["b", "beta", "p"]:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else (f"{v:.6g}" if c == "p" else f"{v:.6f}"))
        return title + "\n" + t.to_string(index=False) + "\n"

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", full_table_text(tab1, "Model 1 (SES): coefficients (b), standardized betas (beta), p-values, stars"))
    write_text("./output/model2_full.txt", full_table_text(tab2, "Model 2 (Demographic): coefficients (b), standardized betas (beta), p-values, stars"))
    write_text("./output/model3_full.txt", full_table_text(tab3, "Model 3 (Political intolerance): coefficients (b), standardized betas (beta), p-values, stars"))

    t1 = table1_display(tab1)
    t2 = table1_display(tab2)
    t3 = table1_display(tab3)

    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from data; standardized betas for predictors; unstandardized intercept).")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 1 (Table 1 style):")
    summary_lines.append(t1.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 2 (Table 1 style):")
    summary_lines.append(t2.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 3 (Table 1 style):")
    summary_lines.append(t3.to_string(index=False))
    summary_lines.append("")
    write_text("./output/table1_summary.txt", "\n".join(summary_lines) + "\n")

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