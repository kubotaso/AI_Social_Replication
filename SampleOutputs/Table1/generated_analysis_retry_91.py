def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(series, extra_na=()):
        x = to_num(series)
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

    def standardized_betas_on_sample(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y) computed on the estimation sample
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
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_on_sample(y, X, res.params)

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

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
        # Constant: unstandardized b; Predictors: standardized beta with stars
        out_vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out_vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out_vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out_vals})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # -----------------------------
    # Read data + restrict to 1993
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # -----------------------------
    # DV: Number of music genres disliked (0-18)
    # - For each of 18 items: disliked=1 if 4/5, 0 if 1/2/3, missing otherwise
    # - DV missing if ANY of 18 items missing
    # -----------------------------
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

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum across 18 genres of I(response in {4,5}); DK/NA -> missing; "
        "if any of 18 items missing -> DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # -----------------------------
    # Predictors (Table 1)
    # -----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))

    # Income per capita:
    # Use REALINC if available; otherwise fall back to INCOME.
    # Then divide by HOMPOP. Keep missing only if components are missing/invalid.
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan

    if "realinc" in df.columns:
        inc_base = clean_gss(df["realinc"])
    else:
        inc_base = clean_gss(df.get("income", np.nan))
    df["inc_base"] = inc_base
    df["inc_pc"] = df["inc_base"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male, 2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: enforce mutually exclusive categories with White non-Hispanic as reference
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=White, 2=Black, 3=Other

    # Hispanic indicator from ETHNIC if available
    hisp = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Common binary coding: 1=not Hispanic, 2=Hispanic
        # If not binary, treat code==1 as not Hispanic, any other positive code as Hispanic (best-effort).
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            hisp = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 99)).astype(float))
    df["hispanic"] = hisp

    # Build mutually exclusive dummies:
    # - If Hispanic is known (0/1), it overrides race: Hispanic=1 => black=0, otherrace=0.
    # - If Hispanic is missing, allow race dummies but keep Hispanic missing.
    df["black"] = np.nan
    df["otherrace"] = np.nan

    # Start with race-based dummies where race is known
    df.loc[race.notna(), "black"] = (race[race.notna()] == 2).astype(float)
    df.loc[race.notna(), "otherrace"] = (race[race.notna()] == 3).astype(float)

    # Override for known Hispanics
    df.loc[df["hispanic"] == 1, ["black", "otherrace"]] = 0.0

    # If Hispanic known as 0, keep race dummies as-is (already set)
    # If Hispanic missing, keep Hispanic missing but do not force race dummies missing beyond race missing

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant proxy: RELIG==1 and DENOM in (Baptist=1, Other Protestant=6)
    # If Protestant but denom missing, set cons_prot=0 to avoid unnecessary case loss.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South: REGION == 3 per mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15):
    # IMPORTANT FIX: do NOT require all 15 items nonmissing.
    # Use sum over available items; require at least MIN_TOL_ITEMS answered to reduce extra case loss.
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
        # Keep plausible codes only; otherwise missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol_df.notna().sum(axis=1)
    tol_sum = tol_df.sum(axis=1, skipna=True)

    MIN_TOL_ITEMS = 12  # pragmatic to reduce missingness inflation vs strict 15/15
    df["pol_intol"] = np.where(answered >= MIN_TOL_ITEMS, tol_sum, np.nan)
    # ensure range sanity
    df.loc[(df["pol_intol"] < 0) | (df["pol_intol"] > 15), "pol_intol"] = np.nan

    # -----------------------------
    # Diagnostics: missingness + distributions
    # -----------------------------
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

    # Additional quick checks for race/ethnicity exclusivity (on nonmissing)
    race_check = df[["black", "hispanic", "otherrace"]].copy()
    race_check["sum_flags"] = race_check[["black", "hispanic", "otherrace"]].sum(axis=1, skipna=False)
    race_check_desc = pd.Series({
        "n_nonmissing_all3": int(race_check[["black", "hispanic", "otherrace"]].notna().all(axis=1).sum()),
        "pct_sum_gt1_among_complete": float(
            (race_check.loc[race_check[["black", "hispanic", "otherrace"]].notna().all(axis=1), "sum_flags"] > 1).mean()
            if race_check[["black", "hispanic", "otherrace"]].notna().all(axis=1).any()
            else np.nan
        ),
        "mean_black": float(df["black"].mean(skipna=True)) if "black" in df else np.nan,
        "mean_hispanic": float(df["hispanic"].mean(skipna=True)) if "hispanic" in df else np.nan,
        "mean_otherrace": float(df["otherrace"].mean(skipna=True)) if "otherrace" in df else np.nan,
    })
    write_text("./output/table1_race_ethnicity_checks.txt", race_check_desc.to_string() + "\n")

    pol_desc = df["pol_intol"].describe()
    write_text(
        "./output/table1_pol_intol_descriptives.txt",
        f"Political intolerance (0–15)\nConstruction: sum of 15 intolerance indicators; require >= {MIN_TOL_ITEMS} answered items.\n\n"
        + pol_desc.to_string()
        + "\n"
    )

    # -----------------------------
    # Models (Table 1)
    # -----------------------------
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
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Full tables
    def fmt_full(tab):
        t = tab.copy()
        # nicer formatting
        for c in ["b", "beta", "p"]:
            if c in t.columns:
                t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        return t

    write_text("./output/table1_model1_full.txt", fmt_full(tab1).to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", fmt_full(tab2).to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", fmt_full(tab3).to_string(index=False) + "\n")

    # Table1-style display
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined Table 1 style (wide)
    combined = pd.DataFrame({"term": t1["term"]})
    combined = combined.merge(t1.rename(columns={"Table1": "Model 1 (SES)"}), on="term", how="left")
    combined = combined.merge(t2.rename(columns={"Table1": "Model 2 (Demographic)"}), on="term", how="left")
    combined = combined.merge(t3.rename(columns={"Table1": "Model 3 (Political intolerance)"}), on="term", how="left")

    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Short run summary
    summary_lines = []
    summary_lines.append("Replication run summary (computed from provided data)\n")
    summary_lines.append("DV: Number of music genres disliked (0–18)\n")
    summary_lines.append(f"Political intolerance: sum of 15 intolerance items; require >= {MIN_TOL_ITEMS} items answered\n\n")
    summary_lines.append("Fit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nTable 1 style (standardized betas; unstandardized constant):\n")
    summary_lines.append(combined.to_string(index=False))
    summary_lines.append("\n")
    write_text("./output/table1_summary.txt", "\n".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined": combined,
        "missingness": missingness,
        "dv_descriptives": dv_desc,
        "pol_intol_descriptives": pol_desc,
    }