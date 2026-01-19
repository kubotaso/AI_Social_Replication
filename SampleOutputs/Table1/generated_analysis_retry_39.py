def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative set of common GSS missing codes; applied to numeric variables.
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
    # Dislike indicators: 1 if response in {4,5}, 0 if in {1,2,3}, else missing.
    # DV set missing if ANY of 18 items missing (listwise for DV construction).
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
        "Construction: count across 18 genres where response is 4/5; DK/NA are missing; if any genre missing => DV missing.\n\n"
        + dv_desc.to_string()
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

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/Ethnicity: create mutually exclusive categories to avoid overlap/collinearity
    # Use ETHNIC (Hispanic origin) where available; treat missing as non-Hispanic (0) to avoid large N loss.
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=White, 2=Black, 3=Other

    eth = None
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
    # Best-effort: 2 indicates Hispanic in many GSS extracts; if binary {1,2}, use that.
    # Otherwise treat codes >=2 as Hispanic-origin. Missing => assume non-Hispanic to preserve sample size.
    if eth is None:
        hisp = pd.Series(0.0, index=df.index)
    else:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = (eth == 2).astype(float)
            hisp.loc[eth.isna()] = 0.0
        else:
            hisp = ((eth >= 2) & (eth <= 99)).astype(float)
            hisp.loc[eth.isna()] = 0.0
    df["hispanic"] = hisp

    # Mutually exclusive categories with White non-Hispanic as reference:
    # Hispanic overrides race.
    df["black"] = np.where(race.isna(), np.nan, 0.0)
    df["otherrace"] = np.where(race.isna(), np.nan, 0.0)

    is_hisp = (df["hispanic"] == 1.0)
    df.loc[~race.isna() & ~is_hisp & (race == 2), "black"] = 1.0
    df.loc[~race.isna() & ~is_hisp & (race == 3), "otherrace"] = 1.0
    # If race missing but hispanic==1, keep hispanic=1 and leave race dummies as missing? That would drop.
    # To avoid dropping due to missing race when Hispanic is known, set race dummies to 0 in that case.
    df.loc[race.isna() & is_hisp, ["black", "otherrace"]] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))

    # Conservative Protestant: Protestant + denomination proxy.
    # Keep missing denom among Protestants as 0 to avoid large N loss.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15), allow partial completion:
    # Sum intolerant responses across non-missing items.
    # Require at least MIN_ANSWERED items to reduce noise and better match reported N.
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
        x = clean_gss(df[c])
        # Keep plausible small integers; everything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol.notna().sum(axis=1)
    tol_sum = tol.sum(axis=1, min_count=1)

    MIN_ANSWERED = 10  # permissive but avoids extreme missingness
    df["pol_intol"] = tol_sum.where(answered >= MIN_ANSWERED, np.nan)

    # ----------------------------
    # Diagnostics: missingness and key distributions
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        nonmiss = int(df[v].notna().sum()) if v in df.columns else 0
        miss = int(df[v].isna().sum()) if v in df.columns else 0
        denomv = (nonmiss + miss)
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / denomv * 100.0) if denomv else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Check variation of race dummies on each model sample later; write an overall snapshot too
    def freq_str(s):
        vc = s.value_counts(dropna=False)
        return vc.to_string()

    write_text(
        "./output/table1_race_ethnic_diagnostics.txt",
        "Overall frequency snapshots (1993 only):\n\n"
        f"hispanic:\n{freq_str(df['hispanic'])}\n\n"
        f"black:\n{freq_str(df['black'])}\n\n"
        f"otherrace:\n{freq_str(df['otherrace'])}\n\n"
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
        "pol_intol": "Political intolerance",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Write model-sample race variation diagnostics to ensure "Other race" is not constant/dropped
    def sample_diag(frame, name):
        parts = [f"{name} analytic-sample diagnostics (n={len(frame)}):"]
        for v in ["black", "hispanic", "otherrace"]:
            if v in frame.columns:
                parts.append(f"\n{v} value_counts:\n{frame[v].value_counts(dropna=False).to_string()}")
        return "\n".join(parts) + "\n"

    write_text("./output/table1_model_sample_diagnostics.txt",
               sample_diag(frame1, "Model 1") + "\n" +
               sample_diag(frame2, "Model 2") + "\n" +
               sample_diag(frame3, "Model 3"))

    # Full tables for checking (b, beta, p, sig)
    def full_table_text(meta, tab):
        lines = []
        lines.append(f"{meta['model']}")
        lines.append(f"N = {meta['n']}")
        lines.append(f"R^2 = {meta['r2']:.6f}" if pd.notna(meta["r2"]) else "R^2 = NA")
        lines.append(f"Adj R^2 = {meta['adj_r2']:.6f}" if pd.notna(meta["adj_r2"]) else "Adj R^2 = NA")
        if meta.get("dropped"):
            lines.append(f"Dropped (zero variance): {meta['dropped']}")
        lines.append("")
        lines.append(tab.to_string(index=False))
        lines.append("")
        return "\n".join(lines)

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/table1_model1_full.txt", full_table_text(meta1, tab1))
    write_text("./output/table1_model2_full.txt", full_table_text(meta2, tab2))
    write_text("./output/table1_model3_full.txt", full_table_text(meta3, tab3))

    # Table 1-style displays (constant unstd; predictors standardized betas + stars)
    t1_1 = table1_display(tab1)
    t1_2 = table1_display(tab2)
    t1_3 = table1_display(tab3)

    write_text("./output/table1_model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Single combined "paper-like" summary
    summary_lines = []
    summary_lines.append("Table 1 replication outputs (computed from raw data; standardized OLS betas shown for predictors).")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 1 (SES) - Table 1 style:")
    summary_lines.append(t1_1.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 2 (Demographic) - Table 1 style:")
    summary_lines.append(t1_2.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 3 (Political intolerance) - Table 1 style:")
    summary_lines.append(t1_3.to_string(index=False))
    summary_lines.append("")
    write_text("./output/table1_summary.txt", "\n".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "missingness": missingness
    }