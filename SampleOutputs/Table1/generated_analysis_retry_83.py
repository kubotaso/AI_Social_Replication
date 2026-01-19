def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # GSS-style missing codes (best-effort; covers common DK/NA/Refused/Inap patterns)
    GSS_NA_CODES = {
        0, 7, 8, 9,
        97, 98, 99,
        997, 998, 999,
        9997, 9998, 9999
    }

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_gss(series, extra_na=()):
        s = to_num(series)
        na = set(GSS_NA_CODES) | set(extra_na)
        return s.where(~s.isin(list(na)), np.nan)

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

    def standardized_betas_from_unstd(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y) computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # drop zero-variance predictors on THIS analytic sample
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

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_from_unstd(y, X, res.params)

        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        })

        # preserve requested order (even if some dropped)
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
        # Constant: unstandardized b; Predictors: standardized beta + stars
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

    def freq_table(s, dropna=False, max_levels=50):
        vc = s.value_counts(dropna=not dropna)
        vc = vc.head(max_levels)
        return vc.to_string()

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
    # DV: Number of music genres disliked (0–18)
    # Rule: count (response in {4,5}) across 18 items; if ANY of 18 missing => DV missing
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

    # Female
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male,2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race: dummies (ref=white)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC; IMPORTANT: treat "not Hispanic" as 0, not missing.
    # We do not blanket-mark all codes >=2 as Hispanic (that caused collapse before).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth_raw = df["ethnic"]
        eth = clean_gss(eth_raw)

        # GSS ETHNIC is commonly 1..N categories; in some extracts 1=not hispanic, 2=hispanic.
        # To avoid catastrophic missingness, only use a conservative binary mapping:
        # - if values include 1 and 2 (and mostly in that set), treat 2 as Hispanic
        # - else if values include 1 and a small set including 2, still treat 2 as Hispanic
        # - else: fall back to "unknown" (keep as missing) rather than miscode.
        vals = set(pd.unique(eth.dropna()))
        if len(vals) > 0:
            if vals.issubset({1.0, 2.0}):
                df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
            else:
                # If 2 exists and 1 exists, map only 2 to Hispanic and all other *valid* codes to 0
                # This keeps missingness low while avoiding over-coding Hispanics.
                if 1.0 in vals and 2.0 in vals:
                    df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
                else:
                    # unknown coding scheme in this extract; avoid making it worse
                    df["hispanic"] = np.nan

    # Religion: norelig
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: Protestant + DENOM conservative bucket (best-effort, low-missingness)
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    # Conservative Protestant approximation (common in GSS recodes): Baptist + Other Protestant
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # Avoid dropping Protestants just because denom missing: set cons_prot=0 if prot & denom missing
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators; if ANY missing => scale missing
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
        x = x.where(x.isin([1, 2, 3, 4, 5, 6]), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics (to catch sample collapses)
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

    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "race", "ethnic", "black", "hispanic", "otherrace",
        "relig", "denom", "cons_prot", "norelig", "region", "south",
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
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Raw-code frequency sanity checks for key recodes
    sanity = []
    if "race" in df.columns:
        sanity.append("RACE (raw cleaned) frequency:\n" + freq_table(clean_gss(df["race"]).where(clean_gss(df["race"]).isin([1, 2, 3])), dropna=True) + "\n")
    if "ethnic" in df.columns:
        sanity.append("ETHNIC (raw cleaned) frequency (top 30):\n" + freq_table(clean_gss(df["ethnic"]), dropna=True, max_levels=30) + "\n")
    if "region" in df.columns:
        sanity.append("REGION (raw cleaned) frequency:\n" + freq_table(clean_gss(df["region"]), dropna=True) + "\n")
    if "relig" in df.columns:
        sanity.append("RELIG (raw cleaned) frequency:\n" + freq_table(clean_gss(df["relig"]), dropna=True) + "\n")
    if "denom" in df.columns:
        sanity.append("DENOM (raw cleaned) frequency:\n" + freq_table(clean_gss(df["denom"]), dropna=True) + "\n")
    sanity.append("Constructed dummy means (on nonmissing):\n" + pd.Series({
        "female_mean": df["female"].mean(skipna=True),
        "black_mean": df["black"].mean(skipna=True),
        "hispanic_mean": df["hispanic"].mean(skipna=True),
        "otherrace_mean": df["otherrace"].mean(skipna=True),
        "south_mean": df["south"].mean(skipna=True),
        "cons_prot_mean": df["cons_prot"].mean(skipna=True),
        "norelig_mean": df["norelig"].mean(skipna=True),
    }).to_string() + "\n")
    write_text("./output/table1_sanity_checks.txt", "\n".join(sanity))

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # Stepwise complete-case audit per model (to identify culprit variables)
    def stepwise_audit(df, dv, ordered_vars):
        rows = []
        current = [dv]
        for v in ordered_vars:
            current.append(v)
            tmp = df[current].dropna()
            rows.append({
                "added": v,
                "vars_included": len(current),
                "n_complete": int(len(tmp)),
                "pct_complete_of_year": float(len(tmp) / len(df) * 100.0) if len(df) else np.nan
            })
        return pd.DataFrame(rows)

    audit_m1 = stepwise_audit(df, "num_genres_disliked", ["educ_yrs", "inc_pc", "prestg80_v"])
    audit_m2 = stepwise_audit(df, "num_genres_disliked", ["educ_yrs", "inc_pc", "prestg80_v",
                                                         "female", "age_v", "black", "hispanic", "otherrace",
                                                         "cons_prot", "norelig", "south"])
    audit_m3 = stepwise_audit(df, "num_genres_disliked", ["educ_yrs", "inc_pc", "prestg80_v",
                                                         "female", "age_v", "black", "hispanic", "otherrace",
                                                         "cons_prot", "norelig", "south", "pol_intol"])
    write_text("./output/table1_stepwise_audit_m1.txt", audit_m1.to_string(index=False) + "\n")
    write_text("./output/table1_stepwise_audit_m2.txt", audit_m2.to_string(index=False) + "\n")
    write_text("./output/table1_stepwise_audit_m3.txt", audit_m3.to_string(index=False) + "\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Full regression tables (includes b, beta, p used to generate stars; p not shown in Table 1)
    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    # Table 1 style (what the paper prints: beta with stars; constant as b)
    t1_1 = table1_style(tab1)
    t1_2 = table1_style(tab2)
    t1_3 = table1_style(tab3)
    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Combined table for convenience
    combined = (
        t1_1.rename(columns={"Table1": "Model1"})
        .merge(t1_2.rename(columns={"Table1": "Model2"}), on="term", how="outer")
        .merge(t1_3.rename(columns={"Table1": "Model3"}), on="term", how="outer")
    )
    # Order rows in a Table-1-like way
    desired_order = ["Constant"] + [labels.get(c, c) for c in (m3)]
    combined["order"] = combined["term"].apply(lambda t: desired_order.index(t) if t in desired_order else 999)
    combined = combined.sort_values(["order", "term"]).drop(columns=["order"])
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Short human-readable summary
    summary_lines = []
    summary_lines.append("Replication summary: Table 1-style OLS with standardized coefficients (beta) for predictors.\n")
    summary_lines.append("NOTE: Table 1 in the paper does not report p-values/SE; stars here are based on conventional OLS two-tailed p-values.\n")
    summary_lines.append("\nFit statistics:\n" + fit_stats.to_string(index=False) + "\n")
    summary_lines.append("\nCombined Table 1-style coefficients (beta for predictors, b for constant):\n" + combined.to_string(index=False) + "\n")
    summary_lines.append("\nKey distribution checks:\n")
    summary_lines.append("DV (num_genres_disliked) describe:\n" + dv_desc.to_string() + "\n")
    summary_lines.append("\nSee ./output/table1_stepwise_audit_m*.txt for where complete-case N drops.\n")
    write_text("./output/table1_summary.txt", "\n".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "table1_combined": combined,
        "missingness": missingness,
        "audit_m1": audit_m1,
        "audit_m2": audit_m2,
        "audit_m3": audit_m3
    }