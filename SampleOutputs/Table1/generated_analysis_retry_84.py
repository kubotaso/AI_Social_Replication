def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS-style missing codes (best-effort; dataset-specific extracts vary)
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
        # model-specific complete-case selection
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
        if len(frame) == 0:
            # empty model shell
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

        # constant: unstandardized intercept
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        })

        # predictors: include even if dropped (show NaN)
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
        # Table 1 style: Constant = unstandardized b; predictors = standardized beta + stars
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

    def stepwise_n_audit(df, dv, blocks, audit_name):
        # blocks: list of (block_name, [vars...]) applied cumulatively
        rows = []
        cur = [dv]
        for bname, vars_ in blocks:
            cur = cur + list(vars_)
            n_complete = int(df[cur].dropna().shape[0])
            rows.append({"audit": audit_name, "step": bname, "vars_included": ", ".join(cur), "n_complete": n_complete})
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
    # Rule: disliked = 1 if response in {4,5}, else 0 if {1,2,3}, missing otherwise.
    # DV missing if ANY of 18 items missing (listwise on the 18 items).
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
        # keep only valid substantive 1..5; everything else missing
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items where response is 4/5; DK/NA treated as missing;\n"
        "DV set to missing if any of 18 items missing.\n\n"
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
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic (FIX): use ETHNIC with minimal missingness; treat NIU/out-of-range as not hispanic when plausible.
    # Goal: do NOT turn most respondents into NA.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth_raw = to_num(df["ethnic"])
        # First, mark hard-missing codes as NaN, but keep "inapplicable"-like values as 0 when they clearly mean "not hispanic".
        # In many extracts, ETHNIC is a category code with 1=not hispanic, 2=... hispanic origins; also sometimes 0/9 for DK.
        eth = eth_raw.copy()
        eth = eth.where(~eth.isin(list(GSS_NA_CODES)), np.nan)

        # Best-effort recode:
        # - If values include 1 and 2 (common binary-ish): 1->0, 2->1, others -> (>=2)->1.
        # - If many category codes exist: treat 1 as "not hispanic"; any code 2..99 as "hispanic origin".
        # - If ETHNIC looks like percent/other nonsense, fall back to missing.
        uniq = set(pd.unique(eth.dropna()))
        if len(uniq) > 0:
            if uniq.issubset({1.0, 2.0}):
                df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
            elif any(v in uniq for v in [1.0, 2.0]) and all((v >= 1.0 and v <= 99.0) for v in uniq):
                df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 99)).astype(float))
            else:
                # Last resort: if ETHNIC appears numeric but not in plausible code range, leave missing
                df["hispanic"] = np.nan

        # Critical "minimal missingness" rule:
        # If still largely missing, but raw has many numeric entries including 0, interpret 0 as "not hispanic"
        # (common in some exports).
        if df["hispanic"].notna().sum() < 0.5 * len(df) and eth_raw.notna().sum() > 0.5 * len(df):
            eth2 = eth_raw.copy()
            eth2 = eth2.where(~eth2.isin([7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999]), np.nan)
            # If 0 exists and 1 exists, interpret 0/1 as "not hispanic"
            uniq2 = set(pd.unique(eth2.dropna()))
            if len(uniq2) > 0 and all((v >= 0 and v <= 99) for v in uniq2) and (0.0 in uniq2 or 1.0 in uniq2):
                df["hispanic"] = np.where(eth2.isna(), np.nan, ((eth2 >= 2) & (eth2 <= 99)).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: Protestant and denom in a "conservative" bucket.
    # With limited documentation, use a conservative proxy: Baptist (1) and "other" (6).
    # If Protestant but denom missing, set to 0 to avoid dropping many.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern (REGION==3 per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15)
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
        # Keep plausible small integers; everything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + stepwise N audit
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
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

    # Stepwise audit per model variable blocks (to find where N collapses)
    m1_vars = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2_add = ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_add = ["pol_intol"]

    audit = []
    audit.append(stepwise_n_audit(df, "num_genres_disliked", [
        ("y only", []),
        ("+ SES", m1_vars),
    ], "Model 1"))
    audit.append(stepwise_n_audit(df, "num_genres_disliked", [
        ("y only", []),
        ("+ SES", m1_vars),
        ("+ Demographics", m2_add),
    ], "Model 2"))
    audit.append(stepwise_n_audit(df, "num_genres_disliked", [
        ("y only", []),
        ("+ SES", m1_vars),
        ("+ Demographics", m2_add),
        ("+ Political intolerance", m3_add),
    ], "Model 3"))
    audit_df = pd.concat(audit, ignore_index=True)
    write_text("./output/table1_stepwise_n_audit.txt", audit_df.to_string(index=False) + "\n")

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

    m1 = m1_vars
    m2 = m1 + m2_add
    m3 = m2 + m3_add

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # "Table 1 style" displays (constant unstd; predictors standardized betas with stars)
    t1 = table1_display(tab1)
    t2 = table1_display(tab2)
    t3 = table1_display(tab3)

    # Save human-readable outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    def save_model_tables(model_tag, tab, tstyle):
        write_text(f"./output/table1_{model_tag}_full.txt", tab.to_string(index=False) + "\n")
        write_text(f"./output/table1_{model_tag}_table1style.txt", tstyle.to_string(index=False) + "\n")

    save_model_tables("model1_ses", tab1, t1)
    save_model_tables("model2_demographic", tab2, t2)
    save_model_tables("model3_polintol", tab3, t3)

    # Combined Table 1-style side-by-side
    combined = t1.merge(t2, on="term", how="outer", suffixes=("_m1", "_m2"))
    combined = combined.merge(t3, on="term", how="outer")
    combined = combined.rename(columns={"Table1": "Model3"})
    combined = combined.rename(columns={"Table1_m1": "Model1", "Table1_m2": "Model2"})
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Compact textual summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from raw data; OLS; predictors reported as standardized betas; constant unstandardized)\n")
    summary_lines.append("FIT STATS:")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\nNOTES:")
    summary_lines.append("- Standardized beta computed as b * sd(x)/sd(y) on the estimation sample for each model.")
    summary_lines.append("- Stars based on two-tailed p-values from OLS: * p<.05, ** p<.01, *** p<.001.")
    summary_lines.append("- Model-specific listwise deletion is applied only to variables in that model.")
    summary_lines.append("\nSTEPWISE N AUDIT (to diagnose sample-size collapses):")
    summary_lines.append(audit_df.to_string(index=False))
    summary_text = "\n".join(summary_lines) + "\n"
    write_text("./output/table1_summary.txt", summary_text)

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "combined_table1style": combined,
        "missingness": missingness,
        "stepwise_n_audit": audit_df
    }