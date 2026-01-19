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
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0) else b * (sdx / sdy)
        return out

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_model(df, dv, xcols, model_name, labels, fail_on_drop=True):
        """
        Model-specific listwise deletion on dv + xcols only.
        If any predictor is dropped due to zero-variance/collinearity, fail (per feedback).
        """
        frame = df[[dv] + xcols].copy()
        n0 = len(frame)
        frame = frame.dropna(axis=0, how="any").copy()
        n1 = len(frame)

        # Zero-variance check on estimation sample
        zero_var = [c for c in xcols if frame[c].nunique(dropna=True) <= 1]

        # Collinearity / rank check (including intercept)
        X = frame[xcols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        # Rank check: if rank < num columns => perfect collinearity
        rank = np.linalg.matrix_rank(Xc.to_numpy())
        k = Xc.shape[1]
        collinear = (rank < k)

        if fail_on_drop and (len(zero_var) > 0 or collinear):
            msg = (
                f"{model_name} failed due to predictor dropping risk.\n"
                f"Start rows: {n0}, after listwise deletion: {n1}\n"
                f"Zero-variance predictors on estimation sample: {zero_var}\n"
                f"Design matrix rank: {rank} < {k} (collinearity={collinear})\n"
                f"Tip: inspect coding/variation of these variables in the model-specific sample.\n"
            )
            write_text(f"./output/{model_name.replace(' ', '_').lower()}_FAIL.txt", msg)
            raise RuntimeError(msg)

        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "dropped": ""  # should remain empty if fail_on_drop=True
        }

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        })

        for c in xcols:
            p = float(res.pvalues.get(c, np.nan))
            rows.append({
                "term": labels[c],
                "b": float(res.params.get(c, np.nan)),
                "beta": float(betas.get(c, np.nan)),
                "p": p,
                "sig": sig_star(p)
            })

        tab = pd.DataFrame(rows)
        return meta, tab, frame, res

    def table1_style(tab):
        # Constant: unstandardized b; predictors: standardized beta + stars
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def combine_table_union(model_tabs, model_names):
        # Union of terms across models; blank where not present
        all_terms = []
        for t in model_tabs:
            all_terms.extend(list(t["term"].values))
        all_terms = list(dict.fromkeys(all_terms))  # preserve first-seen order

        out = pd.DataFrame({"term": all_terms})
        for name, t in zip(model_names, model_tabs):
            tt = t.set_index("term")["Table1"]
            out[name] = out["term"].map(tt).fillna("")
        return out

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
    # Rule: count of 18 items rated 4/5; if ANY of 18 items missing => DV missing.
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

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum over 18 genre ratings; disliked=4/5; 1/2/3=not disliked; DK/NA missing.\n"
        "Listwise for DV: if any of the 18 items is missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1)
    # Keep internal names stable; labels separate.
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc"] = clean_gss(df.get("realinc", np.nan))
    df["hompop"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age"] <= 0, "age"] = np.nan

    # Race/ethnicity: make dummies non-missing wherever base variables are present.
    # Use RACE for Black/Other, ETHNIC for Hispanic (best available in provided data).
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If binary {1,2}: 1=not hispanic, 2=hispanic
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Conservative fallback: treat 1 as non-Hispanic and any other positive code as Hispanic-origin.
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))

    # Ensure we do not propagate missingness unnecessarily:
    # If ETHNIC is missing but RACE is present, Hispanic remains missing and will drop case.
    # This mirrors listwise deletion, but avoids accidental NA for known non-Hispanics.
    # (No extra action needed; above coding already makes non-Hispanics 0 when ETHNIC is known.)

    # Religion dummies
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (best-effort using RELIG+DENOM): Protestant and denom in (1 Baptist, 6 Other).
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # Avoid dropping Protestants with missing denom: set 0 for Protestant with denom missing
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South dummy
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance 0–15
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
        # Keep plausible codes as in documentation; otherwise missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + quick checks for dummy variance
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80",
        "female", "age", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        nonmiss = int(df[v].notna().sum()) if v in df.columns else 0
        miss = int(df[v].isna().sum()) if v in df.columns else 0
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Dummy variance checks (do not fail here; fail happens at model fit)
    def freq01(series):
        s = series.dropna()
        return {"n": int(len(s)), "mean": float(s.mean()) if len(s) else np.nan, "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan}

    dummy_report = {
        "female": freq01(df["female"]),
        "black": freq01(df["black"]),
        "hispanic": freq01(df["hispanic"]) if "hispanic" in df.columns else {},
        "otherrace": freq01(df["otherrace"]),
        "cons_prot": freq01(df["cons_prot"]),
        "norelig": freq01(df["norelig"]),
        "south": freq01(df["south"]),
    }
    write_text("./output/table1_dummy_diagnostics.txt", pd.DataFrame(dummy_report).T.to_string() + "\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    labels = {
        "educ_yrs": "Education (years)",
        "inc_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "otherrace": "Other race",
        "cons_prot": "Conservative Protestant",
        "norelig": "No religion",
        "south": "Southern",
        "pol_intol": "Political intolerance (0–15)",
    }

    # Ensure all required columns exist
    for k in labels.keys():
        if k not in df.columns:
            df[k] = np.nan

    m1 = ["educ_yrs", "inc_pc", "prestg80"]
    m2 = m1 + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1, res1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels, fail_on_drop=True)
    meta2, tab2, frame2, res2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels, fail_on_drop=True)
    meta3, tab3, frame3, res3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels, fail_on_drop=True)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save full regression tables (b, beta, p, stars)
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    # Table 1 style outputs
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    combined = combine_table_union([t1, t2, t3], ["Model 1", "Model 2", "Model 3"])
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Also save a compact human-readable summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from raw data; standardized betas reported for predictors).\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nModel 1 (SES) - Table 1 style:\n" + t1.to_string(index=False))
    summary_lines.append("\n\nModel 2 (Demographic) - Table 1 style:\n" + t2.to_string(index=False))
    summary_lines.append("\n\nModel 3 (Political intolerance) - Table 1 style:\n" + t3.to_string(index=False))
    summary_lines.append("\n\nCombined (union of terms across models):\n" + combined.to_string(index=False))
    write_text("./output/table1_summary.txt", "\n".join(summary_lines) + "\n")

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined_table1style": combined,
        "missingness": missingness
    }