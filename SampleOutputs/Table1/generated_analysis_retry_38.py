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

        # Shell if no data
        if len(frame) == 0 or len(kept) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
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

    def missingness_table(df, vars_):
        rows = []
        for v in vars_:
            if v not in df.columns:
                continue
            nonmiss = int(df[v].notna().sum())
            miss = int(df[v].isna().sum())
            denom = nonmiss + miss
            rows.append({
                "variable": v,
                "nonmissing": nonmiss,
                "missing": miss,
                "pct_missing": (miss / denom * 100.0) if denom else np.nan
            })
        return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)

    def freq_table(s, name):
        vc = s.value_counts(dropna=False).rename_axis(name).reset_index(name="n")
        vc["pct"] = vc["n"] / vc["n"].sum() * 100.0
        return vc

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
    # - 18 items, codes 1..5; dislike if 4 or 5
    # - DK/NA treated as missing
    # - DV missing if ANY item missing
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

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum over 18 genre items of I(response in {4,5}); if any genre item missing => DV missing.\n\n"
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

    # Race (from RACE): dummies should be 0/1 for all with non-missing RACE
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic (from ETHNIC in this extract): ensure 0/1 for all nonmissing ETHNIC
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If binary 1/2, assume 2==Hispanic.
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = (eth == 2).astype(float)
            df.loc[eth.isna(), "hispanic"] = np.nan
        else:
            # Best-effort: code 1 as not Hispanic; 2+ as Hispanic (common GSS pattern in some extracts)
            df["hispanic"] = ((eth >= 2) & (eth <= 99)).astype(float)
            df.loc[eth.isna(), "hispanic"] = np.nan

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (best-effort given available fields):
    # Protestant & denomination in {Baptist (1), Other Protestant (6)}
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, set 0 to avoid unnecessary deletion
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South (per mapping instruction: REGION==3)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators; require all 15 nonmissing
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
        # keep small integer codes only (varies by item); treat others as missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics (overall)
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss = missingness_table(df, diag_vars)
    write_text("./output/table1_missingness_overall.txt", miss.to_string(index=False) + "\n")

    # Provide quick sanity frequencies for key dummies
    dummy_freqs = {
        "female": freq_table(df["female"], "female"),
        "black": freq_table(df["black"], "black"),
        "hispanic": freq_table(df["hispanic"], "hispanic") if "hispanic" in df.columns else pd.DataFrame(),
        "otherrace": freq_table(df["otherrace"], "otherrace"),
        "cons_prot": freq_table(df["cons_prot"], "cons_prot"),
        "norelig": freq_table(df["norelig"], "norelig"),
        "south": freq_table(df["south"], "south"),
    }
    for k, tab in dummy_freqs.items():
        if isinstance(tab, pd.DataFrame) and len(tab) > 0:
            write_text(f"./output/table1_freq_{k}.txt", tab.to_string(index=False) + "\n")

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

    # Missingness within each model's analytic sample driver (which vars cause drops)
    # (Reported as missingness in the *pre-drop* data restricted to nonmissing DV)
    def model_drop_diagnostics(df, dv, xcols):
        base = df[[dv] + xcols].copy()
        # show missingness conditional on DV present (because DV construction is strict)
        base = base.loc[base[dv].notna()].copy()
        out = []
        for c in xcols:
            miss = int(base[c].isna().sum())
            n = int(len(base))
            out.append({"variable": c, "missing_given_dv": miss, "pct_missing_given_dv": (miss / n * 100.0) if n else np.nan})
        return pd.DataFrame(out).sort_values("pct_missing_given_dv", ascending=False)

    dropdiag1 = model_drop_diagnostics(df, "num_genres_disliked", m1)
    dropdiag2 = model_drop_diagnostics(df, "num_genres_disliked", m2)
    dropdiag3 = model_drop_diagnostics(df, "num_genres_disliked", m3)

    write_text("./output/table1_dropdiag_model1.txt", dropdiag1.to_string(index=False) + "\n")
    write_text("./output/table1_dropdiag_model2.txt", dropdiag2.to_string(index=False) + "\n")
    write_text("./output/table1_dropdiag_model3.txt", dropdiag3.to_string(index=False) + "\n")

    # Save full regression tables (computed; not from paper)
    def full_table_text(meta, tab):
        lines = []
        lines.append(f"{meta['model']}")
        lines.append(f"n={meta['n']}  R2={meta['r2']:.6f}  AdjR2={meta['adj_r2']:.6f}  dropped={meta['dropped']}")
        lines.append("")
        t = tab.copy()
        # nicer formatting
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        t = t[["term", "b", "beta", "p", "sig"]]
        lines.append(t.to_string(index=False))
        lines.append("")
        lines.append("Table-1-style display: Constant is unstandardized; predictors are standardized beta with stars.")
        lines.append(table1_display(tab).to_string(index=False))
        return "\n".join(lines) + "\n"

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/table1_model1_full.txt", full_table_text(meta1, tab1))
    write_text("./output/table1_model2_full.txt", full_table_text(meta2, tab2))
    write_text("./output/table1_model3_full.txt", full_table_text(meta3, tab3))

    # Combined Table 1-style matrix
    t1 = table1_display(tab1).rename(columns={"Table1": "Model 1 (SES)"})
    t2 = table1_display(tab2).rename(columns={"Table1": "Model 2 (Demographic)"})
    t3 = table1_display(tab3).rename(columns={"Table1": "Model 3 (Political intolerance)"})
    combined = t1.merge(t2, on="term", how="outer").merge(t3, on="term", how="outer")

    # Add fit stats to bottom as rows (as in many tables)
    fit_rows = pd.DataFrame({
        "term": ["N", "R²", "Adj. R²"],
        "Model 1 (SES)": [str(meta1["n"]), f"{meta1['r2']:.3f}" if pd.notna(meta1["r2"]) else "", f"{meta1['adj_r2']:.3f}" if pd.notna(meta1["adj_r2"]) else ""],
        "Model 2 (Demographic)": [str(meta2["n"]), f"{meta2['r2']:.3f}" if pd.notna(meta2["r2"]) else "", f"{meta2['adj_r2']:.3f}" if pd.notna(meta2["adj_r2"]) else ""],
        "Model 3 (Political intolerance)": [str(meta3["n"]), f"{meta3['r2']:.3f}" if pd.notna(meta3["r2"]) else "", f"{meta3['adj_r2']:.3f}" if pd.notna(meta3["adj_r2"]) else ""],
    })
    combined_out = pd.concat([combined, fit_rows], ignore_index=True)

    write_text("./output/table1_combined_table1style.txt", combined_out.to_string(index=False) + "\n")

    # Return a dict of useful outputs
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": table1_display(tab1),
        "model2_table1style": table1_display(tab2),
        "model3_table1style": table1_display(tab3),
        "combined_table1style": combined_out,
        "missingness_overall": miss,
        "dropdiag_model1": dropdiag1,
        "dropdiag_model2": dropdiag2,
        "dropdiag_model3": dropdiag3,
        "analytic_sample_model1": frame1,
        "analytic_sample_model2": frame2,
        "analytic_sample_model3": frame3,
    }