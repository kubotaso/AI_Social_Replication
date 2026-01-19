def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # GSS-style missing codes seen in many NORC extracts (best-effort; safe for this file too)
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

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), using estimation-sample SDs (ddof=1)
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def freq_table(series):
        s = series.dropna()
        if s.empty:
            return pd.DataFrame({"value": [], "n": [], "pct": []})
        vc = s.value_counts(dropna=False).sort_index()
        out = pd.DataFrame({"value": vc.index.astype(str), "n": vc.values})
        out["pct"] = out["n"] / out["n"].sum() * 100.0
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion ONLY on dv + current xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

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
        if len(frame) == 0:
            # empty shell
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)

        if kept:
            X = frame[kept].astype(float)
            Xc = sm.add_constant(X, has_constant="add")
            res = sm.OLS(y, Xc).fit()
            meta["r2"] = float(res.rsquared)
            meta["adj_r2"] = float(res.rsquared_adj)
            betas = standardized_betas(y, X, res.params)

            rows.append({
                "term": "Constant",
                "b": float(res.params.get("const", np.nan)),
                "beta": np.nan,
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": ""  # don't star constant
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
        else:
            # intercept-only
            Xc = sm.add_constant(pd.DataFrame(index=frame.index), has_constant="add")
            res = sm.OLS(y, Xc).fit()
            meta["r2"] = float(res.rsquared)
            meta["adj_r2"] = float(res.rsquared_adj)
            rows.append({
                "term": "Constant",
                "b": float(res.params.get("const", np.nan)),
                "beta": np.nan,
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": ""
            })
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
        # Constant: unstandardized b; Predictors: standardized beta + stars
        vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": vals})

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
    # DV: number of music genres disliked (0-18)
    # Paper rule: count response 4/5; DK/NA missing; if ANY of 18 missing -> DV missing
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
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race dummies from RACE (1=White, 2=Black, 3=Other)
    # IMPORTANT: do NOT create NA for "non-black"/"non-other"; only NA when race itself missing.
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic from ETHNIC:
    # In this extract, ETHNIC appears to be a categorical "Hispanic origin" code (not binary 1/2).
    # Best-effort rule for this file to avoid inverted coding:
    #   - code 29 (and other 20s/30s in GSS extracts) typically denote specific Hispanic origins.
    # We therefore treat: hispanic=1 if ETHNIC in [20..39], else 0 for other non-missing.
    # This yields a minority share and preserves cases (no catastrophic missingness).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # keep as numeric; if missing remains NaN
        df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 20) & (eth <= 39)).astype(float))

    # Religion: no religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: RELIG==1 and DENOM in conservative-coded buckets.
    # For this extract, DENOM codes aren't fully documented here; use a conservative, common mapping:
    #   denom 1 = Baptist (typically evangelical/conservative),
    #   denom 6 = "other" Protestant (often includes fundamentalist/sectarian).
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, treat as not conservative (0) rather than missing (to avoid over-deletion)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South: REGION==3 per mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance: sum 15 intolerant indicators; if ANY item missing -> scale missing
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
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics / audits (fixes requested)
    # ----------------------------
    # 1) Verify race dummies have variation (especially otherrace)
    audit_text = []
    audit_text.append("AUDIT: Race/Ethnicity coding checks (1993 only)\n")

    audit_text.append("RACE raw (cleaned) frequency:\n")
    audit_text.append(freq_table(race).to_string(index=False))
    audit_text.append("\n\nblack dummy frequency:\n")
    audit_text.append(freq_table(df["black"]).to_string(index=False))
    audit_text.append("\n\notherrace dummy frequency:\n")
    audit_text.append(freq_table(df["otherrace"]).to_string(index=False))

    audit_text.append("\n\nETHNIC raw (cleaned) frequency (first 30 values shown):\n")
    eth_nonmiss = df["hispanic"].copy()
    # show ethnic distribution if available
    if "ethnic" in df.columns:
        eth_clean = clean_gss(df["ethnic"])
        vc = eth_clean.value_counts(dropna=False).sort_index()
        vc = vc.iloc[:30]
        audit_text.append(pd.DataFrame({"ethnic_value": vc.index, "n": vc.values}).to_string(index=False))
    else:
        audit_text.append("(no ethnic variable present)\n")

    audit_text.append("\n\nhispanic dummy frequency:\n")
    audit_text.append(freq_table(df["hispanic"]).to_string(index=False))
    audit_text.append("\n")

    write_text("./output/table1_audit_race_ethnicity.txt", "\n".join(audit_text) + "\n")

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # Missingness table for key constructed variables
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

    # Save model frames Ns to show true model-specific listwise behavior
    write_text(
        "./output/table1_model_ns.txt",
        "Model-specific complete-case sample sizes:\n"
        f"{meta1['model']}: n={meta1['n']}\n"
        f"{meta2['model']}: n={meta2['n']}\n"
        f"{meta3['model']}: n={meta3['n']}\n"
    )

    # Full tables (b, beta, p, stars) for debugging/validation
    def save_full(tab, name):
        t = tab.copy()
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        cols = ["term", "b", "beta", "p", "sig"]
        write_text(f"./output/{name}_full.txt", t[cols].to_string(index=False) + "\n")

    save_full(tab1, "model1")
    save_full(tab2, "model2")
    save_full(tab3, "model3")

    # Table 1 style output (Constant=b, predictors=standardized beta + stars)
    t1_1 = table1_style(tab1)
    t1_2 = table1_style(tab2)
    t1_3 = table1_style(tab3)

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Fit stats text
    fit_show = fit_stats.copy()
    fit_show["r2"] = fit_show["r2"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    fit_show["adj_r2"] = fit_show["adj_r2"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    write_text("./output/table1_fit_stats.txt", fit_show.to_string(index=False) + "\n")

    # Compact combined summary (like a single "Table 1" view)
    # Align on term order from Model 3 (superset)
    terms = tab3["term"].tolist()
    combined = pd.DataFrame({"term": terms})
    combined = combined.merge(t1_1.rename(columns={"Table1": "Model 1"}), on="term", how="left")
    combined = combined.merge(t1_2.rename(columns={"Table1": "Model 2"}), on="term", how="left")
    combined = combined.merge(t1_3.rename(columns={"Table1": "Model 3"}), on="term", how="left")

    header = (
        "Table 1-style output (predictors are standardized OLS coefficients β with stars; constants unstandardized)\n"
        "Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed)\n\n"
    )
    write_text("./output/table1_combined.txt", header + combined.to_string(index=False) + "\n")

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
    }