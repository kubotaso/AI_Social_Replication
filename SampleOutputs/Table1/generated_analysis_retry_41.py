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

    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            return s * np.nan
        return (s - mu) / sd

    def fit_model_standardized(df, dv, xcols, model_name, labels):
        """
        Fit OLS, then report standardized coefficients (beta) for predictors by
        estimating on z-scored y and z-scored X within the model-specific complete-case sample.
        Constant is from the unstandardized regression (to match table convention).
        """
        cols_needed = [dv] + xcols
        frame = df[cols_needed].copy()

        # Model-specific listwise deletion only
        frame = frame.dropna(axis=0, how="any").copy()

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
        }

        # Handle degenerate cases
        if len(frame) < 5:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        # Unstandardized model (for constant, fit stats, p-values)
        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        # Standardized model to get betas (and p-values aligned with standardized coeffs)
        yz = zscore(frame[dv].astype(float))
        Xz = pd.DataFrame({c: zscore(frame[c].astype(float)) for c in xcols}, index=frame.index)

        # Drop any columns that became all-NaN or zero-variance after z-scoring (should be rare)
        kept = [c for c in xcols if Xz[c].notna().any() and Xz[c].std(ddof=1) > 0]
        dropped = [c for c in xcols if c not in kept]
        if dropped:
            meta["dropped"] = ",".join(dropped)
        else:
            meta["dropped"] = ""

        Xzc = sm.add_constant(Xz[kept], has_constant="add")
        res_z = sm.OLS(yz, Xzc).fit()

        # Build table rows
        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""
        })

        for c in xcols:
            term = labels.get(c, c)
            b = float(res.params.get(c, np.nan))  # unstandardized slope (for debugging)
            if c in kept:
                beta = float(res_z.params.get(c, np.nan))
                p = float(res_z.pvalues.get(c, np.nan))
                rows.append({"term": term, "b": b, "beta": beta, "p": p, "sig": sig_star(p)})
            else:
                rows.append({"term": term, "b": b, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
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
    # Read + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0–18), complete-case across 18 items
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
    # SES
    # ----------------------------
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    # Keep plausible years range; do not over-restrict (paper uses years)
    df.loc[(df["educ_yrs"] < 0) | (df["educ_yrs"] > 30), "educ_yrs"] = np.nan

    df["prestg80"] = clean_gss(df.get("prestg80", np.nan))
    df.loc[(df["prestg80"] < 0) | (df["prestg80"] > 1000), "prestg80"] = np.nan  # permissive

    # Income per capita: REALINC / HOMPOP
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    # Permissive bounds (avoid accidental NA explosion)
    df.loc[(df["inc_pc"] < 0) | (df["inc_pc"] > 1e7), "inc_pc"] = np.nan

    # ----------------------------
    # Demographics / group identities
    # ----------------------------
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age"] = clean_gss(df.get("age", np.nan))
    df.loc[(df["age"] <= 0) | (df["age"] > 120), "age"] = np.nan

    # Race/ethnicity: create mutually exclusive categories with WHITE NON-HISP as reference.
    # Use ETHNIC (available in this extract) as Hispanic-origin; treat missing ETHNIC as non-Hispanic
    # to avoid collapsing N (the paper's N suggests Hispanic was not causing huge case loss).
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    eth = clean_gss(df.get("ethnic", np.nan)) if "ethnic" in df.columns else pd.Series(np.nan, index=df.index)
    # GSS often uses 1=not hispanic, 2=hispanic. Use that when present; otherwise, best-effort.
    # Crucial replication choice to match Table 1-like Ns: treat missing ethnic as NOT Hispanic (0),
    # not as NA, so Model 2 doesn't drop a third of cases just for Hispanic.
    hisp_flag = pd.Series(np.nan, index=df.index, dtype="float64")
    if "ethnic" in df.columns:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp_flag = (eth == 2).astype(float)
            hisp_flag = hisp_flag.where(eth.notna(), 0.0)  # missing -> 0
        else:
            # If multi-category, treat any >=2 as Hispanic-origin; missing -> 0
            hisp_flag = ((eth >= 2) & (eth <= 99)).astype(float)
            hisp_flag = hisp_flag.where(eth.notna(), 0.0)
    else:
        hisp_flag[:] = 0.0

    # Mutually exclusive race/ethnicity dummies:
    # Hispanic overrides race categories; then Black is non-Hispanic Black; Other is non-Hispanic Other.
    df["hispanic"] = hisp_flag

    df["black"] = np.where(race.isna(), np.nan, ((race == 2) & (df["hispanic"] == 0)).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, ((race == 3) & (df["hispanic"] == 0)).astype(float))
    # If race is missing, keep as NA (listwise rules will decide). If race present, these are 0/1.

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: Protestant + (Baptist or "other Protestant")
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # For Protestants with missing denom, treat as not conservative (0) to avoid extra missingness
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION == 3 per mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance scale (0–15), complete-case across 15 items (as described)
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
        # Keep only plausible response codes; everything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Outputs: descriptives + missingness
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4/5; DK/NA treated as missing; if any of 18 missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80",
        "female", "age", "black", "hispanic", "otherrace",
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
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "otherrace": "Other race",
        "cons_prot": "Conservative Protestant",
        "norelig": "No religion",
        "south": "Southern",
        "pol_intol": "Political intolerance",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80"]
    m2 = m1 + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model_standardized(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model_standardized(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model_standardized(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Write human-readable outputs
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    def full_table_text(tab, title):
        # Show b, beta, p, sig (for diagnostics); Table 1 display is separate.
        t = tab.copy()
        # formatting
        def fmt(x, nd=6):
            if pd.isna(x):
                return ""
            return f"{float(x):.{nd}f}"
        lines = [title, ""]
        show = t[["term", "b", "beta", "p", "sig"]].copy()
        show["b"] = show["b"].map(lambda x: fmt(x, 6))
        show["beta"] = show["beta"].map(lambda x: fmt(x, 6))
        show["p"] = show["p"].map(lambda x: fmt(x, 6))
        lines.append(show.to_string(index=False))
        lines.append("")
        return "\n".join(lines)

    write_text("./output/model1_full.txt", full_table_text(tab1, "Model 1 (SES): unstandardized b + standardized beta (via z-score regression)"))
    write_text("./output/model2_full.txt", full_table_text(tab2, "Model 2 (Demographic): unstandardized b + standardized beta (via z-score regression)"))
    write_text("./output/model3_full.txt", full_table_text(tab3, "Model 3 (Political intolerance): unstandardized b + standardized beta (via z-score regression)"))

    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)
    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Summary file
    summary_lines = []
    summary_lines.append("Table 1 replication: standardized OLS coefficients (predictors), unstandardized constants")
    summary_lines.append("")
    summary_lines.append("Fit stats:")
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
        "n_model1": int(len(frame1)),
        "n_model2": int(len(frame2)),
        "n_model3": int(len(frame3)),
    }