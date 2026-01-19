def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS "not ascertained / don't know / no answer / not applicable" style codes.
    # Keep this conservative: treat only clearly-non-substantive codes as missing.
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

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_ols(df, dv, xcols, labels, model_name):
        # model-specific listwise deletion ONLY on variables in the model
        use = df[[dv] + xcols].copy()
        use = use.dropna(axis=0, how="any").copy()

        # Drop any constant predictors (avoid runtime errors)
        kept, dropped = [], []
        for c in xcols:
            if use[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(use)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ",".join(dropped) if dropped else ""
        }

        rows = []
        if len(use) == 0:
            # empty shell
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), use

        y = use[dv].astype(float)
        X = use[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        # Output rows: constant first, then in original xcols order
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star the constant in display
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

        return meta, pd.DataFrame(rows), use

    def table1_style(tab, include_replication_p=False):
        # Paper-style: unstandardized constant, standardized betas for predictors.
        # Feedback fix: by default do NOT present p-values/SEs; only show stars and betas.
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            row = {"term": r["term"], "Table1": val}
            if include_replication_p:
                row["p_replication"] = r.get("p", np.nan)
            out.append(row)
        return pd.DataFrame(out)

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
    # Rule: for each item, disliked=1 if response in {4,5}; 0 if in {1,2,3}; else missing.
    # DV missing if ANY of the 18 items missing.
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
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

    # ----------------------------
    # Predictors for Table 1
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

    # Hispanic (derive from ETHNIC; ensure non-Hispanic is coded 0, not missing)
    # The file includes 'ethnic'. Treat any non-missing value that clearly indicates "not Hispanic" as 0.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Common binary coding is 1=not Hispanic, 2=Hispanic.
        # If it isn't binary, still do best-effort:
        #   - code 1 as not Hispanic
        #   - code >=2 as Hispanic-origin
        # Do not force missing for non-Hispanic; only missing if ETHNIC missing.
        df["hispanic"] = np.where(
            eth.isna(),
            np.nan,
            np.where(eth == 1, 0.0, 1.0)
        )

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant approximation (must be computable with available vars):
    # Protestant (RELIG==1) and DENOM in conservative-leaning categories.
    # With this extract, use a minimal, stable approximation:
    #   - Baptist (1) and "Other" (6) (often includes evangelical/fundamentalist)
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, set to 0 (avoid unnecessary listwise loss)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern
    region = clean_gss(df.get("region", np.nan))
    # Keep plausible codes; treat others as missing
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
    missing_tol = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # keep only plausible small integers; else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    # Primary (strict) construction: require all 15 items (matches many index descriptions)
    df["pol_intol_strict"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol_strict"] = np.nan

    # Secondary (lenient) construction (to diagnose N collapse): allow partial, rescale to 0–15 if enough items answered.
    # This can be toggled if strict yields extremely low N.
    answered = tol_df.notna().sum(axis=1)
    raw_sum = tol_df.sum(axis=1, min_count=1)
    df["pol_intol_lenient"] = np.where(
        answered >= 12,  # require most items answered; chosen to be faithful while avoiding extreme loss
        (raw_sum / answered) * 15.0,
        np.nan
    )

    # Choose intolerance variable:
    # If strict produces very small non-missing vs lenient, use lenient; otherwise strict.
    strict_nm = int(df["pol_intol_strict"].notna().sum())
    lenient_nm = int(df["pol_intol_lenient"].notna().sum())
    df["pol_intol"] = df["pol_intol_strict"]
    pol_choice = "strict"
    if strict_nm < 0.75 * lenient_nm:
        df["pol_intol"] = df["pol_intol_lenient"]
        pol_choice = "lenient_rescaled(>=12 answered)"

    # ----------------------------
    # Labels
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

    # ----------------------------
    # Diagnostics outputs
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # Missingness table (pre-model)
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
    write_text(
        "./output/table1_missingness.txt",
        "Missingness (within YEAR==1993):\n"
        f"Political intolerance construction chosen: {pol_choice}\n\n"
        + missingness.to_string(index=False)
        + "\n"
    )

    # Stepwise complete-case audit per model variable set
    def complete_case_n(cols):
        return int(df[cols].dropna().shape[0])

    m1_cols = ["num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v"]
    m2_cols = m1_cols + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_cols = m2_cols + ["pol_intol"]
    audit = pd.DataFrame([
        {"model": "Model 1 (SES)", "n_complete": complete_case_n(m1_cols)},
        {"model": "Model 2 (Demographic)", "n_complete": complete_case_n(m2_cols)},
        {"model": "Model 3 (Political intolerance)", "n_complete": complete_case_n(m3_cols)},
    ])
    write_text("./output/table1_complete_case_audit.txt", audit.to_string(index=False) + "\n")

    # ----------------------------
    # Models (fit separately with correct model-specific listwise deletion)
    # ----------------------------
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, used1 = fit_ols(df, "num_genres_disliked", m1, labels, "Model 1 (SES)")
    meta2, tab2, used2 = fit_ols(df, "num_genres_disliked", m2, labels, "Model 2 (Demographic)")
    meta3, tab3, used3 = fit_ols(df, "num_genres_disliked", m3, labels, "Model 3 (Political intolerance)")

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Save tables (human-readable)
    # Feedback fix: "Table 1 style" omits p-values/SEs by default.
    # Full regression tables with p are saved separately for debugging, but clearly labeled as replication output.
    # ----------------------------
    t1_1 = table1_style(tab1, include_replication_p=False)
    t1_2 = table1_style(tab2, include_replication_p=False)
    t1_3 = table1_style(tab3, include_replication_p=False)

    # Combined table-like output
    combined = t1_1[["term", "Table1"]].rename(columns={"Table1": "Model 1"})
    combined = combined.merge(t1_2[["term", "Table1"]].rename(columns={"Table1": "Model 2"}), on="term", how="outer")
    combined = combined.merge(t1_3[["term", "Table1"]].rename(columns={"Table1": "Model 3"}), on="term", how="outer")

    # Save
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text(
        "./output/table1_combined_table1style.txt",
        "Table 1 style output:\n"
        "- Constant shown as unstandardized intercept.\n"
        "- Predictors shown as standardized beta with significance stars.\n"
        "- p-values/SEs are not displayed here (paper-style).\n\n"
        + combined.to_string(index=False)
        + "\n"
    )

    # Replication full tables (explicitly labeled)
    def full_table_text(tab, model_name):
        t = tab.copy()
        # Round for readability
        for col in ["b", "beta", "p"]:
            if col in t.columns:
                t[col] = pd.to_numeric(t[col], errors="coerce")
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        cols = ["term", "b", "beta", "p", "sig"]
        return f"{model_name} (replication full output; p-values computed from this run)\n\n" + t[cols].to_string(index=False) + "\n"

    write_text("./output/model1_full_replication.txt", full_table_text(tab1, "Model 1 (SES)"))
    write_text("./output/model2_full_replication.txt", full_table_text(tab2, "Model 2 (Demographic)"))
    write_text("./output/model3_full_replication.txt", full_table_text(tab3, "Model 3 (Political intolerance)"))

    # Summary text file
    summary = []
    summary.append("Replication run summary (computed from provided data)\n")
    summary.append(f"Data source: {data_source}\n")
    summary.append("Filter: YEAR == 1993\n\n")
    summary.append(f"Political intolerance construction chosen: {pol_choice}\n")
    summary.append("\nFit statistics:\n")
    summary.append(fit_stats.to_string(index=False))
    summary.append("\n\nComplete-case audit (should align with fitted n):\n")
    summary.append(audit.to_string(index=False))
    summary.append("\n\nDV descriptives (within YEAR==1993):\n")
    summary.append(dv_desc.to_string())
    summary.append("\n")

    write_text("./output/summary.txt", "\n".join(summary))

    return {
        "fit_stats": fit_stats,
        "table1_combined": combined,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "missingness": missingness,
        "complete_case_audit": audit,
    }