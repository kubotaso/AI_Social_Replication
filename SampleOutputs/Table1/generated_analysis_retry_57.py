def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative, broadly correct GSS missing codes for numeric fields
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

    def safe_value_counts(s):
        vc = s.value_counts(dropna=False)
        # display-friendly index
        vc.index = [("NaN" if pd.isna(i) else str(int(i)) if float(i).is_integer() else str(i)) for i in vc.index]
        return vc

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        dropped = []
        kept = []
        for c in xcols:
            nun = frame[c].nunique(dropna=True)
            if nun <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ",".join(dropped) if dropped else "",
        }

        # If no data or no varying predictors, return empty-ish table
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

        # Constant: unstandardized
        rows.append(
            {"term": "Constant", "b": float(res.params.get("const", np.nan)), "beta": np.nan,
             "p": float(res.pvalues.get("const", np.nan)), "sig": ""}  # no stars for constant
        )

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append(
                    {"term": term, "b": float(res.params.get(c, np.nan)), "beta": float(betas.get(c, np.nan)),
                     "p": p, "sig": sig_star(p)}
                )
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
        # Table 1: standardized betas for predictors (with stars), unstandardized constant
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def combine_table1(m1_t1, m2_t1, m3_t1):
        a = m1_t1.rename(columns={"Table1": "Model 1"})
        b = m2_t1.rename(columns={"Table1": "Model 2"})
        c = m3_t1.rename(columns={"Table1": "Model 3"})
        out = a.merge(b, on="term", how="outer").merge(c, on="term", how="outer")
        # Use a stable, sensible order: constant first then all other terms as they appear
        order = ["Constant"]
        for t in list(a["term"]) + list(b["term"]) + list(c["term"]):
            if t not in order:
                order.append(t)
        out["__ord"] = out["term"].map({t: i for i, t in enumerate(order)}).fillna(10_000).astype(int)
        out = out.sort_values("__ord").drop(columns="__ord")
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
    # Rule: count items == 4 or 5; if any of 18 missing => DV missing.
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
    # SES predictors
    # ----------------------------
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # ----------------------------
    # Demographic / group identity controls
    # ----------------------------
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan
    # Keep 89 top-code as 89, as-is.

    # Race/ethnicity: implement mutually exclusive categories to avoid collinearity/zero-variance drops
    # Use ETHNIC as Hispanic-origin indicator when available: 1=not hispanic, 2=hispanic (common in GSS extracts).
    # If ETHNIC is not binary, treat >=2 as Hispanic-origin (best-effort).
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other

    eth = None
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
    # If eth is present but non-informative (all missing), treat as absent
    if eth is not None and eth.notna().sum() == 0:
        eth = None

    hisp_flag = pd.Series(np.nan, index=df.index, dtype="float")
    if eth is not None:
        u = set(pd.unique(eth.dropna()))
        if u and u.issubset({1.0, 2.0}):
            hisp_flag = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # best-effort: 1 => not hispanic, >=2 => hispanic origin
            hisp_flag = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))
        hisp_flag = pd.Series(hisp_flag, index=df.index, dtype="float")

    # Build mutually exclusive dummies with White non-Hispanic as reference
    # If Hispanic info missing, keep as missing (causes listwise deletion in models using Hispanic).
    df["black"] = np.nan
    df["hispanic"] = np.nan
    df["otherrace"] = np.nan

    # Start with missing where RACE missing
    valid_race = race.notna()

    # Determine Hispanic category (takes precedence over race categories)
    # If hisp_flag is missing but race is present, keep Hispanic as missing to reflect unknown origin;
    # this matches the idea that Hispanic isn't inferable from race alone in this extract.
    is_hisp = (hisp_flag == 1.0)
    known_hisp = hisp_flag.notna()

    # Initialize reference-category coding when both race and hispanic are known
    known_both = valid_race & known_hisp
    df.loc[known_both, "black"] = 0.0
    df.loc[known_both, "hispanic"] = 0.0
    df.loc[known_both, "otherrace"] = 0.0

    # Hispanic: 1 if Hispanic-origin (regardless of race)
    df.loc[known_both & is_hisp, "hispanic"] = 1.0
    # If Hispanic, keep black/otherrace at 0.0 (mutually exclusive scheme)

    # Non-Hispanic: classify by race
    non_hisp = known_both & (hisp_flag == 0.0)
    df.loc[non_hisp & (race == 2), "black"] = 1.0
    df.loc[non_hisp & (race == 3), "otherrace"] = 1.0
    # White non-Hispanic is all zeros (reference)

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: approximation using RELIG==1 and DENOM in {1=Baptist, 6=Other Protestant}
    # Keep denom missing among Protestants as 0 to avoid unnecessary deletions.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per provided mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance scale (0–15), with strict item-level missing handling
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
        # keep only plausible small integer response codes, else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics outputs
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: 18 items; disliked=4/5; DK/NA treated missing; any missing item => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # Missingness summary (in 1993 sample)
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

    # Frequency tables for key categorical constructions (to debug race/ethnicity especially)
    freq_text = []
    if "race" in df.columns:
        freq_text.append("Raw RACE value counts (cleaned):\n")
        freq_text.append(safe_value_counts(race).to_string() + "\n\n")
    if eth is not None:
        freq_text.append("Raw ETHNIC value counts (cleaned):\n")
        freq_text.append(safe_value_counts(eth).to_string() + "\n\n")
    freq_text.append("Constructed dummies value counts (including NaN):\n")
    for v in ["black", "hispanic", "otherrace", "female", "south", "norelig", "cons_prot"]:
        freq_text.append(f"{v}:\n{df[v].value_counts(dropna=False).to_string()}\n\n")
    write_text("./output/table1_frequencies.txt", "".join(freq_text))

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

    # Save fit stats
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Save "full" coefficient tables (diagnostic; includes b, beta, p)
    def tab_to_text(tab):
        t = tab.copy()
        # round for readability; keep p in scientific
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        return t.to_string(index=False)

    write_text("./output/table1_model1_full.txt", tab_to_text(tab1) + "\n")
    write_text("./output/table1_model2_full.txt", tab_to_text(tab2) + "\n")
    write_text("./output/table1_model3_full.txt", tab_to_text(tab3) + "\n")

    # Save Table 1 style tables (betas + stars; constant unstandardized)
    t1_1 = table1_style(tab1)
    t1_2 = table1_style(tab2)
    t1_3 = table1_style(tab3)

    write_text("./output/table1_model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    combined = combine_table1(t1_1, t1_2, t1_3)
    write_text("./output/table1_combined_table.txt", combined.to_string(index=False) + "\n")

    # Also write a compact summary text
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from microdata)\n")
    summary_lines.append("Coefficients shown in Table1-style output are standardized betas (β) with stars; intercept is unstandardized.\n")
    summary_lines.append("Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed).\n\n")
    summary_lines.append("Fit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False) + "\n\n")
    summary_lines.append("Note: The printed paper Table 1 omits SEs; p-values are computed here for star assignment only.\n")
    write_text("./output/table1_summary.txt", "".join(summary_lines))

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
        "n_model_frames": pd.DataFrame(
            {"model": ["Model 1", "Model 2", "Model 3"], "n": [len(frame1), len(frame2), len(frame3)]}
        ),
    }