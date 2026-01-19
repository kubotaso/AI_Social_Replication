def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative "special missing" set (kept intentionally small to avoid over-dropping).
    # Note: 0/7/8/9 and 97/98/99 variants are common in GSS extracts.
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

    def label_with_raw(label, raw):
        return f"{label} [{raw}]"

    def fit_model(df, dv, xcols, model_name, label_map):
        use_cols = [dv] + xcols
        frame = df[use_cols].copy()

        # model-specific listwise deletion ONLY on dv + xcols
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors within this model sample
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

        if len(frame) == 0:
            return meta, pd.DataFrame(), pd.DataFrame(), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows_full = []
        # Constant: unstandardized only (no stars in printed table)
        rows_full.append({
            "term": "Constant",
            "raw": "const",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""
        })

        for c in xcols:
            term_label = label_with_raw(label_map.get(c, c), c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows_full.append({
                    "term": term_label,
                    "raw": c,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows_full.append({
                    "term": term_label,
                    "raw": c,
                    "b": np.nan,
                    "beta": np.nan,
                    "p": np.nan,
                    "sig": ""
                })

        full = pd.DataFrame(rows_full)

        # Table 1 style: constant (b), predictors (standardized beta + stars), omit p-values
        table1_rows = []
        for _, r in full.iterrows():
            if r["raw"] == "const":
                v = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                v = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            table1_rows.append({"term": r["term"], "Table1": v})
        table1 = pd.DataFrame(table1_rows)

        return meta, full, table1, frame

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def freq_table(s):
        return s.value_counts(dropna=False).sort_index()

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
    # - For each genre: 1 if response in {4,5}, 0 if in {1,2,3}, else missing
    # - DV is missing if ANY of the 18 items missing
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
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)  # only valid substantive codes
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum over 18 genre items; indicator=1 if response in {4,5}, 0 if in {1,2,3}; "
        "if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Income per capita: REALINC / HOMPOP (as mapping instruction)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics / group identity
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: build mutually exclusive categories:
    # - White non-Hispanic is reference
    # - black=1 if race==2 (regardless of Hispanic)
    # - otherrace=1 if race==3 (regardless of Hispanic)
    # - hispanic=1 if ethnic indicates Hispanic AND race==1 (so categories are mutually exclusive)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)

    # Hispanic origin from ETHNIC (best effort with available variables)
    hisp = np.nan * np.ones(len(df))
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Common: 1=not Hispanic, 2=Hispanic. Otherwise: treat code 1 as not; other positive codes as Hispanic.
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            hisp = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 99)).astype(float))
    df["hisp_origin"] = hisp

    # Race dummies (non-missing whenever race is non-missing)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: mutually exclusive with black/otherrace by construction
    # If Hispanic origin missing, keep hispanic missing (will contribute to listwise deletion in models using it)
    # If Hispanic origin known:
    #   - if race==1 and Hispanic origin==1 -> hispanic=1
    #   - else hispanic=0
    df["hispanic"] = np.nan
    known_eth = pd.notna(df["hisp_origin"]) & pd.notna(race)
    df.loc[known_eth, "hispanic"] = ((race == 1) & (df["hisp_origin"] == 1.0)).astype(float)
    # If race known but ethnic missing, keep hispanic as NaN (do not force to 0)

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: approximate using RELIG==1 and DENOM in {1 (Baptist), 6 (Other Protestant)}
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0  # avoid extra dropping among Protestants

    # Southern
    region = clean_gss(df.get("region", np.nan))
    # Keep a broad plausible set; treat others as missing
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance: 15 items (5 groups x 3 contexts)
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
        # Keep small integer substantive codes; others missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + key frequencies
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

    # Frequencies for dummies (to detect dummy-trap / empty categories)
    freq_txt = []
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        freq_txt.append(f"\n{v} value counts (incl NA):\n{freq_table(df[v]).to_string()}\n")
    write_text("./output/table1_dummy_frequencies.txt", "".join(freq_txt))

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    label_map = {
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

    meta1, full1, t1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", label_map)
    meta2, full2, t2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", label_map)
    meta3, full3, t3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", label_map)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save human-readable outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    if not full1.empty:
        write_text("./output/table1_model1_full.txt", full1.to_string(index=False) + "\n")
        write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    if not full2.empty:
        write_text("./output/table1_model2_full.txt", full2.to_string(index=False) + "\n")
        write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    if not full3.empty:
        write_text("./output/table1_model3_full.txt", full3.to_string(index=False) + "\n")
        write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combine Table 1-style columns (outer join on term label)
    combined = t1.rename(columns={"Table1": "Model 1 (SES)"}).merge(
        t2.rename(columns={"Table1": "Model 2 (Demographic)"}),
        on="term", how="outer"
    ).merge(
        t3.rename(columns={"Table1": "Model 3 (Political intolerance)"}),
        on="term", how="outer"
    )

    # Order terms in a Table 1-like order
    desired_order = (
        ["Constant"] +
        [label_with_raw(label_map[c], c) for c in m1] +
        [label_with_raw(label_map[c], c) for c in ["female", "age_v", "black", "hispanic", "otherrace",
                                                   "cons_prot", "norelig", "south"]] +
        [label_with_raw(label_map["pol_intol"], "pol_intol")]
    )
    combined["__order"] = combined["term"].map({t: i for i, t in enumerate(desired_order)})
    combined = combined.sort_values(["__order", "term"], na_position="last").drop(columns="__order")

    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Additional: report estimation-sample sizes from frames to confirm listwise behavior
    sample_sizes_txt = (
        f"Model 1 estimation rows: {len(frame1)}\n"
        f"Model 2 estimation rows: {len(frame2)}\n"
        f"Model 3 estimation rows: {len(frame3)}\n"
    )
    write_text("./output/table1_estimation_sample_sizes.txt", sample_sizes_txt)

    # Return results for programmatic use
    return {
        "fit_stats": fit_stats,
        "model1_full": full1,
        "model2_full": full2,
        "model3_full": full3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined": combined,
        "missingness": missingness,
        "estimation_frames": {
            "model1": frame1,
            "model2": frame2,
            "model3": frame3
        }
    }