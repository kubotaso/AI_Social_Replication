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
        # beta_j = b_j * SD(x_j) / SD(y) computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion on dv + xcols only
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # drop zero-variance predictors for this model/sample
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

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
        # Constant: unstandardized b. Predictors: standardized beta + stars.
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "value": out})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def combine_table_union(model_tabs, model_names):
        # UNION of all terms across models (fixes previous intersection bug)
        all_terms = []
        for tab in model_tabs:
            all_terms.extend(list(tab["term"].values))
        # preserve order of first appearance
        seen = set()
        terms = []
        for t in all_terms:
            if t not in seen:
                seen.add(t)
                terms.append(t)

        combined = pd.DataFrame({"term": terms})
        for tab, name in zip(model_tabs, model_names):
            t = tab[["term", "value"]].copy()
            combined = combined.merge(t, on="term", how="left", suffixes=("", ""))
            combined = combined.rename(columns={"value": name})
        combined = combined.fillna("")
        return combined

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
    # 1 if response in {4,5}, 0 if in {1,2,3}, missing otherwise.
    # DV missing if ANY of 18 items missing.
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
        "Construction: sum of 18 genre items coded 1 if response 4/5, 0 if 1/2/3; DK/NA -> missing; if any genre missing -> DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Income per capita: use REALINC/HOMPOP, but rescale to thousands to stabilize numerics
    # (Standardized beta is scale-invariant; this helps constants/OLS numerics)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = (df["realinc_v"] / df["hompop_v"]) / 1000.0
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race and ethnicity: build mutually exclusive categories with White non-Hispanic as reference
    # Use ETHNIC as Hispanic indicator where possible; otherwise set to missing.
    race = clean_gss(df.get("race", np.nan)).where(clean_gss(df.get("race", np.nan)).isin([1, 2, 3]), np.nan)

    eth = clean_gss(df.get("ethnic", np.nan)) if "ethnic" in df.columns else pd.Series(np.nan, index=df.index)
    # Best-effort: treat 2 as Hispanic when binary {1,2}; otherwise treat any code >=2 as Hispanic-origin.
    if eth.notna().any():
        uniq = set(pd.unique(eth.dropna()))
        if uniq.issubset({1.0, 2.0}):
            hisp = (eth == 2)
        else:
            hisp = (eth >= 2) & (eth <= 99)
        df["hispanic"] = np.where(eth.isna(), np.nan, hisp.astype(float))
    else:
        df["hispanic"] = np.nan

    # Mutually exclusive: if Hispanic==1 => Hispanic category regardless of race
    # Else use RACE categories.
    df["black"] = np.nan
    df["otherrace"] = np.nan

    known_hisp = df["hispanic"].notna()
    known_race = race.notna()

    # initialize to missing; then fill 0/1 where definable
    # For non-Hispanic known (hispanic==0) and known race, define black/otherrace
    mask_nonh = (df["hispanic"] == 0) & known_race
    df.loc[mask_nonh, "black"] = (race.loc[mask_nonh] == 2).astype(float)
    df.loc[mask_nonh, "otherrace"] = (race.loc[mask_nonh] == 3).astype(float)

    # For Hispanic==1, force black/otherrace to 0 (so Hispanic is its own mutually exclusive group)
    mask_h = (df["hispanic"] == 1)
    df.loc[mask_h, "black"] = 0.0
    df.loc[mask_h, "otherrace"] = 0.0

    # For cases where Hispanic missing but race known, keep race-based dummies but Hispanic will be missing
    mask_hisp_missing_race_known = df["hispanic"].isna() & known_race
    df.loc[mask_hisp_missing_race_known, "black"] = (race.loc[mask_hisp_missing_race_known] == 2).astype(float)
    df.loc[mask_hisp_missing_race_known, "otherrace"] = (race.loc[mask_hisp_missing_race_known] == 3).astype(float)

    # Religion
    relig = clean_gss(df.get("relig", np.nan)).where(clean_gss(df.get("relig", np.nan)).isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant approximation (documentation-supported): Protestant + specific denom codes.
    # Keep conservative definition narrow to avoid overclassification.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # Baptist, Other Protestant
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0  # don't drop Protestants due to denom missing

    # Region: South indicator (REGION==3 per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15), strict sum; missing if any item missing (matches mapping instruction)
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
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness
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

    # quick frequency checks for key dummies
    def freq(series):
        s = series.copy()
        return pd.Series({
            "n": int(s.shape[0]),
            "nonmissing": int(s.notna().sum()),
            "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
            "sum": float(s.sum(skipna=True)) if s.notna().any() else np.nan,
            "n_unique": int(s.nunique(dropna=True))
        })

    checks = pd.DataFrame({
        "female": freq(df["female"]),
        "black": freq(df["black"]),
        "hispanic": freq(df["hispanic"]),
        "otherrace": freq(df["otherrace"]),
        "south": freq(df["south"]),
    }).T
    write_text("./output/table1_dummy_checks.txt", checks.to_string() + "\n")

    # ----------------------------
    # Models
    # ----------------------------
    labels = {
        "educ_yrs": "Education (years)",
        "inc_pc": "Household income per capita (thousands, per capita)",
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

    # Save full regression tables (b, beta, p, stars)
    def tab_to_text(tab):
        t = tab.copy()
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        return t[["term", "b", "beta", "p", "sig"]]

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/table1_model1_full.txt", tab_to_text(tab1).to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", tab_to_text(tab2).to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", tab_to_text(tab3).to_string(index=False) + "\n")

    # Save Table-1-style (beta + stars; constant unstandardized)
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)
    write_text("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined table using UNION of terms across models (fix)
    combined = combine_table_union(
        [t1, t2, t3],
        ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    )
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Return key objects
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "combined_table1style": combined,
        "missingness": missingness
    }