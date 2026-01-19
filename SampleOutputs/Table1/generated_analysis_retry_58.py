def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS-style special codes (best-effort; dataset may not use all of them)
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
        # beta_j = b_j * SD(x_j) / SD(y), computed on the estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on the variables in this model
        needed = [dv] + list(xcols)
        frame = df.loc[:, needed].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
        }

        rows = []
        if len(frame) == 0:
            # Empty shell
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[list(xcols)].astype(float)

        # Drop any zero-variance predictors in THIS estimation sample to avoid runtime errors
        kept = []
        dropped = []
        for c in xcols:
            if X[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            meta["dropped"] = ",".join(dropped)
            return meta, pd.DataFrame(rows), frame

        Xk = X[kept]
        Xc = sm.add_constant(Xk, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, Xk, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)
        meta["dropped"] = ",".join(dropped) if dropped else ""

        # Constant: unstandardized only
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # never star constant
        })

        # Predictors: include b/beta/p internally (we'll format Table-1 style separately)
        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p),
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
        # Constant: unstandardized b; others: standardized beta + stars
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

    def df_to_text(df_, title=None):
        s = []
        if title:
            s.append(title)
            s.append("-" * len(title))
        s.append(df_.to_string(index=False))
        s.append("")
        return "\n".join(s)

    # ----------------------------
    # Read + Year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: Musical exclusiveness = count of genres disliked (4/5) across 18 items
    # Rule: if ANY of 18 items missing => DV missing (listwise within DV construction)
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
        "Construction: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
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

    # Race/ethnicity (mutually exclusive categories; white non-hisp reference implied by dummies)
    # Use RACE for black/other; use ETHNIC for Hispanic (best-effort from provided vars).
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other
    eth = clean_gss(df.get("ethnic", np.nan)) if "ethnic" in df.columns else pd.Series(np.nan, index=df.index)

    # Hispanic coding (best-effort):
    # - If binary {1,2}: 2 = Hispanic
    # - Else: treat >=2 as Hispanic-origin category (keeps 1 as non-Hispanic)
    hisp = pd.Series(np.nan, index=df.index, dtype="float64")
    if "ethnic" in df.columns:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            hisp = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 99)).astype(float))
    df["hispanic"] = pd.to_numeric(hisp, errors="coerce")

    # Make race dummies mutually exclusive with Hispanic taking precedence (common practice)
    # If Hispanic==1: set black=0 and otherrace=0 (so race dummies represent non-Hispanic race)
    black = np.where(race.isna(), np.nan, (race == 2).astype(float))
    otherr = np.where(race.isna(), np.nan, (race == 3).astype(float))

    df["black"] = black
    df["otherrace"] = otherr

    # Apply Hispanic precedence only where hispanic is observed (not missing)
    idx_hisp1 = (df["hispanic"] == 1)
    df.loc[idx_hisp1, "black"] = 0.0
    df.loc[idx_hisp1, "otherrace"] = 0.0

    # If race is missing but Hispanic is observed: treat race dummies as 0 (so we don't lose cases)
    # This avoids unnecessary attrition from an ethnicity-only report.
    idx_race_missing_hisp_obs = df["hispanic"].notna() & race.isna()
    df.loc[idx_race_missing_hisp_obs, "black"] = 0.0
    df.loc[idx_race_missing_hisp_obs, "otherrace"] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # best-effort proxy (Baptist or other Protestant)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, treat as not conservative Protestant (keep cases)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South
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
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + quick distributions
    # ----------------------------
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
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Frequency checks for key dummies (helps debug "dropped" issues)
    freq_text = []
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        if v in df.columns:
            vc = df[v].value_counts(dropna=False).sort_index()
            freq_text.append(f"{v} value_counts (including NA):\n{vc.to_string()}\n")
    write_text("./output/table1_dummy_frequencies.txt", "\n".join(freq_text))

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

    # Table-1 style outputs
    t1_1 = table1_style(tab1)
    t1_2 = table1_style(tab2)
    t1_3 = table1_style(tab3)

    # Combined "Table 1" view (outer join on terms so all predictors show up)
    combined = (
        t1_1.rename(columns={"Table1": "Model 1 (SES)"})
        .merge(t1_2.rename(columns={"Table1": "Model 2 (Demographic)"}), on="term", how="outer")
        .merge(t1_3.rename(columns={"Table1": "Model 3 (Political intolerance)"}), on="term", how="outer")
    )

    # Order terms: constant first, then in the order variables appear across models
    desired_order = (
        ["Constant"]
        + [labels[c] for c in m1]
        + [labels[c] for c in m2 if c not in m1]
        + [labels[c] for c in m3 if c not in m2]
    )
    combined["__ord"] = combined["term"].map({t: i for i, t in enumerate(desired_order)})
    combined = combined.sort_values(["__ord", "term"], na_position="last").drop(columns="__ord")

    # ----------------------------
    # Write outputs
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    summary_parts = []
    summary_parts.append(df_to_text(fit_stats, "Fit statistics"))
    summary_parts.append(df_to_text(combined, "Table 1 style (standardized betas; unstandardized constant)"))
    write_text("./output/table1_summary.txt", "\n".join(summary_parts))

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
        "n_model_frames": pd.DataFrame([
            {"model": "Model 1 (SES)", "n_frame": len(frame1)},
            {"model": "Model 2 (Demographic)", "n_frame": len(frame2)},
            {"model": "Model 3 (Political intolerance)", "n_frame": len(frame3)},
        ])
    }