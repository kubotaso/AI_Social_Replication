def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative "special missing" set; also treat any negative values as missing.
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA_CODES) | set(extra_na)
        x = x.where(~x.isin(list(na)), np.nan)
        x = x.where(~(x < 0), np.nan)
        return x

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

    def fit_model(df, dv, xcols, model_name, labels, weight_col=None):
        frame = df[[dv] + xcols + ([weight_col] if weight_col else [])].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors
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
        if len(frame) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        if weight_col:
            w = frame[weight_col].astype(float)
            # ensure positive weights
            w = w.where(w > 0, np.nan)
            ok = w.notna()
            y, Xc, X = y.loc[ok], Xc.loc[ok], X.loc[ok]
            w = w.loc[ok]
            res = sm.WLS(y, Xc, weights=w).fit()
        else:
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
        # Constant: unstandardized b; predictors: standardized beta + stars.
        vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": vals})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

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
    # Dependent variable: # genres disliked (0–18)
    # - Use 18 items
    # - disliked = 1 if 4/5, 0 if 1/2/3
    # - If any of 18 missing => DV missing
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

    # Income per capita: REALINC / HOMPOP
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

    # Race dummies (reference = White)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy from ETHNIC, but avoid massive attrition:
    # - If ETHNIC is missing, set to 0 (assume not Hispanic) rather than NA.
    #   This is a pragmatic replication choice to avoid collapsing N when ETHNIC wasn't asked.
    df["hispanic"] = 0.0
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If ETHNIC is 1/2 (not hispanic / hispanic): use it.
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth == 2, 1.0, 0.0)
        else:
            # Best-effort: treat code==1 as not Hispanic, other positive substantive codes as Hispanic-origin.
            df["hispanic"] = np.where((eth.notna()) & (eth >= 2) & (eth <= 99), 1.0, 0.0)

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)

    # Conservative Protestant: approximation with RELIG==1 and DENOM in {1,6}
    # Missing denom among Protestants -> treat as 0 rather than NA (to limit attrition).
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin([1, 6])).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South dummy: REGION==3 per mapping instruction
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15):
    # - Sum 15 intolerant indicators
    # - To avoid excessive attrition, allow partial completion:
    #   require at least MIN_ANSWERED items answered; then scale is raw sum of answered intolerant (0..answered)
    #   and rescale to 0..15 by multiplying by (15/answered).
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

    tol_mat = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Keep plausible small integers only
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_mat[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol_mat.notna().sum(axis=1).astype(float)
    intolerant_sum = tol_mat.sum(axis=1, skipna=True).astype(float)

    MIN_ANSWERED = 12  # pragmatic to reduce attrition while staying close to "battery asked" logic
    pol_intol = np.where(answered >= MIN_ANSWERED, intolerant_sum * (15.0 / answered), np.nan)
    df["pol_intol"] = pol_intol

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
        "pol_intol": "Political intolerance",
    }

    # ----------------------------
    # Models: model-wise listwise deletion only
    # ----------------------------
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, fr1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, fr2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, fr3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Output files (human-readable)
    # ----------------------------
    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: For each of 18 music items, disliked=1 if response is 4/5, else 0 if 1/2/3; "
        "DK/NA treated as missing; if any of 18 items missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # Political intolerance descriptives
    pol_desc = df["pol_intol"].describe()
    write_text(
        "./output/table1_pol_intol_descriptives.txt",
        "Political intolerance scale (rescaled to 0–15)\n"
        "Construction: 15 items (5 groups x 3 liberties) coded intolerant=1 for specified responses.\n"
        f"Missing rule: require at least {MIN_ANSWERED}/15 answered; rescale sum by (15/answered).\n\n"
        + pol_desc.to_string()
        + "\n"
    )

    # Missingness summary for key variables
    diag_vars = ["num_genres_disliked"] + sorted(set(m3))
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

    # Model outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Full tables include b, beta, p, sig (for debugging), and Table-1 style
    for name, tab in [("model1", tab1), ("model2", tab2), ("model3", tab3)]:
        write_text(f"./output/table1_{name}_full.txt", tab.to_string(index=False) + "\n")
        t1 = table1_style(tab)
        write_text(f"./output/table1_{name}_table1style.txt", t1.to_string(index=False) + "\n")

    # Consolidated "Table 1 style" panel
    t1_1 = table1_style(tab1).rename(columns={"Table1": "Model 1"})
    t1_2 = table1_style(tab2).rename(columns={"Table1": "Model 2"}).drop(columns=["term"])
    t1_3 = table1_style(tab3).rename(columns={"Table1": "Model 3"}).drop(columns=["term"])
    table1_panel = pd.concat([t1_1, t1_2, t1_3], axis=1)
    write_text("./output/table1_panel.txt", table1_panel.to_string(index=False) + "\n")

    # Also return structured outputs
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": table1_style(tab1),
        "model2_table1style": table1_style(tab2),
        "model3_table1style": table1_style(tab3),
        "table1_panel": table1_panel,
        "missingness": missingness,
    }