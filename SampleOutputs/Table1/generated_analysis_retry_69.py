def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS missing-value codes (best-effort). Do NOT treat 1/2/3/4/5 etc as missing.
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
            "dropped_predictors": ",".join(dropped) if dropped else ""
        }

        # Empty shell if insufficient
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

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # never star constant
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

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_style(tab):
        """
        Table 1 style:
          - Constant is unstandardized b
          - Predictors are standardized beta (β) + stars
        """
        out = []
        for _, r in tab.iterrows():
            if str(r["term"]) == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].astype(str), "Table1": out})

    def merge_table_panels(model_tables, model_names):
        # Full outer join on 'term' to avoid term loss / NaN term rows
        panel = None
        for tab, name in zip(model_tables, model_names):
            t = tab.copy()
            t["term"] = t["term"].astype(str)
            t = t.rename(columns={"Table1": name})
            t = t[["term", name]]
            if panel is None:
                panel = t
            else:
                panel = panel.merge(t, on="term", how="outer")
        return panel

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fmt_fit_stats(fit_df):
        cols = ["model", "n", "r2", "adj_r2", "dropped_predictors"]
        d = fit_df[cols].copy()
        d["r2"] = d["r2"].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
        d["adj_r2"] = d["adj_r2"].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
        return d

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
    # - 18 items; disliked=1 if 4 or 5; like/neutral=0 if 1..3
    # - any DK/NA on any of the 18 items => DV missing (listwise for DV construction)
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
        "Construction: count across 18 genre items where response in {4,5}; {1,2,3}=0; DK/NA missing.\n"
        "Rule: if ANY of 18 items missing => DV missing.\n\n"
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
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male,2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race + ethnicity (mutually exclusive categories, reference = non-Hispanic white)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white, 2=black, 3=other

    eth = None
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Common GSS coding: 1=not hispanic, 2=hispanic
        # If not binary, best-effort: 1=not hispanic; >=2 indicates some Hispanic-origin category.
        if set(pd.unique(eth.dropna())).issubset({1.0, 2.0}):
            hisp_flag = (eth == 2)
        else:
            hisp_flag = (eth >= 2) & (eth <= 99)
        hisp_flag = hisp_flag.where(eth.notna(), np.nan)
    else:
        hisp_flag = pd.Series(np.nan, index=df.index)

    # Mutually exclusive assignment:
    # - Hispanic = 1 if Hispanic flag true (any race)
    # - Else Black = 1 if race==2
    # - Else Other race = 1 if race==3
    # White (non-Hispanic) is implied reference: all dummies 0
    df["hispanic"] = np.where(hisp_flag.isna(), np.nan, hisp_flag.astype(float))

    black_me = (df["hispanic"] == 0) & (race == 2)
    other_me = (df["hispanic"] == 0) & (race == 3)
    # If hispanic is missing, keep race dummies missing to avoid forcing sample loss patterns unpredictably
    df["black"] = np.where(df["hispanic"].isna() | race.isna(), np.nan, black_me.astype(float))
    df["otherrace"] = np.where(df["hispanic"].isna() | race.isna(), np.nan, other_me.astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant: best-effort per provided fields.
    # Mark conservative as Protestant (RELIG==1) with DENOM in {1=Baptist, 6=Other}.
    # If Protestant but denom missing: set to 0 so denom missing doesn't drop respondents.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15)
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
        # Keep plausible response range; else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Missingness report (raw + model-specific)
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

    # Save full regression tables
    def full_table_text(tab, title):
        t = tab.copy()
        t["term"] = t["term"].astype(str)
        # pretty formatting
        t["b"] = t["b"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
        t["beta"] = t["beta"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
        t["p"] = t["p"].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        cols = ["term", "b", "beta", "p", "sig"]
        return (
            f"{title}\n"
            "Note: b = unstandardized OLS coefficient; beta = standardized coefficient (β). Constant is b.\n\n"
            + t[cols].to_string(index=False)
            + "\n"
        )

    write_text("./output/model1_full.txt", full_table_text(tab1, "Model 1 (SES)"))
    write_text("./output/model2_full.txt", full_table_text(tab2, "Model 2 (Demographic)"))
    write_text("./output/model3_full.txt", full_table_text(tab3, "Model 3 (Political intolerance)"))

    # Table-1-style panels (Constant=b; Predictors=β)
    t1s = table1_style(tab1)
    t2s = table1_style(tab2)
    t3s = table1_style(tab3)

    write_text(
        "./output/table1_style_notes.txt",
        "Table 1 style output:\n"
        "- Constant is unstandardized b\n"
        "- Predictors are standardized coefficients (β) with stars from two-tailed p-values\n"
        "- Stars: * p<.05, ** p<.01, *** p<.001\n"
    )

    write_text("./output/model1_table1style.txt", t1s.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2s.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3s.to_string(index=False) + "\n")

    # Combined panel (fix: keyed full join on term; never lose variable names)
    panel = merge_table_panels(
        [t1s, t2s, t3s],
        ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    )

    # Order rows in a sensible Table-1 order
    desired_order = [
        "Constant",
        "Education (years)",
        "Household income per capita",
        "Occupational prestige",
        "Female",
        "Age",
        "Black",
        "Hispanic",
        "Other race",
        "Conservative Protestant",
        "No religion",
        "Southern",
        "Political intolerance (0–15)",
    ]
    panel["term"] = panel["term"].astype(str)
    panel["__ord"] = panel["term"].map({k: i for i, k in enumerate(desired_order)})
    panel = panel.sort_values(["__ord", "term"], na_position="last").drop(columns="__ord")

    write_text("./output/table1_panel.txt", panel.to_string(index=False) + "\n")

    # Fit stats text
    write_text("./output/table1_fit_stats.txt", fmt_fit_stats(fit_stats).to_string(index=False) + "\n")

    # Model-specific case counts and quick DV mean/SD in each estimation sample
    def sample_summ(frame, name):
        y = frame["num_genres_disliked"].astype(float)
        return {
            "model": name,
            "n": int(len(frame)),
            "dv_mean": "" if len(frame) == 0 else f"{float(y.mean()):.3f}",
            "dv_sd": "" if len(frame) == 0 else f"{float(y.std(ddof=1)):.3f}",
        }

    sample_info = pd.DataFrame([
        sample_summ(frame1, "Model 1 (SES)"),
        sample_summ(frame2, "Model 2 (Demographic)"),
        sample_summ(frame3, "Model 3 (Political intolerance)"),
    ])
    write_text("./output/table1_model_samples.txt", sample_info.to_string(index=False) + "\n")

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "table1_panel": panel,
        "missingness": missingness,
        "model_samples": sample_info,
    }