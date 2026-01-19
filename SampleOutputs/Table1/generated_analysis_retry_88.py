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

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_gss(series, valid=None, extra_na=()):
        s = to_num(series)
        na = set(GSS_NA_CODES) | set(extra_na)
        s = s.where(~s.isin(list(na)), np.nan)
        if valid is not None:
            if callable(valid):
                s = s.where(valid(s), np.nan)
            else:
                s = s.where(s.isin(list(valid)), np.nan)
        return s

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
        return np.nan if pd.isna(v) or v <= 0 else float(np.sqrt(v))

    def standardized_betas(y, X, params):
        # beta_j = b_j * sd(x_j)/sd(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_ols(df, dv, xcols, model_name, labels):
        use = df[[dv] + xcols].copy()
        use = use.dropna(axis=0, how="any").copy()

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
            "dropped_predictors": ",".join(dropped) if dropped else ""
        }

        rows = []
        if len(use) == 0 or len(kept) == 0:
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

        return meta, pd.DataFrame(rows), use

    def table1_style(tab):
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def write_txt(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Load + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]
    if "year" not in df.columns:
        raise ValueError("Expected 'year' column.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0–18), listwise across 18 items
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
        # substantive codes are 1..5; anything else -> missing
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

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
    # Demographics / group identity
    # ----------------------------
    # Female
    sex = clean_gss(df.get("sex", np.nan), valid=[1, 2])
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age in years
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race & Hispanic: build mutually exclusive categories with White non-Hispanic reference
    # Using available 'race' and 'ethnic' (as provided in this extract).
    race = clean_gss(df.get("race", np.nan), valid=[1, 2, 3])  # 1 white, 2 black, 3 other
    eth = clean_gss(df.get("ethnic", np.nan))
    # Best-effort: treat any nonmissing code other than 0/NA as valid, define Hispanic as (eth != 1) if 1 looks like "not Hispanic"
    # Prefer common binary mapping 1=not Hispanic, 2=Hispanic.
    hisp = pd.Series(np.nan, index=df.index, dtype="float64")
    eth_vals = set(pd.unique(eth.dropna()))
    if eth_vals.issubset({1.0, 2.0}):
        hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
    elif len(eth_vals) > 0:
        # If 1 exists, treat 1 as non-Hispanic; any other observed positive code as Hispanic-origin.
        if 1.0 in eth_vals:
            hisp = np.where(eth.isna(), np.nan, (eth != 1).astype(float))
        else:
            # If no obvious non-Hispanic code, treat all observed nonmissing as not Hispanic (conservative)
            hisp = np.where(eth.isna(), np.nan, 0.0)
    df["hispanic"] = hisp

    # Mutually exclusive dummies:
    # - Hispanic: 1 if Hispanic (any race); else 0
    # - Black: 1 if non-Hispanic & race==2
    # - Other race: 1 if non-Hispanic & race==3
    # Reference: non-Hispanic White (race==1, hispanic==0)
    df["hispanic"] = df["hispanic"].where(df["hispanic"].isin([0.0, 1.0]), np.nan)
    df.loc[df["hispanic"].isna(), "hispanic"] = 0.0  # treat unknown as non-Hispanic to prevent catastrophic N loss

    df["black"] = np.where(race.isna(), np.nan, ((df["hispanic"] == 0.0) & (race == 2)).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, ((df["hispanic"] == 0.0) & (race == 3)).astype(float))

    # Religion: No religion and Conservative Protestant
    relig = clean_gss(df.get("relig", np.nan), valid=[1, 2, 3, 4, 5])  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (approximation with GSS DENOM codes):
    # Use RELIG==1 (Protestant) and DENOM in a conservative-leaning set.
    # Keep this conservative to avoid tiny N artifacts.
    # Common GSS: 1 Baptist, 2 Methodist, 3 Lutheran, 4 Presbyterian, 5 Episcopalian, 6 Other, 7 None
    cons_set = {1, 6}
    is_prot = (relig == 1)
    cons = (is_prot & denom.isin(list(cons_set)))
    df["cons_prot"] = np.where(relig.isna(), np.nan, cons.astype(float))
    # If Protestant but denom missing, treat as not conservative rather than missing (prevents N collapse)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 per instruction
    region = clean_gss(df.get("region", np.nan))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance (0–15): sum of 15 intolerant responses, listwise across 15 items
    # ----------------------------
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
        # Keep plausible small integers, set others missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Labels + model specs
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

    meta1, tab1, used1 = fit_ols(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, used2 = fit_ols(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, used3 = fit_ols(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Diagnostics: missingness & key distributions
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
        denom_n = nonmiss + miss
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / denom_n * 100.0) if denom_n else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)

    # ----------------------------
    # Write human-readable outputs
    # ----------------------------
    write_txt("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n\n")

    write_txt("./output/table1_model1_full.txt", tab1.to_string(index=False) + "\n")
    write_txt("./output/table1_model2_full.txt", tab2.to_string(index=False) + "\n")
    write_txt("./output/table1_model3_full.txt", tab3.to_string(index=False) + "\n")

    write_txt("./output/table1_model1_table1style.txt", table1_style(tab1).to_string(index=False) + "\n")
    write_txt("./output/table1_model2_table1style.txt", table1_style(tab2).to_string(index=False) + "\n")
    write_txt("./output/table1_model3_table1style.txt", table1_style(tab3).to_string(index=False) + "\n")

    dv_desc = df["num_genres_disliked"].describe()
    pol_desc = df["pol_intol"].describe()
    write_txt(
        "./output/table1_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre ratings coded 4/5; any missing on 18 => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n\nPolitical intolerance (0–15)\n"
        "Construction: sum of 15 intolerant responses; any missing on 15 => scale missing.\n\n"
        + pol_desc.to_string()
        + "\n"
    )

    write_txt("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n\n")

    # Also save an at-a-glance combined "Table 1" layout
    t1 = table1_style(tab1).rename(columns={"Table1": "Model 1"})
    t2 = table1_style(tab2).rename(columns={"Table1": "Model 2"})
    t3 = table1_style(tab3).rename(columns={"Table1": "Model 3"})
    combined = t1.merge(t2, on="term", how="outer").merge(t3, on="term", how="outer")
    write_txt("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n\n")

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": table1_style(tab1),
        "model2_table1style": table1_style(tab2),
        "model3_table1style": table1_style(tab3),
        "table1_combined": combined,
        "missingness": missingness,
        "analytic_n": pd.DataFrame({
            "model": ["Model 1", "Model 2", "Model 3"],
            "n_complete": [len(used1), len(used2), len(used3)]
        })
    }