def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative, common GSS missing codes (numeric). We only apply these where appropriate.
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_numeric(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA_CODES) | set(extra_na)
        return x.where(~x.isin(list(na)), np.nan)

    def clean_gss_likert_1_5(s):
        x = clean_gss_numeric(s)
        return x.where(x.isin([1, 2, 3, 4, 5]), np.nan)

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

    def standardized_betas_posthoc(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name):
        # Model-specific listwise deletion ONLY on dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors in this estimation sample
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

        # If nothing to fit, return shells
        if len(frame) == 0 or len(kept) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": c, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas_posthoc(y, X, res.params)

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        })

        # keep original xcols order; fill NaN for dropped-by-variance predictors
        for c in xcols:
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": c,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": c, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab, labels):
        # Constant: unstandardized b; predictors: standardized beta + stars
        out_vals = []
        out_terms = []
        for _, r in tab.iterrows():
            term = r["term"]
            out_terms.append(labels.get(term, term) if term != "Constant" else "Constant")
            if term == "Constant":
                out_vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out_vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": out_terms, "Table1": out_vals})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read data + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")
    df["year_v"] = clean_gss_numeric(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: number of music genres disliked (0–18)
    # Rule: count of 18 items coded 4/5; DK/NA => missing; if ANY of 18 missing => DV missing
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
        music[c] = clean_gss_likert_1_5(df[c])

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: For each of 18 genres, indicator=1 if response in {4,5}, 0 if in {1,2,3}, missing otherwise.\n"
        "Listwise rule for DV: if any of 18 items missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ"] = clean_gss_numeric(df.get("educ", np.nan))
    df["prestg80"] = clean_gss_numeric(df.get("prestg80", np.nan))

    df["realinc"] = clean_gss_numeric(df.get("realinc", np.nan))
    df["hompop"] = clean_gss_numeric(df.get("hompop", np.nan))
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss_numeric(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male,2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age"] = clean_gss_numeric(df.get("age", np.nan))
    df.loc[df["age"] <= 0, "age"] = np.nan

    # Race + Hispanic: construct mutually exclusive categories
    # Reference category: White, non-Hispanic
    race = clean_gss_numeric(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white,2=black,3=other

    eth = None
    if "ethnic" in df.columns:
        eth = clean_gss_numeric(df["ethnic"])
        # try to interpret: if binary {1,2}, treat 2 as hispanic
        uniq = set(pd.unique(eth.dropna()))
        if uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # best-effort: 1=not hispanic, >=2 indicates Hispanic-origin categories
            hisp = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 99)).astype(float))
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan

    # Mutually exclusive: if Hispanic==1, then Black/Other set to 0 (and White ref not used)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Apply exclusivity where Hispanic known
    hisp_known = df["hispanic"].notna()
    df.loc[hisp_known & (df["hispanic"] == 1.0), "black"] = 0.0
    df.loc[hisp_known & (df["hispanic"] == 1.0), "otherrace"] = 0.0
    # If Hispanic known and ==0, leave race-based dummies as-is.
    # If Hispanic missing, leave as-is (Model 2 will drop those cases only if Hispanic used).

    # Religion
    relig = clean_gss_numeric(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss_numeric(df.get("denom", np.nan))
    # Conservative Protestant proxy (best-effort with available fields):
    # Protestant AND denomination in (Baptist=1, Other Protestant=6). If Protestant but denom missing, treat as 0.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern
    region = clean_gss_numeric(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15 sum), allow partial completion by requiring at least MIN_NONMISS items
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
        x = clean_gss_numeric(df[c])
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_mat[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    # To better match the table's Model 3 N (asked of ~2/3 sample), do NOT require all 15 items.
    # Require at least 12 non-missing (tunable), and rescale to 0..15 by prorating.
    MIN_NONMISS = 12
    nonmiss = tol_mat.notna().sum(axis=1)
    raw_sum = tol_mat.sum(axis=1, skipna=True)
    pol_intol = np.where(nonmiss >= MIN_NONMISS, raw_sum * (15.0 / nonmiss), np.nan)
    df["pol_intol"] = pol_intol

    # ----------------------------
    # Diagnostics: missingness and key distributions
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ", "inc_pc", "prestg80",
        "female", "age", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonmiss_n = int(df[v].notna().sum())
        miss_n = int(df[v].isna().sum())
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss_n,
            "missing": miss_n,
            "pct_missing": (miss_n / (nonmiss_n + miss_n) * 100.0) if (nonmiss_n + miss_n) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    write_text(
        "./output/table1_pol_intol_diagnostics.txt",
        "Political intolerance construction:\n"
        f"- 15 items (5 groups x 3 contexts), intolerant responses coded 1 else 0\n"
        f"- Partial scoring: require at least {MIN_NONMISS} non-missing items; prorate to 0..15 via sum*(15/nonmissing)\n\n"
        "Non-missing item counts (among all rows):\n"
        + nonmiss.describe().to_string()
        + "\n\nPolitical intolerance (prorated) descriptives:\n"
        + pd.Series(df["pol_intol"]).describe().to_string()
        + "\n"
    )

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    # Keep table labels close to paper (avoid adding units/ranges in labels)
    labels = {
        "educ": "Education",
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

    m1 = ["educ", "inc_pc", "prestg80"]
    m2 = m1 + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)")
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)")
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)")

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Build Table 1-style outputs (constant unstd; predictors standardized betas)
    t1_1 = table1_style(tab1.replace({"term": labels}), labels={})
    t1_2 = table1_style(tab2.replace({"term": labels}), labels={})
    t1_3 = table1_style(tab3.replace({"term": labels}), labels={})

    # Human-readable text outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    def full_table_text(meta, tab, labels):
        tmp = tab.copy()
        tmp["term"] = tmp["term"].map(lambda x: "Constant" if x == "Constant" else labels.get(x, x))
        # Round for readability
        tmp2 = tmp.copy()
        for col in ["b", "beta", "p"]:
            tmp2[col] = pd.to_numeric(tmp2[col], errors="coerce")
        tmp2["b"] = tmp2["b"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        tmp2["beta"] = tmp2["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        tmp2["p"] = tmp2["p"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        return (
            f"{meta['model']}\n"
            f"n={meta['n']}, R2={meta['r2']:.6f}, AdjR2={meta['adj_r2']:.6f}"
            + (f", dropped_zero_var={meta['dropped']}" if meta.get("dropped") else "")
            + "\n\n"
            + tmp2[["term", "b", "beta", "p", "sig"]].to_string(index=False)
            + "\n"
        )

    summary_text = []
    summary_text.append("Replication output: Standardized OLS coefficients (beta) reported for predictors; constants are unstandardized.\n")
    summary_text.append("Note: Source Table 1 does not report standard errors; we compute p-values/stars from OLS for internal replication checks.\n\n")
    summary_text.append("FIT STATS\n" + fit_stats.to_string(index=False) + "\n\n")
    summary_text.append(full_table_text(meta1, tab1, labels) + "\n")
    summary_text.append(full_table_text(meta2, tab2, labels) + "\n")
    summary_text.append(full_table_text(meta3, tab3, labels) + "\n")

    write_text("./output/table1_summary.txt", "".join(summary_text))

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Return structured outputs
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1.assign(term=tab1["term"].map(lambda x: "Constant" if x == "Constant" else labels.get(x, x))),
        "model2_full": tab2.assign(term=tab2["term"].map(lambda x: "Constant" if x == "Constant" else labels.get(x, x))),
        "model3_full": tab3.assign(term=tab3["term"].map(lambda x: "Constant" if x == "Constant" else labels.get(x, x))),
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "missingness": missingness,
        "dv_descriptives": df["num_genres_disliked"].describe(),
        "pol_intol_descriptives": pd.Series(df["pol_intol"]).describe(),
    }