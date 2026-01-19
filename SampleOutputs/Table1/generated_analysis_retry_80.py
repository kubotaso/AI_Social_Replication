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

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j)/SD(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_ols_table1(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion ONLY on dv + xcols
        use = df[[dv] + xcols].dropna(axis=0, how="any").copy()

        # Ensure all predictors have variation
        kept, dropped = [], []
        for c in xcols:
            nun = use[c].nunique(dropna=True)
            if nun <= 1:
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

        # If empty, return shell
        if len(use) == 0 or len(kept) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return meta, tab, use

        y = use[dv].astype(float)
        X = use[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas(y, X, res.params)

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # never star the constant
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
        return meta, tab, use

    def table1_style(tab):
        # Constant: unstandardized b; Predictors: standardized beta + stars
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

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
    # Rule: count of 18 items where response in {4,5}
    # Missing: if ANY of 18 items missing -> DV missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    miss_music = [c for c in music_items if c not in df.columns]
    if miss_music:
        raise ValueError(f"Missing required music columns: {miss_music}")

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
    # Demographics / identities
    # ----------------------------
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use 'ethnic' if present.
    # Critical fixes:
    #  - Do NOT invert coding.
    #  - Do NOT coerce missing to 0.
    #  - Use a conservative rule: code 1 as not-Hispanic, 2 as Hispanic when those codes appear.
    #  - Otherwise: if there are many categories, treat 1 as not-Hispanic and 2..(max substantive) as Hispanic
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Keep only positive codes; anything else already NaN
        eth = eth.where(eth > 0, np.nan)

        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best-effort: if code 1 exists, treat it as "not Hispanic";
            # treat codes >=2 as Hispanic only if they look like substantive categories (<= 10 or <= 20).
            # This avoids accidentally labeling strange large codes as Hispanic.
            max_code = float(np.nanmax(eth.values)) if eth.notna().any() else np.nan
            if pd.isna(max_code):
                df["hispanic"] = np.nan
            else:
                upper = 20.0 if max_code <= 20 else 10.0 if max_code <= 10 else 20.0
                df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= upper)).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (approximation): Protestant (RELIG==1) and denomination in {1 Baptist, 6 Other}
    # Keep non-Protestants as 0; keep Protestants with missing denom as 0 to avoid needless case loss.
    is_prot = relig.eq(1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0
    df.loc[(~is_prot) & relig.notna(), "cons_prot"] = 0.0

    # South: mapping instruction specifies REGION==3
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance scale (0–15)
    # Key fix: code item-level missing correctly and only treat well-defined response codes as valid.
    # ----------------------------
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    miss_tol = [c for c, _ in tol_items if c not in df.columns]
    if miss_tol:
        raise ValueError(f"Missing required political tolerance columns: {miss_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Allow common GSS codes for these items:
        # speech: 1/2; college: 4/5; library: 1/2 typically; but keep 1..6 as "plausible"
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

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
    # Diagnostics: distributions and missingness
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

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

    # Quick frequencies to catch dummy miscoding (esp. Hispanic / Other race)
    freq_txt = []
    for v in ["black", "hispanic", "otherrace", "female", "south", "norelig", "cons_prot"]:
        s = df[v]
        freq_txt.append(f"\n{v} (nonmissing={int(s.notna().sum())}):\n{s.value_counts(dropna=False).to_string()}\n")
    write_text("./output/table1_dummy_frequencies.txt", "\n".join(freq_txt))

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, use1 = fit_ols_table1(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, use2 = fit_ols_table1(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, use3 = fit_ols_table1(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Save full regression tables (for debugging) and Table1-style tables (paper-facing)
    tab1_style = table1_style(tab1)
    tab2_style = table1_style(tab2)
    tab3_style = table1_style(tab3)

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", tab1_style.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", tab2_style.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", tab3_style.to_string(index=False) + "\n")

    # Create a compact combined Table 1-like panel
    all_terms = pd.Index(tab1_style["term"]).union(tab2_style["term"]).union(tab3_style["term"])
    combined = pd.DataFrame({"term": all_terms})

    combined = combined.merge(tab1_style.rename(columns={"Table1": "Model 1"}), on="term", how="left")
    combined = combined.merge(tab2_style.rename(columns={"Table1": "Model 2"}), on="term", how="left")
    combined = combined.merge(tab3_style.rename(columns={"Table1": "Model 3"}), on="term", how="left")

    # Add fit stats rows at bottom
    fit_rows = pd.DataFrame({
        "term": ["N", "R²", "Adj. R²", "Dropped predictors"],
        "Model 1": [str(meta1["n"]), f"{meta1['r2']:.3f}" if pd.notna(meta1["r2"]) else "", f"{meta1['adj_r2']:.3f}" if pd.notna(meta1["adj_r2"]) else "", meta1["dropped"]],
        "Model 2": [str(meta2["n"]), f"{meta2['r2']:.3f}" if pd.notna(meta2["r2"]) else "", f"{meta2['adj_r2']:.3f}" if pd.notna(meta2["adj_r2"]) else "", meta2["dropped"]],
        "Model 3": [str(meta3["n"]), f"{meta3['r2']:.3f}" if pd.notna(meta3["r2"]) else "", f"{meta3['adj_r2']:.3f}" if pd.notna(meta3["adj_r2"]) else "", meta3["dropped"]],
    })
    combined_out = pd.concat([combined, pd.DataFrame({"term": ["---"], "Model 1": [""], "Model 2": [""], "Model 3": [""]}), fit_rows], ignore_index=True)

    write_text("./output/table1_combined.txt", combined_out.to_string(index=False) + "\n")

    # Also provide model-specific sample summaries to diagnose unexpected N drops
    def sample_summary(use_df, name):
        lines = [f"{name} sample size: {len(use_df)}"]
        for v in use_df.columns:
            s = use_df[v]
            if v == "num_genres_disliked":
                lines.append(f"{v}: mean={s.mean():.3f}, sd={s.std(ddof=1):.3f}, min={s.min():.3f}, max={s.max():.3f}")
            else:
                # show mean/sd for numeric; dummies too
                lines.append(f"{v}: mean={s.mean():.3f}, sd={s.std(ddof=1):.3f}, nonmissing={int(s.notna().sum())}")
        return "\n".join(lines) + "\n"

    write_text("./output/model1_sample_summary.txt", sample_summary(use1, "Model 1"))
    write_text("./output/model2_sample_summary.txt", sample_summary(use2, "Model 2"))
    write_text("./output/model3_sample_summary.txt", sample_summary(use3, "Model 3"))

    # Return results
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": tab1_style,
        "model2_table1style": tab2_style,
        "model3_table1style": tab3_style,
        "table1_combined": combined_out,
        "missingness": missingness
    }