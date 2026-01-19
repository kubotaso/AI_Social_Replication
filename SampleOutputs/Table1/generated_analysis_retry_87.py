def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Broad GSS-style missing codes; only applied where appropriate.
    GSS_NA = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_numeric(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA) | set(extra_na)
        return x.where(~x.isin(list(na)), np.nan)

    def clean_gss_discrete(s, valid_values, extra_na=()):
        x = clean_gss_numeric(s, extra_na=extra_na)
        return x.where(x.isin(list(valid_values)), np.nan)

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
        # beta_j = b_j * SD(x_j) / SD(y) computed on the estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, labels, model_name):
        # Strict listwise deletion on dv + xcols (Table 1 approach)
        frame = df[[dv] + xcols].copy().dropna(axis=0, how="any")
        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        rows = []
        rows.append(
            {
                "term": "Constant",
                "b": float(res.params.get("const", np.nan)),
                "beta": np.nan,
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": "",  # no stars on intercept
            }
        )
        for c in xcols:
            p = float(res.pvalues.get(c, np.nan))
            rows.append(
                {
                    "term": labels.get(c, c),
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p),
                }
            )

        tab = pd.DataFrame(rows)
        meta = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(len(frame)),
                    "r2": float(res.rsquared),
                    "adj_r2": float(res.rsquared_adj),
                }
            ]
        )
        return meta, tab, frame

    def table1_style(tab):
        # Constant shown as b; predictors shown as standardized beta with stars
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
    # Read + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected 'year' column in input CSV.")

    df["year_v"] = clean_gss_numeric(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: number of music genres disliked (0–18)
    # - Each item: 1..5 substantive, count 4/5 as disliked
    # - Treat DK/NA as missing; DV missing if ANY of 18 missing (strict listwise for DV construction)
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    miss_music = [c for c in music_items if c not in df.columns]
    if miss_music:
        raise ValueError(f"Missing required music genre columns: {miss_music}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        music[c] = clean_gss_discrete(df[c], valid_values=[1, 2, 3, 4, 5])

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss_numeric(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss_numeric(df.get("prestg80", np.nan))

    # Income per capita: REALINC / HOMPOP (strict missing if either missing; hompop>0)
    df["realinc_v"] = clean_gss_numeric(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss_numeric(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss_discrete(df.get("sex", np.nan), valid_values=[1, 2])
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss_numeric(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss_discrete(df.get("race", np.nan), valid_values=[1, 2, 3])  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy (use ETHNIC if present). IMPORTANT: do NOT induce missing for non-Hispanic.
    # Conservative rule: if ETHNIC is missing -> missing; else hispanic = 1 only when code==1 (common in GSS: 1=Hispanic)
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss_numeric(df["ethnic"])
        # Try to infer coding:
        # - If codes include {1,2} only: assume 1=Hispanic, 2=Not Hispanic (common in some extracts)
        # - Else if codes include 1 and others: treat 1 as Hispanic, otherwise not Hispanic.
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))
        else:
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))

    # Religion
    relig = clean_gss_discrete(df.get("relig", np.nan), valid_values=[1, 2, 3, 4, 5])  # 4=none
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss_numeric(df.get("denom", np.nan))
    # Keep missing denom as missing (do NOT force 0), to match typical listwise deletion in published tables.
    # Approx "conservative Protestant": Protestant + (Baptist or Other Protestant).
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna() | denom.isna(), np.nan, (is_prot & denom_cons).astype(float))

    # Southern dummy (per mapping instruction)
    region = clean_gss_numeric(df.get("region", np.nan))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance 0–15:
    # Item-level coding per mapping; treat non-substantive as missing
    tol_items = [
        ("spkath", 2, [1, 2]),
        ("colath", 5, [4, 5]),
        ("libath", 1, [1, 2]),
        ("spkrac", 2, [1, 2]),
        ("colrac", 5, [4, 5]),
        ("librac", 1, [1, 2]),
        ("spkcom", 2, [1, 2]),
        ("colcom", 4, [3, 4]),
        ("libcom", 1, [1, 2]),
        ("spkmil", 2, [1, 2]),
        ("colmil", 5, [4, 5]),
        ("libmil", 1, [1, 2]),
        ("spkhomo", 2, [1, 2]),
        ("colhomo", 5, [4, 5]),
        ("libhomo", 1, [1, 2]),
    ]
    miss_tol = [c for c, _, _ in tol_items if c not in df.columns]
    if miss_tol:
        raise ValueError(f"Missing required political tolerance columns: {miss_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_code, valid_vals in tol_items:
        x = clean_gss_discrete(df[c], valid_values=valid_vals)
        tol_df[c] = np.where(x.isna(), np.nan, (x == intolerant_code).astype(float))

    # Scale rule: require ALL 15 items nonmissing (strict; matches table-style listwise for model 3 if paper did that)
    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Descriptives and missingness
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_txt(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genres rated 4/5; DK/NA -> missing; if any of 18 items missing -> DV missing.\n\n"
        + dv_desc.to_string()
        + "\n",
    )

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
        miss_rows.append(
            {
                "variable": v,
                "nonmissing": nonmiss,
                "missing": miss,
                "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan,
            }
        )
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_txt("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

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

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, labels, "Model 1 (SES)")
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, labels, "Model 2 (Demographic)")
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, labels, "Model 3 (Political intolerance)")

    fit_stats = pd.concat([meta1, meta2, meta3], axis=0, ignore_index=True)

    # Save full regression tables (debuggable)
    write_txt("./output/table1_model1_full.txt", tab1.to_string(index=False) + "\n")
    write_txt("./output/table1_model2_full.txt", tab2.to_string(index=False) + "\n")
    write_txt("./output/table1_model3_full.txt", tab3.to_string(index=False) + "\n")

    # Save Table 1-style outputs (only constant and standardized betas with stars)
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    write_txt("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_txt("./output/table1_model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_txt("./output/table1_model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_txt("./output/table1_model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined Table 1-style view
    combined = (
        t1.rename(columns={"Table1": "Model 1"})
        .merge(t2.rename(columns={"Table1": "Model 2"}), on="term", how="outer")
        .merge(t3.rename(columns={"Table1": "Model 3"}), on="term", how="outer")
    )
    # Order rows roughly like the paper
    order = [
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
    combined["__ord"] = combined["term"].map({k: i for i, k in enumerate(order)})
    combined = combined.sort_values(["__ord", "term"], na_position="last").drop(columns="__ord")
    write_txt("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Human-readable summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from provided data)\n")
    summary_lines.append("DV: number of music genres disliked (0–18)\n")
    summary_lines.append("\nFit statistics:\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nModel 1 (Table 1 style):\n")
    summary_lines.append(t1.to_string(index=False))
    summary_lines.append("\n\nModel 2 (Table 1 style):\n")
    summary_lines.append(t2.to_string(index=False))
    summary_lines.append("\n\nModel 3 (Table 1 style):\n")
    summary_lines.append(t3.to_string(index=False))
    summary_lines.append("\n")
    write_txt("./output/table1_summary.txt", "\n".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined": combined,
        "missingness": missingness,
        "analytic_n": pd.DataFrame(
            [
                {"model": "Model 1 (SES)", "n": len(frame1)},
                {"model": "Model 2 (Demographic)", "n": len(frame2)},
                {"model": "Model 3 (Political intolerance)", "n": len(frame3)},
            ]
        ),
    }