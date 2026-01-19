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

    def zscore(series):
        s = pd.to_numeric(series, errors="coerce")
        m = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - m) / sd

    def safe_dummy_from_codes(x, true_codes, valid_codes=None):
        """
        x: numeric Series with NA already cleaned.
        true_codes: set of codes defining 1.
        valid_codes: optional set/range defining acceptable substantive values; values outside => NA.
        """
        x = pd.to_numeric(x, errors="coerce")
        if valid_codes is not None:
            x = x.where(x.isin(list(valid_codes)), np.nan)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.notna()] = 0.0
        out.loc[x.isin(list(true_codes))] = 1.0
        return out

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_ols_with_betas(df, dv, xcols, model_name, labels):
        """
        Paper-style:
          - OLS on unstandardized variables (for constant, R2, etc.)
          - Standardized coefficients computed by running OLS on standardized y and standardized X (no intercept).
            (This is numerically equal to b * sd(x)/sd(y) when computed on the same estimation sample.)
          - Stars from p-values of unstandardized OLS (typical table convention).
        """
        needed = [dv] + xcols
        frame = df[needed].copy().dropna(axis=0, how="any")

        # drop zero-variance predictors within this estimation sample (prevents singular fit)
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
            rows.append({"term": "Constant", "const_b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "const_b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return meta, tab, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)

        # Unstandardized OLS (for constant, p-values, R2)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        # Standardized betas computed via standardized regression with no intercept
        y_z = zscore(y)
        betas = {}
        for c in kept:
            betas[c] = np.nan
        Xz = pd.DataFrame({c: zscore(X[c]) for c in kept})
        # Drop any columns that become all-NA after zscoring (shouldn't happen if variance > 0, but guard)
        Xz = Xz.dropna(axis=1, how="all")
        common = y_z.notna()
        if len(Xz.columns) > 0:
            common = common & Xz.notna().all(axis=1)
            if common.sum() > 0:
                res_z = sm.OLS(y_z.loc[common], Xz.loc[common]).fit()
                for c in Xz.columns:
                    betas[c] = float(res_z.params.get(c, np.nan))

        # Build table
        rows.append({
            "term": "Constant",
            "const_b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""
        })

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "const_b": np.nan,
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": term, "const_b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_style(tab):
        # Constant: unstandardized; predictors: beta with stars; omit p-values in display
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["const_b"]) else f"{float(r['const_b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    # ----------------------------
    # Load + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0–18)
    # Rule: count items coded 4/5, DK/NA missing; if any of 18 missing => DV missing
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

    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: 18 genre ratings; disliked=1 if response in {4,5}, else 0 if in {1,2,3}; "
        "DK/NA treated as missing; if any genre item missing => DV missing (listwise on 18 items).\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Per-capita income: REALINC / HOMPOP
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = safe_dummy_from_codes(sex, true_codes={2}, valid_codes={1, 2})

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=White,2=Black,3=Other
    df["black"] = safe_dummy_from_codes(race, true_codes={2}, valid_codes={1, 2, 3})
    df["otherrace"] = safe_dummy_from_codes(race, true_codes={3}, valid_codes={1, 2, 3})

    # Hispanic: use ETHNIC. In this extract, values like 29,21,97 appear; treat 97/98/99 as NA already.
    # Use conservative rule: hispanic=1 if ETHNIC==1, else 0 for other non-missing.
    # (This avoids the prior sign-flip from treating ">=2" as Hispanic.)
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # keep plausible codes 1..99 (after NA cleaning); treat others as missing
        eth = eth.where((eth >= 1) & (eth <= 99), np.nan)
        df["hispanic"] = safe_dummy_from_codes(eth, true_codes={1})

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = safe_dummy_from_codes(relig, true_codes={4}, valid_codes={1, 2, 3, 4, 5})

    denom = clean_gss(df.get("denom", np.nan))
    # Best-effort conservative Protestant definition given limited extract:
    # Protestant (RELIG==1) and DENOM in {1 Baptist, 6 Other Protestant}.
    denom = denom.where(denom.isin([0, 1, 2, 3, 4, 5, 6, 7]), np.nan)  # allow typical denom codes if present
    cons = pd.Series(np.nan, index=df.index, dtype="float64")
    cons.loc[relig.notna()] = 0.0
    cons.loc[(relig == 1) & (denom.isin([1, 6]))] = 1.0
    # If Protestant but denom missing, keep as 0 (avoid unnecessary case loss)
    cons.loc[(relig == 1) & (denom.isna())] = 0.0
    df["cons_prot"] = cons

    # Southern (per mapping instruction): REGION==3
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = safe_dummy_from_codes(region, true_codes={3})

    # Political intolerance scale (0–15)
    # Build item indicators; allow partial completion (sum across non-missing items),
    # but require at least MIN_NONMISS items (to reduce artificial missingness while avoiding tiny partials).
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
        # keep plausible small codes; anything else missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    nonmiss_counts = tol_df.notna().sum(axis=1)
    df["pol_intol"] = tol_df.sum(axis=1, min_count=1)

    # Require sufficient answered items to be comparable; choose 12/15 to align with "asked battery" and limit noise.
    MIN_NONMISS = 12
    df.loc[nonmiss_counts < MIN_NONMISS, "pol_intol"] = np.nan

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
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Additional checks for dummies
    def freq_text(series, name):
        s = series.copy()
        return (
            f"{name} (non-missing n={int(s.notna().sum())}):\n"
            + s.value_counts(dropna=False).sort_index().to_string()
            + "\n\n"
        )

    dummy_check = ""
    for nm in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        if nm in df.columns:
            dummy_check += freq_text(df[nm], nm)
    dummy_check += "Political intolerance answered-items count (non-missing):\n"
    dummy_check += nonmiss_counts.describe().to_string() + "\n"
    write_text("./output/table1_dummy_and_polintol_checks.txt", dummy_check)

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

    meta1, tab1, frame1 = fit_ols_with_betas(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_ols_with_betas(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_ols_with_betas(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Paper-style tables
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    # Save outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combined Table 1 style
    # Align rows by term
    merged = t1.merge(t2, on="term", how="outer", suffixes=("_m1", "_m2"))
    merged = merged.merge(t3, on="term", how="outer")
    merged = merged.rename(columns={"Table1_m1": "Model 1", "Table1_m2": "Model 2", "Table1": "Model 3"})
    # Ensure desired term order
    term_order = ["Constant"] + [labels[c] for c in m3]
    merged["__order"] = merged["term"].map({t: i for i, t in enumerate(term_order)}).fillna(9999)
    merged = merged.sort_values("__order").drop(columns="__order")
    write_text("./output/table1_combined_table1style.txt", merged.to_string(index=False) + "\n")

    # Return results
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1,
        "model2_table1style": t2,
        "model3_table1style": t3,
        "table1_combined": merged,
        "missingness": missingness,
        "model_frames": {
            "model1": frame1,
            "model2": frame2,
            "model3": frame3,
        }
    }