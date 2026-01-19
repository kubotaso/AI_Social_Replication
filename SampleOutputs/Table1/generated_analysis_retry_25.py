def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def gss_na_to_nan(series):
        """
        Conservative missing handling:
        - Coerce to numeric
        - Set common GSS missing codes to NaN
        Notes:
        - We do NOT blanket-drop all 0 values (some variables legitimately take 0).
        - Item-specific validity checks happen later.
        """
        s = to_num(series)
        na_codes = {
            7, 8, 9,  # common DK/NA/refused (esp. older GSS)
            97, 98, 99,
            997, 998, 999,
            9997, 9998, 9999
        }
        return s.where(~s.isin(list(na_codes)), np.nan)

    def star(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def sd_sample(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas_from_unstd(y, X, params):
        """
        beta_j = b_j * sd(x_j) / sd(y) using estimation-sample SDs.
        X excludes constant.
        """
        sdy = sd_sample(y)
        betas = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sd_sample(X[c])
            if pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (sdx / sdy)
        return betas

    def fit_ols_table1(df, dv, xcols, label_map, model_name):
        """
        Fit OLS on raw variables (with intercept), compute standardized betas for slopes,
        keep unstandardized intercept. Listwise deletion only on dv + included xcols.
        """
        needed = [dv] + xcols
        frame = df[needed].copy()
        n_before = len(frame)
        frame = frame.dropna(axis=0, how="any").copy()
        n = len(frame)

        # If empty, return shells
        rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
        for c in xcols:
            rows.append({"term": label_map.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
        tab = pd.DataFrame(rows)

        fit_stats = {
            "model": model_name,
            "n_before": int(n_before),
            "n": int(n),
            "r2": np.nan,
            "adj_r2": np.nan
        }

        if n == 0:
            return tab, fit_stats, None

        # Drop predictors with zero variance after listwise deletion
        kept = []
        for c in xcols:
            if frame[c].nunique(dropna=True) > 1:
                kept.append(c)

        if len(kept) == 0:
            return tab, fit_stats, None

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(y, X, res.params)

        out_rows = []
        out_rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant (Table 1 style)
        })

        for c in xcols:
            term = label_map.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                out_rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": star(p)
                })
            else:
                out_rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(out_rows)

        fit_stats["r2"] = float(res.rsquared)
        fit_stats["adj_r2"] = float(res.rsquared_adj)
        return tab, fit_stats, frame

    def to_table1_display(tab):
        """
        Display like Table 1:
        - Constant: unstandardized b (no stars)
        - Predictors: standardized beta with stars
        """
        disp = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                disp.append("" if pd.isna(r["b"]) else f"{r['b']:.3f}")
            else:
                disp.append("" if pd.isna(r["beta"]) else f"{r['beta']:.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"], "Table1": disp})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Load + restrict year
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column.")
    df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # Dependent variable: musical exclusiveness = count of 18 genres disliked
    # Disliked: response 4 or 5 on each genre (responses 1..5 valid)
    # DK/NA treated as missing; listwise over 18 items for DV construction.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    miss_music = [c for c in music_items if c not in df.columns]
    if miss_music:
        raise ValueError(f"Missing required music items: {miss_music}")

    music = pd.DataFrame({c: gss_na_to_nan(df[c]) for c in music_items})
    # Only 1..5 are valid; anything else becomes missing
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv = df["num_genres_disliked"]
    dv_desc = pd.Series({
        "n_nonmissing": int(dv.notna().sum()),
        "mean": float(dv.mean(skipna=True)),
        "sd": float(dv.std(ddof=1, skipna=True)),
        "min": float(dv.min(skipna=True)) if dv.notna().any() else np.nan,
        "max": float(dv.max(skipna=True)) if dv.notna().any() else np.nan
    })
    write_text("./output/table1_dv_descriptives.txt", dv_desc.to_string() + "\n")

    # ----------------------------
    # Predictors (coding per mapping; keep simple & faithful)
    # ----------------------------
    # Education (years)
    df["educ_yrs"] = gss_na_to_nan(df["educ"]) if "educ" in df.columns else np.nan
    # EDUC: set clearly invalid nonpositive to missing
    df.loc[df["educ_yrs"].notna() & (df["educ_yrs"] <= 0), "educ_yrs"] = np.nan

    # Occupational prestige
    df["prestg80_v"] = gss_na_to_nan(df["prestg80"]) if "prestg80" in df.columns else np.nan
    df.loc[df["prestg80_v"].notna() & (df["prestg80_v"] <= 0), "prestg80_v"] = np.nan

    # Income per capita = REALINC / HOMPOP (no log; paper reports standardized betas anyway)
    df["realinc_v"] = gss_na_to_nan(df["realinc"]) if "realinc" in df.columns else np.nan
    df["hompop_v"] = gss_na_to_nan(df["hompop"]) if "hompop" in df.columns else np.nan
    df.loc[df["hompop_v"].notna() & (df["hompop_v"] <= 0), "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    # Do NOT drop inc_pc==0 mechanically; just treat negative as missing
    df.loc[df["inc_pc"].notna() & (df["inc_pc"] < 0), "inc_pc"] = np.nan

    # Female dummy
    if "sex" in df.columns:
        sex = gss_na_to_nan(df["sex"])
        # 1=male, 2=female
        df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))
    else:
        df["female"] = np.nan

    # Age (years; keep top-coded 89 as 89)
    df["age_v"] = gss_na_to_nan(df["age"]) if "age" in df.columns else np.nan
    df.loc[df["age_v"].notna() & (df["age_v"] <= 0), "age_v"] = np.nan

    # Race dummies (RACE: 1=white, 2=black, 3=other)
    if "race" in df.columns:
        race = gss_na_to_nan(df["race"]).where(lambda s: s.isin([1, 2, 3]), np.nan)
        df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
        df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))
    else:
        df["black"] = np.nan
        df["otherrace"] = np.nan

    # Hispanic dummy: use 'ethnic' as provided in this dataset.
    # In this extract it appears to be a Hispanic-origin style code with many missings.
    # Use a robust rule:
    # - If ETHNIC is binary {1,2}: treat 2 as Hispanic.
    # - Else, treat codes in 1..9 as Hispanic categories (common recode) and others as non-Hispanic,
    #   leaving GSS missing codes as NaN.
    if "ethnic" in df.columns:
        eth = gss_na_to_nan(df["ethnic"])
        uniq = sorted(pd.unique(eth.dropna()))
        uniq_set = set([float(u) for u in uniq if np.isfinite(u)])
        if len(uniq_set) > 0 and uniq_set.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # define hispanic among observed non-missing values; keep missing as NaN
            df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 1) & (eth <= 9)).astype(float))
    else:
        df["hispanic"] = np.nan

    # Religion: no religion dummy (RELIG: 4=none)
    if "relig" in df.columns:
        relig = gss_na_to_nan(df["relig"])
        df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))
    else:
        relig = pd.Series(np.nan, index=df.index)
        df["norelig"] = np.nan

    # Conservative Protestant: derived from RELIG and DENOM
    # Keep non-missing when RELIG is known; if denom missing but protestant, set 0 (unknown denom treated as not conservative).
    if "denom" in df.columns:
        denom = gss_na_to_nan(df["denom"])
    else:
        denom = pd.Series(np.nan, index=df.index)

    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # 1=Baptist; 6=Other Protestant (common coding)
    cons = is_prot & denom_cons
    df["cons_prot"] = np.where(relig.isna(), np.nan, cons.astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern dummy (REGION: 3=south)
    if "region" in df.columns:
        region = gss_na_to_nan(df["region"])
        df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))
    else:
        df["south"] = np.nan

    # Political intolerance (0-15) from 15 items; require all 15 present for index
    tol_map = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    miss_tol = [v for v, _ in tol_map if v not in df.columns]
    if miss_tol:
        raise ValueError(f"Missing required political tolerance items: {miss_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for v, intolerant_codes in tol_map:
        x = gss_na_to_nan(df[v])
        # Keep only plausible substantive codes for these items; otherwise missing.
        # SPK*: typically 1/2; COL*: 4/5 or 1/2 depending; LIB*: 1/2.
        # We accept 1..9 and treat others as missing to be safe.
        x = x.where((x >= 0) & (x <= 9), np.nan)
        tol_df[v] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan
    # sanity: enforce range 0..15
    df.loc[df["pol_intol"].notna() & ((df["pol_intol"] < 0) | (df["pol_intol"] > 15)), "pol_intol"] = np.nan

    # ----------------------------
    # Missingness report (key vars)
    # ----------------------------
    key_vars = [
        "num_genres_disliked",
        "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol"
    ]
    miss_rows = []
    for v in key_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss = int(df[v].isna().sum())
        miss_rows.append({"variable": v, "nonmissing": nonmiss, "missing": miss, "pct_missing": miss / max(1, (nonmiss + miss))})
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

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
        "pol_intol": "Political intolerance (0–15)"
    }

    dv_name = "num_genres_disliked"

    m1_x = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2_x = m1_x + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    m1_tab, m1_fit, m1_frame = fit_ols_table1(df, dv_name, m1_x, label_map, "Model 1 (SES)")
    m2_tab, m2_fit, m2_frame = fit_ols_table1(df, dv_name, m2_x, label_map, "Model 2 (Demographic)")
    m3_tab, m3_fit, m3_frame = fit_ols_table1(df, dv_name, m3_x, label_map, "Model 3 (Political intolerance)")

    fit_stats = pd.DataFrame([m1_fit, m2_fit, m3_fit])[["model", "n", "r2", "adj_r2"]]

    # Save "paper style" tables
    m1_disp = to_table1_display(m1_tab)
    m2_disp = to_table1_display(m2_tab)
    m3_disp = to_table1_display(m3_tab)

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/table1_model1.txt", m1_disp.to_string(index=False) + "\n")
    write_text("./output/table1_model2.txt", m2_disp.to_string(index=False) + "\n")
    write_text("./output/table1_model3.txt", m3_disp.to_string(index=False) + "\n")

    # Save fuller diagnostics (b, beta, p)
    write_text("./output/table1_model1_full.txt", m1_tab.to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", m2_tab.to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", m3_tab.to_string(index=False) + "\n")

    # Combined summary file
    summary_lines = []
    summary_lines.append("Table 1 replication from GSS 1993 (computed from microdata)\n")
    summary_lines.append("DV: Number of music genres disliked (0–18), count of 18 genre items rated 4/5; listwise across 18 items.\n")
    summary_lines.append("Reported coefficients: standardized betas (β) for predictors; unstandardized intercept.\n")
    summary_lines.append("Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed). SEs not shown (to match paper style).\n\n")
    summary_lines.append("FIT STATS\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nMODEL 1 (SES)\n" + m1_disp.to_string(index=False))
    summary_lines.append("\n\nMODEL 2 (Demographic)\n" + m2_disp.to_string(index=False))
    summary_lines.append("\n\nMODEL 3 (Political intolerance)\n" + m3_disp.to_string(index=False))
    write_text("./output/table1_summary.txt", "\n".join(summary_lines) + "\n")

    return {
        "fit_stats": fit_stats,
        "model1_table1": m1_disp,
        "model2_table1": m2_disp,
        "model3_table1": m3_disp,
        "model1_full": m1_tab,
        "model2_full": m2_tab,
        "model3_full": m3_tab,
        "missingness": missingness,
        "dv_descriptives": dv_desc
    }