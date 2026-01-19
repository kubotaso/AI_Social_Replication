def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    GSS_NA_CODES = {7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_gss_numeric(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA_CODES) | set(extra_na)
        x = x.where(~x.isin(list(na)), np.nan)
        return x

    def safe_sd(s):
        s = pd.to_numeric(s, errors="coerce")
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return np.nan
        return float(sd)

    def p_to_stars(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def make_beta_table(res, y, X_no_const, label_map):
        # Standardized beta for each predictor: beta = b * sd(x)/sd(y)
        sd_y = safe_sd(y)
        rows = []

        # Constant: report unstandardized b (no stars in Table 1 presentation)
        b0 = float(res.params.get("const", np.nan))
        rows.append({"term": "Constant", "b": b0, "beta": np.nan, "p": np.nan, "sig": ""})

        for col in X_no_const.columns:
            b = float(res.params.get(col, np.nan))
            p = float(res.pvalues.get(col, np.nan))
            sd_x = safe_sd(X_no_const[col])
            beta = np.nan
            if pd.notna(b) and pd.notna(sd_x) and pd.notna(sd_y):
                beta = b * (sd_x / sd_y)
            rows.append(
                {
                    "term": label_map.get(col, col),
                    "b": b,
                    "beta": beta,
                    "p": p,
                    "sig": p_to_stars(p),
                }
            )
        return pd.DataFrame(rows)

    def format_table1_like(tab):
        # Display column: Constant as b (3 decimals), predictors as beta+stars (3 decimals)
        out = tab.copy()
        disp = []
        for _, r in out.iterrows():
            if r["term"] == "Constant":
                disp.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                disp.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        out["Table1"] = disp
        return out[["term", "Table1"]]

    def fit_model(df, dv, x_cols, model_name, label_map):
        needed = [dv] + x_cols
        d = df[needed].copy()

        # Model-specific listwise deletion
        d = d.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors (avoid singularities)
        kept = []
        dropped = []
        for c in x_cols:
            nun = d[c].nunique(dropna=True)
            if nun <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            return None, {
                "model": model_name,
                "n": 0,
                "r2": np.nan,
                "adj_r2": np.nan,
                "dropped": dropped,
            }, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        tab = make_beta_table(res, y, X, label_map)
        fit = {
            "model": model_name,
            "n": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "dropped": dropped,
        }
        return tab, fit, d

    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[clean_gss_numeric(df["year"]) == 1993].copy()

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
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: clean_gss_numeric(df[c]) for c in music_items})

    # valid responses are 1..5; everything else -> missing
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked_ind = music.isin([4, 5]).astype(float)
    disliked_ind = disliked_ind.where(music.notna(), np.nan)

    # listwise across 18 items: if any missing, DV missing
    df["num_genres_disliked"] = disliked_ind.sum(axis=1, min_count=len(music_items))
    df.loc[disliked_ind.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv = df["num_genres_disliked"]
    dv_desc = dv.describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Constructed as count of 18 genre items coded 4/5 = dislike/dislike very much.\n"
        "DK/NA/refused/etc. treated as missing; DV requires complete data on all 18 items.\n\n"
        f"N non-missing DV: {int(dv.notna().sum())}\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss_numeric(df.get("educ", np.nan), extra_na=[0])
    df["prestg80_v"] = clean_gss_numeric(df.get("prestg80", np.nan), extra_na=[0])

    # income per capita = realinc / hompop
    df["realinc_v"] = clean_gss_numeric(df.get("realinc", np.nan), extra_na=[0])
    df["hompop_v"] = clean_gss_numeric(df.get("hompop", np.nan), extra_na=[0])
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    df.loc[df["inc_pc"] <= 0, "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss_numeric(df.get("sex", np.nan))
    # GSS: 1=male, 2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss_numeric(df.get("age", np.nan), extra_na=[0])

    # Race dummies from RACE (1 white, 2 black, 3 other)
    race = clean_gss_numeric(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: this dataset has ETHNIC; treat "Hispanic" as ETHNIC==1 (common GSS coding).
    # Crucially: do NOT set to all-missing; keep non-Hispanic as 0 where ETHNIC is valid.
    if "ethnic" in df.columns:
        eth = clean_gss_numeric(df["ethnic"])
        # Keep only plausible codes (positive integers); treat others missing
        eth = eth.where(eth > 0, np.nan)
        # In many GSS extracts, ETHNIC==1 indicates "Hispanic". Use that.
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))
    else:
        df["hispanic"] = np.nan

    # Religion: No religion dummy from RELIG (4 = none)
    relig = clean_gss_numeric(df.get("relig", np.nan), extra_na=[0])
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: Use RELIG==1 (Protestant) AND DENOM in a conservative set.
    # GSS DENOM codes vary by extract; to avoid collapsing N, code it deterministically:
    # - If Protestant and denom is observed, mark conservative when denom indicates Baptist (often 1)
    #   or "other Protestant" (often 6). Otherwise 0. Missing only when relig/denom missing.
    denom = clean_gss_numeric(df.get("denom", np.nan), extra_na=[0])
    denom = denom.where(denom > 0, np.nan)
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where((relig.notna()) & (denom.notna()), (is_prot & denom_cons).astype(float), np.nan)

    # Southern: REGION==3
    region = clean_gss_numeric(df.get("region", np.nan), extra_na=[0])
    region = region.where(region > 0, np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0–15): sum of 15 intolerant indicators.
    # Listwise across the 15 items (consistent with the rest of this replication's strictness).
    tol_specs = {
        # Anti-religionist
        "spkath": ("eq", 2),
        "colath": ("eq", 5),
        "libath": ("eq", 1),
        # Racist
        "spkrac": ("eq", 2),
        "colrac": ("eq", 5),
        "librac": ("eq", 1),
        # Communist
        "spkcom": ("eq", 2),
        "colcom": ("eq", 4),
        "libcom": ("eq", 1),
        # Military-rule advocate
        "spkmil": ("eq", 2),
        "colmil": ("eq", 5),
        "libmil": ("eq", 1),
        # Homosexual
        "spkhomo": ("eq", 2),
        "colhomo": ("eq", 5),
        "libhomo": ("eq", 1),
    }
    missing_tol = [k for k in tol_specs.keys() if k not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance items: {missing_tol}")

    tol = pd.DataFrame({k: clean_gss_numeric(df[k]) for k in tol_specs.keys()})
    # keep only positive codes; others -> missing
    tol = tol.where(tol > 0, np.nan)

    tol_ind = pd.DataFrame(index=df.index)
    for k, (_, target) in tol_specs.items():
        tol_ind[k] = np.where(tol[k].isna(), np.nan, (tol[k] == target).astype(float))

    df["pol_intol"] = tol_ind.sum(axis=1, min_count=len(tol_specs))
    df.loc[tol_ind.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + variation
    # ----------------------------
    diag_vars = [
        "num_genres_disliked",
        "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v",
        "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol",
    ]
    diag = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        s = df[v]
        nonmiss = int(s.notna().sum())
        miss = int(s.isna().sum())
        pct = float(miss / len(df) * 100) if len(df) else np.nan
        nun = int(s.nunique(dropna=True))
        diag.append({"variable": v, "nonmissing": nonmiss, "missing": miss, "pct_missing": pct, "n_unique": nun})
    diag_df = pd.DataFrame(diag).sort_values(["pct_missing", "variable"])
    write_text("./output/table1_missingness_diagnostics.txt", diag_df.to_string(index=False) + "\n")

    # ----------------------------
    # Table 1 models
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
    dv_name = "num_genres_disliked"

    m1_x = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2_x = m1_x + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    tab1, fit1, frame1 = fit_model(df, dv_name, m1_x, "Model 1 (SES)", labels)
    tab2, fit2, frame2 = fit_model(df, dv_name, m2_x, "Model 2 (Demographic)", labels)
    tab3, fit3, frame3 = fit_model(df, dv_name, m3_x, "Model 3 (Political intolerance)", labels)

    # Fit stats dataframe
    fit_stats = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2", "dropped"]]
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Write model tables (Table 1 style) + brief notes
    def write_model_file(path, model_name, tab, fit, x_cols, frame):
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"N (complete cases for this model): {fit['n']}")
        lines.append(f"R^2: {fit['r2'] if pd.notna(fit['r2']) else np.nan}")
        lines.append(f"Adj R^2: {fit['adj_r2'] if pd.notna(fit['adj_r2']) else np.nan}")
        if fit.get("dropped"):
            lines.append("")
            lines.append("Dropped predictors due to zero variance after listwise deletion:")
            lines.append(", ".join(fit["dropped"]) if fit["dropped"] else "(none)")
        lines.append("")
        lines.append("Coefficients (Table 1 style):")
        lines.append("- Constant is unstandardized intercept (b)")
        lines.append("- Predictors are standardized betas (β) with stars from two-tailed OLS p-values")
        lines.append("- Stars: * p<.05, ** p<.01, *** p<.001")
        lines.append("")
        if tab is None or fit["n"] == 0:
            lines.append("MODEL COULD NOT BE ESTIMATED (no complete cases).")
        else:
            tshow = format_table1_like(tab)
            lines.append(tshow.to_string(index=False))

            # Add quick variable distribution check on estimation sample
            lines.append("")
            lines.append("Estimation-sample quick checks (mean for dummies; mean/sd for continuous):")
            chk_rows = []
            for v in x_cols:
                if v not in frame.columns:
                    continue
                s = frame[v]
                mean = float(s.mean()) if len(s) else np.nan
                sd = float(s.std(ddof=0)) if len(s) else np.nan
                chk_rows.append({"var": v, "mean": mean, "sd": sd, "n_unique": int(s.nunique(dropna=True))})
            chk = pd.DataFrame(chk_rows)
            if len(chk):
                lines.append(chk.to_string(index=False))
        write_text(path, "\n".join(lines) + "\n")

    write_model_file("./output/table1_model1_ses.txt", "Model 1 (SES)", tab1, fit1, m1_x, frame1)
    write_model_file("./output/table1_model2_demographic.txt", "Model 2 (Demographic)", tab2, fit2, m2_x, frame2)
    write_model_file("./output/table1_model3_political_intolerance.txt", "Model 3 (Political intolerance)", tab3, fit3, m3_x, frame3)

    # Return results as a dict of DataFrames
    out = {
        "fit_stats": fit_stats,
        "model1": tab1 if tab1 is not None else pd.DataFrame(),
        "model2": tab2 if tab2 is not None else pd.DataFrame(),
        "model3": tab3 if tab3 is not None else pd.DataFrame(),
        "missingness": diag_df,
    }
    return out