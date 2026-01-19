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

    def star(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def zscore_series(s):
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def coerce_valid_range(s, valid_min=None, valid_max=None, valid_set=None):
        s = to_num(s)
        if valid_set is not None:
            return s.where(s.isin(valid_set), np.nan)
        if valid_min is not None:
            s = s.where(s >= valid_min, np.nan)
        if valid_max is not None:
            s = s.where(s <= valid_max, np.nan)
        return s

    def make_dummy_from_codes(s, one_code, valid_codes):
        s = coerce_valid_range(s, valid_set=valid_codes)
        out = pd.Series(np.where(s.isna(), np.nan, (s == one_code).astype(float)), index=s.index)
        return out

    def standardized_betas_from_unstd(y, X, res_unstd):
        # beta_j = b_j * SD(Xj) / SD(Y), computed on the estimation sample
        y_sd = y.std(ddof=0)
        betas = {}
        for c in X.columns:
            x_sd = X[c].std(ddof=0)
            b = res_unstd.params.get(c, np.nan)
            if pd.isna(b) or pd.isna(x_sd) or pd.isna(y_sd) or x_sd == 0 or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(b) * float(x_sd) / float(y_sd)
        return betas

    def fit_table1_ols(df, dv, x_cols, model_label):
        # Build model frame (listwise on model variables only)
        need = [dv] + x_cols
        d = df[need].copy()
        for c in need:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum().sort_index()
        d = d.dropna(axis=0, how="any").copy()

        # Drop any constant predictors after listwise deletion
        kept, dropped = [], []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            tab = pd.DataFrame([{"term": "Constant", "beta": np.nan, "p": np.nan, "sig": ""}] +
                               [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols])
            fit = {"model": model_label, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            return tab, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(y, X, res)

        rows = []
        # Constant unstandardized
        rows.append({
            "term": "Constant",
            "beta": float(res.params.get("const", np.nan)),
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": star(res.pvalues.get("const", np.nan)),
        })
        # Predictors: standardized betas; p-values from unstandardized model (same t-tests for slopes)
        for c in x_cols:
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({"term": c, "beta": float(betas.get(c, np.nan)), "p": p, "sig": star(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        fit = {"model": model_label, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return tab, fit, nonmissing_before, dropped, d

    def write_model_output(model_label, tab, fit, nonmissing_before, dropped, path_txt):
        out = tab.copy()
        out["beta"] = pd.to_numeric(out["beta"], errors="coerce").round(6)
        out["p"] = pd.to_numeric(out["p"], errors="coerce").round(6)

        with open(path_txt, "w", encoding="utf-8") as f:
            f.write(f"{model_label}\n")
            f.write("=" * len(model_label) + "\n\n")
            f.write("Non-missing counts BEFORE listwise deletion (model variables):\n")
            f.write(nonmissing_before.to_string())
            f.write("\n\n")
            if dropped:
                f.write("Dropped predictors due to zero variance AFTER listwise deletion:\n")
                f.write(", ".join(dropped) + "\n\n")
            f.write("Fit statistics:\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.6f}\n")
            f.write(f"Adj R^2 = {fit['adj_r2']:.6f}\n\n")
            f.write("Table 1-style coefficients:\n")
            f.write("- Constant is unstandardized (raw DV units)\n")
            f.write("- Predictors are standardized coefficients (β = b * SD(X)/SD(Y))\n")
            f.write("- Stars from two-tailed p-values: * <.05, ** <.01, *** <.001\n\n")
            f.write(out.to_string(index=False))
            f.write("\n")

    # ----------------------------
    # Read data and restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: Musical exclusiveness (# genres disliked), listwise across 18 items
    # 1..5 valid; disliked if 4 or 5; anything else -> missing; if any missing across 18 -> DV missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = df[music_items].apply(to_num)
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["music_exclusive"] = disliked.sum(axis=1, min_count=len(music_items))
    df.loc[disliked.isna().any(axis=1), "music_exclusive"] = np.nan

    dv_desc = df["music_exclusive"].describe()
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Musical exclusiveness (# of music genres disliked)\n")
        f.write("Count of 18 genres with response 4 ('dislike') or 5 ('dislike very much').\n")
        f.write("Any missing/non-1..5 on any of the 18 items => DV missing.\n\n")
        f.write(dv_desc.to_string())
        f.write("\n")

    # ----------------------------
    # Predictors (minimal, faithful; avoid over-restricting valid codes to prevent N collapse)
    # ----------------------------
    # SES
    df["educ_yrs"] = coerce_valid_range(df.get("educ", np.nan), valid_min=0, valid_max=30)
    df["prestg80"] = coerce_valid_range(df.get("prestg80", np.nan), valid_min=0, valid_max=100)

    df["realinc"] = coerce_valid_range(df.get("realinc", np.nan), valid_min=0, valid_max=None)
    df["hompop"] = coerce_valid_range(df.get("hompop", np.nan), valid_min=1, valid_max=None)
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    df["female"] = make_dummy_from_codes(df.get("sex", np.nan), one_code=2, valid_codes=[1, 2])
    df["age"] = coerce_valid_range(df.get("age", np.nan), valid_min=18, valid_max=89)

    # Race: 1 white, 2 black, 3 other (assumed)
    race = coerce_valid_range(df.get("race", np.nan), valid_set=[1, 2, 3])
    df["black"] = pd.Series(np.where(race.isna(), np.nan, (race == 2).astype(float)), index=df.index)
    df["otherrace"] = pd.Series(np.where(race.isna(), np.nan, (race == 3).astype(float)), index=df.index)

    # Hispanic: use ETHNIC if present; interpret as 1=not hispanic, 2=hispanic (as in sample rows)
    if "ethnic" in df.columns:
        eth = coerce_valid_range(df["ethnic"], valid_set=[1, 2])
        df["hispanic"] = pd.Series(np.where(eth.isna(), np.nan, (eth == 2).astype(float)), index=df.index)
    else:
        df["hispanic"] = np.nan

    # Religion: norelig from RELIG == 4 (none). Do not overconstrain RELIG codes.
    relig = to_num(df.get("relig", np.nan)).replace([np.inf, -np.inf], np.nan)
    df["norelig"] = pd.Series(np.where(relig.isna(), np.nan, (relig == 4).astype(float)), index=df.index)

    # Conservative Protestant from RELIG & DENOM
    # Keep denom valid set broad to avoid turning most cases missing.
    denom = to_num(df.get("denom", np.nan)).replace([np.inf, -np.inf], np.nan)
    denom = denom.where(denom.isin([0, 1, 2, 3, 4, 5, 6, 7]), np.nan)  # common GSS denom codes subset
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # approximation: Baptist or Other Protestant
    cons = (is_prot & denom_cons)

    # Only define when both relig and denom observed; otherwise missing
    known = relig.notna() & denom.notna()
    df["cons_prot"] = pd.Series(np.where(known, cons.astype(float), np.nan), index=df.index)

    # Region: south if REGION == 3; keep region codes broad (1..9)
    region = to_num(df.get("region", np.nan)).replace([np.inf, -np.inf], np.nan)
    region = region.where(region.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    df["south"] = pd.Series(np.where(region.isna(), np.nan, (region == 3).astype(float)), index=df.index)

    # Political intolerance (0-15): sum of 15 intolerant indicators; listwise across 15 items
    tol_items = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    for v in tol_items:
        if v not in df.columns:
            df[v] = np.nan
        df[v] = to_num(df[v]).replace([np.inf, -np.inf], np.nan)

    intoler = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        s = df[v]
        intoler[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    df["pol_intol"] = intoler.sum(axis=1, min_count=len(tol_items))
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics to prevent silent N collapse
    # ----------------------------
    diag_vars = [
        "music_exclusive", "educ_yrs", "inc_pc", "prestg80", "female", "age",
        "black", "hispanic", "otherrace", "cons_prot", "norelig", "south", "pol_intol"
    ]
    diag_vars = [v for v in diag_vars if v in df.columns]
    diag = pd.DataFrame({
        "nonmissing": df[diag_vars].notna().sum(),
        "missing": df[diag_vars].isna().sum(),
        "n_unique_nonmissing": df[diag_vars].nunique(dropna=True),
    })
    with open("./output/table1_variable_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("Variable diagnostics (1993 sample):\n")
        f.write(diag.to_string())
        f.write("\n\n")
        # Show dummy distributions for key indicators
        for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
            if v in df.columns:
                vc = df[v].value_counts(dropna=False)
                f.write(f"{v} value_counts(dropna=False):\n{vc.to_string()}\n\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    dv = "music_exclusive"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    tables = {}
    fits = []

    for label, xcols in [
        ("Model 1 (SES)", m1_x),
        ("Model 2 (Demographic)", m2_x),
        ("Model 3 (Political intolerance)", m3_x),
    ]:
        tab, fit, nonmissing_before, dropped, mframe = fit_table1_ols(df, dv, xcols, label)
        tables[label] = tab
        fits.append(fit)

        write_model_output(
            label, tab, fit, nonmissing_before, dropped,
            f"./output/table1_{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt"
        )

        # Save model frame preview
        mframe_head = mframe.head(20)
        mframe_head.to_csv(
            f"./output/table1_{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model_frame_head.csv",
            index=False
        )

    fit_df = pd.DataFrame(fits)[["model", "n", "r2", "adj_r2"]]
    with open("./output/table1_fit_stats.txt", "w", encoding="utf-8") as f:
        f.write("Fit statistics (OLS):\n")
        f.write(fit_df.to_string(index=False))
        f.write("\n")

    # Also export combined tables as CSV for convenience
    for name, tab in tables.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        tab.to_csv(f"./output/table1_{safe}_table.csv", index=False)

    # Return a compact dict of DataFrames
    return {
        "fit_stats": fit_df,
        "tables": tables,
        "diagnostics": diag,
    }