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

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def valid_in(s, valid_codes):
        s = to_num(s)
        return s.where(s.isin(valid_codes), np.nan)

    def make_dummy(s, one_code, valid_codes=None):
        s = to_num(s)
        if valid_codes is not None:
            s = s.where(s.isin(valid_codes), np.nan)
        out = pd.Series(np.nan, index=s.index, dtype=float)
        m = s.notna()
        out.loc[m] = (s.loc[m] == one_code).astype(float)
        return out

    def standardize_beta_from_unstd(res, X, y):
        # beta_j = b_j * sd(x_j) / sd(y)
        sd_y = y.std(ddof=0)
        betas = {}
        if sd_y == 0 or pd.isna(sd_y):
            for k in X.columns:
                betas[k] = np.nan
            return betas
        for k in X.columns:
            sd_x = X[k].std(ddof=0)
            b = res.params.get(k, np.nan)
            if pd.isna(sd_x) or sd_x == 0 or pd.isna(b):
                betas[k] = np.nan
            else:
                betas[k] = float(b) * float(sd_x) / float(sd_y)
        return betas

    def fit_table1_style(df, dv, x_cols, model_name):
        needed = [dv] + x_cols
        d = df[needed].copy()
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum().sort_index()

        # listwise deletion per model
        d = d.dropna(axis=0, how="any").copy()

        # drop constant predictors (avoid NaN/collinearity artifacts)
        kept, dropped = [], []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            tab = pd.DataFrame(
                [{"term": "Constant", "beta": np.nan, "p": np.nan, "sig": ""}]
                + [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols]
            )
            return tab, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardize_beta_from_unstd(res, X, y)

        rows = []
        rows.append(
            {
                "term": "Constant",
                "beta": float(res.params.get("const", np.nan)),
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": star(res.pvalues.get("const", np.nan)),
            }
        )
        for c in x_cols:
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({"term": c, "beta": float(betas.get(c, np.nan)), "p": p, "sig": star(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        fit = {
            "model": model_name,
            "n": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
        }
        return tab, fit, nonmissing_before, dropped, d

    def write_model_txt(path, model_name, fit, tab, nonmissing_before, dropped_predictors, dframe):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("=" * len(model_name) + "\n\n")

            f.write("Non-missing counts BEFORE listwise deletion (model variables):\n")
            f.write(nonmissing_before.to_string())
            f.write("\n\n")

            f.write("Estimation sample size (after listwise deletion):\n")
            f.write(f"N = {fit['n']}\n\n")

            if dropped_predictors:
                f.write("Dropped predictors due to zero variance AFTER listwise deletion:\n")
                f.write(", ".join(dropped_predictors) + "\n\n")

            f.write("Fit statistics (unstandardized OLS fit):\n")
            f.write(f"R^2 = {fit['r2']}\n")
            f.write(f"Adjusted R^2 = {fit['adj_r2']}\n\n")

            f.write("Table 1-style coefficients:\n")
            f.write("- Constant is unstandardized (raw DV units).\n")
            f.write("- Predictors are standardized coefficients (beta = b * SD(X) / SD(Y)).\n")
            f.write("- Stars from two-tailed p-values: * <.05, ** <.01, *** <.001.\n\n")

            out = tab.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
            out["p"] = pd.to_numeric(out["p"], errors="coerce")
            out["beta"] = out["beta"].round(6)
            out["p"] = out["p"].round(6)
            f.write(out.to_string(index=False))
            f.write("\n\n")

            f.write("Quick checks (estimation frame):\n")
            f.write(dframe.describe(include="all").to_string())
            f.write("\n")

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # Dependent variable: count of 18 genres disliked (4 or 5), listwise across 18 items
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
    # keep only valid 1..5; other codes treated as missing
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float)
    disliked = disliked.where(music.notna(), np.nan)

    df["music_exclusive"] = np.nan
    complete_music = ~disliked.isna().any(axis=1)
    df.loc[complete_music, "music_exclusive"] = disliked.loc[complete_music].sum(axis=1)

    # DV descriptives
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Musical exclusiveness (# of music genres disliked)\n")
        f.write("Constructed as count across 18 genres with response 4 ('dislike') or 5 ('dislike very much').\n")
        f.write("Non-1..5 codes treated as missing. DV set to missing if any of 18 items missing.\n\n")
        f.write(df["music_exclusive"].describe().to_string())
        f.write("\n")

    # ----------------------------
    # Independent variables
    # ----------------------------
    # SES
    df["educ_yrs"] = to_num(df.get("educ", np.nan)).replace([np.inf, -np.inf], np.nan)
    df["prestg80"] = to_num(df.get("prestg80", np.nan)).replace([np.inf, -np.inf], np.nan)

    realinc = to_num(df.get("realinc", np.nan)).replace([np.inf, -np.inf], np.nan)
    hompop = to_num(df.get("hompop", np.nan)).replace([np.inf, -np.inf], np.nan)
    hompop = hompop.where(hompop > 0, np.nan)
    inc_pc = realinc / hompop
    inc_pc = inc_pc.replace([np.inf, -np.inf], np.nan)
    df["inc_pc"] = inc_pc

    # Demographics / identities
    df["female"] = make_dummy(df.get("sex", np.nan), one_code=2, valid_codes=[1, 2])
    df["age"] = to_num(df.get("age", np.nan)).replace([np.inf, -np.inf], np.nan)

    race = valid_in(df.get("race", np.nan), [1, 2, 3])  # 1 white, 2 black, 3 other
    df["black"] = pd.Series(np.nan, index=df.index, dtype=float)
    df["otherrace"] = pd.Series(np.nan, index=df.index, dtype=float)
    m = race.notna()
    df.loc[m, "black"] = (race.loc[m] == 2).astype(float)
    df.loc[m, "otherrace"] = (race.loc[m] == 3).astype(float)

    # Hispanic: use 'ethnic' from the provided dataset (assumed 1=not hispanic, 2=hispanic)
    if "ethnic" in df.columns:
        eth = valid_in(df["ethnic"], [1, 2])
        df["hispanic"] = pd.Series(np.nan, index=df.index, dtype=float)
        m = eth.notna()
        df.loc[m, "hispanic"] = (eth.loc[m] == 2).astype(float)
    else:
        df["hispanic"] = np.nan

    # Religion: norelig from RELIG==4 (none) using valid codes seen in file (use 1..13 fallback)
    relig = valid_in(df.get("relig", np.nan), list(range(1, 14)))
    df["norelig"] = pd.Series(np.nan, index=df.index, dtype=float)
    m = relig.notna()
    df.loc[m, "norelig"] = (relig.loc[m] == 4).astype(float)

    # Conservative Protestant: defined only when relig and denom are observed.
    # Use a simple denom-based approximation among Protestants: Baptist(1) or Other Protestant(6).
    denom = to_num(df.get("denom", np.nan)).replace([np.inf, -np.inf], np.nan)
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)

    df["cons_prot"] = pd.Series(np.nan, index=df.index, dtype=float)
    known = relig.notna() & denom.notna()
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df.loc[known, "cons_prot"] = (is_prot.loc[known] & denom_cons.loc[known]).astype(float)

    # South: REGION==3 among valid 1..9
    region = valid_in(df.get("region", np.nan), list(range(1, 10)))
    df["south"] = pd.Series(np.nan, index=df.index, dtype=float)
    m = region.notna()
    df.loc[m, "south"] = (region.loc[m] == 3).astype(float)

    # Political intolerance (0-15): sum of 15 intolerant indicators; listwise across 15 items
    tol_bad = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    for v in tol_bad.keys():
        if v not in df.columns:
            df[v] = np.nan
        df[v] = to_num(df[v]).replace([np.inf, -np.inf], np.nan)

    tol = df[list(tol_bad.keys())].copy()
    intoler = pd.DataFrame(index=df.index, columns=list(tol_bad.keys()), dtype=float)
    for v, bad_code in tol_bad.items():
        s = tol[v]
        intoler[v] = np.nan
        m = s.notna()
        intoler.loc[m, v] = (s.loc[m] == bad_code).astype(float)

    df["pol_intol"] = np.nan
    complete_tol = ~intoler.isna().any(axis=1)
    df.loc[complete_tol, "pol_intol"] = intoler.loc[complete_tol].sum(axis=1)

    with open("./output/table1_predictor_missingness.txt", "w", encoding="utf-8") as f:
        cols = [
            "music_exclusive", "educ_yrs", "inc_pc", "prestg80",
            "female", "age", "black", "hispanic", "otherrace",
            "cons_prot", "norelig", "south", "pol_intol"
        ]
        cols = [c for c in cols if c in df.columns]
        miss = pd.DataFrame({
            "nonmissing": df[cols].notna().sum(),
            "missing": df[cols].isna().sum(),
            "nunique_nonmissing": df[cols].nunique(dropna=True),
        }).sort_index()
        f.write(miss.to_string())
        f.write("\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    dv = "music_exclusive"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    model_specs = [
        ("Model 1 (SES)", m1_x),
        ("Model 2 (Demographic)", m2_x),
        ("Model 3 (Political intolerance)", m3_x),
    ]

    tables = {}
    fit_stats = []
    model_frames = {}

    for model_name, x_cols in model_specs:
        tab, fit, nonmissing_before, dropped, dframe = fit_table1_style(df, dv, x_cols, model_name)
        tables[model_name] = tab
        fit_stats.append(fit)
        model_frames[model_name] = dframe

        write_model_txt(
            f"./output/table1_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.txt",
            model_name, fit, tab, nonmissing_before, dropped, dframe
        )

    fit_df = pd.DataFrame(fit_stats)

    # Write a combined summary
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication (computed from provided GSS 1993 extract)\n")
        f.write("===========================================================\n\n")
        f.write("Fit statistics:\n")
        f.write(fit_df.to_string(index=False))
        f.write("\n\n")
        for model_name in tables:
            f.write(model_name + "\n")
            f.write("-" * len(model_name) + "\n")
            f.write(tables[model_name].to_string(index=False))
            f.write("\n\n")

    return {
        "fit_stats": fit_df,
        "tables": tables,
        "model_frames": model_frames,
    }