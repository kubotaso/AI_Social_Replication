def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

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
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def valid_in(s, valid_codes):
        s = to_num(s)
        return s.where(s.isin(valid_codes), np.nan)

    def make_dummy_from_codes(s, one_codes, valid_codes):
        """
        Return float dummy with NaN for invalid/missing.
        one_codes can be int or list-like.
        """
        s = valid_in(s, valid_codes)
        if not isinstance(one_codes, (list, tuple, set, np.ndarray, pd.Index)):
            one_codes = [one_codes]
        out = np.where(s.isna(), np.nan, s.isin(one_codes).astype(float))
        return pd.Series(out, index=s.index)

    def nonmissing_report(df, cols):
        return df[cols].notna().sum().sort_values(ascending=False)

    def value_counts_report(s, name, max_levels=20):
        vc = s.value_counts(dropna=False)
        if len(vc) > max_levels:
            vc = vc.iloc[:max_levels]
        return pd.DataFrame({name: vc})

    def fit_table1_ols(df, dv, x_cols, model_name):
        """
        OLS with:
          - unstandardized intercept (constant)
          - standardized coefficients for predictors: beta = b * sd(x)/sd(y)
          - p-values from unstandardized OLS (same as standardized regression for slopes)
        Listwise deletion per model.
        """
        needed = [dv] + x_cols
        d = df[needed].copy()
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum().sort_index()

        d = d.dropna(axis=0, how="any").copy()

        # Drop predictors with no variation (prevents singular matrix / NaN issues)
        kept, dropped = [], []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0 or len(kept) == 0:
            coef_rows = [{"term": "Constant", "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in x_cols:
                coef_rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})
            coef_table = pd.DataFrame(coef_rows)
            fit = {"model": model_name, "n": int(len(d)), "r2": np.nan, "adj_r2": np.nan}
            return coef_table, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        # Standardized betas for predictors from unstandardized slopes
        sd_y = y.std(ddof=0)
        rows = [{
            "term": "Constant",
            "beta": float(res.params.get("const", np.nan)),
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": star(res.pvalues.get("const", np.nan)),
        }]

        for c in x_cols:
            if c in kept:
                b = float(res.params.get(c, np.nan))
                p = float(res.pvalues.get(c, np.nan))
                sd_x = d[c].astype(float).std(ddof=0)
                beta = np.nan
                if np.isfinite(b) and np.isfinite(sd_x) and np.isfinite(sd_y) and sd_y != 0:
                    beta = b * (sd_x / sd_y)
                rows.append({"term": c, "beta": beta, "p": p, "sig": star(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})

        coef_table = pd.DataFrame(rows)
        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return coef_table, fit, nonmissing_before, dropped, d

    def write_model_output(model_name, coef_table, fit, nonmissing_before, dropped_predictors, model_frame):
        path = f"./output/table1_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(model_name + "\n")
            f.write("=" * len(model_name) + "\n\n")

            f.write("Non-missing counts BEFORE listwise deletion (model variables):\n")
            f.write(nonmissing_before.to_string())
            f.write("\n\n")

            if dropped_predictors:
                f.write("Dropped predictors due to zero variance AFTER listwise deletion:\n")
                f.write(", ".join(dropped_predictors))
                f.write("\n\n")

            f.write("Fit statistics:\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.6f}\n")
            f.write(f"Adj R^2 = {fit['adj_r2']:.6f}\n\n")

            f.write("Coefficients (Table 1 style):\n")
            f.write(" - Constant is unstandardized (raw DV units)\n")
            f.write(" - Predictors are standardized coefficients (beta)\n")
            f.write(" - Two-tailed significance stars from p-values: * <.05, ** <.01, *** <.001\n\n")

            out = coef_table.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
            out["p"] = pd.to_numeric(out["p"], errors="coerce")
            out["beta"] = out["beta"].round(6)
            out["p"] = out["p"].round(6)
            f.write(out.to_string(index=False))
            f.write("\n\n")

            f.write("Model-frame quick checks (after listwise deletion):\n")
            f.write(f"Rows: {len(model_frame)}\n\n")
            for c in model_frame.columns:
                if c == "music_exclusive":
                    f.write(f"{c} descriptives:\n")
                    f.write(model_frame[c].describe().to_string())
                    f.write("\n\n")
                else:
                    if set(model_frame[c].dropna().unique()).issubset({0.0, 1.0}):
                        f.write(f"{c} value counts (0/1/NA):\n")
                        f.write(model_frame[c].value_counts(dropna=False).to_string())
                        f.write("\n\n")

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column.")
    df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: Musical exclusiveness (# genres disliked)
    # Rules:
    #  - each item valid 1..5 (else missing)
    #  - disliked indicator = 1 if 4 or 5, else 0
    #  - listwise across all 18 items (any missing -> DV missing)
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music item columns: {missing_music}")

    music = df[music_items].apply(to_num)
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float)
    disliked = disliked.where(music.notna(), np.nan)

    df["music_exclusive"] = np.nan
    complete_music = ~disliked.isna().any(axis=1)
    df.loc[complete_music, "music_exclusive"] = disliked.loc[complete_music].sum(axis=1)

    # DV descriptives output
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Musical exclusiveness (# of music genres disliked)\n")
        f.write("Construction:\n")
        f.write(" - 18 genre ratings (1..5). Non-1..5 treated as missing.\n")
        f.write(" - Disliked=1 if response in {4,5}; else 0.\n")
        f.write(" - Listwise across 18 items: if any item missing, DV missing.\n\n")
        f.write(df["music_exclusive"].describe().to_string())
        f.write("\n")

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = to_num(df.get("educ", np.nan))
    df["prestg80"] = to_num(df.get("prestg80", np.nan))

    df["realinc"] = to_num(df.get("realinc", np.nan))
    df["hompop"] = to_num(df.get("hompop", np.nan))
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    df["female"] = make_dummy_from_codes(df.get("sex", np.nan), one_codes=2, valid_codes=[1, 2])
    df["age"] = to_num(df.get("age", np.nan))

    # Race / Hispanic:
    # Use RACE for black/other; use ETHNIC (available in provided vars) for hispanic.
    # Keep as separate dummies with white non-hispanic as implied reference.
    race = valid_in(df.get("race", np.nan), [1, 2, 3])
    df["black"] = pd.Series(np.where(race.isna(), np.nan, (race == 2).astype(float)), index=df.index)
    df["otherrace"] = pd.Series(np.where(race.isna(), np.nan, (race == 3).astype(float)), index=df.index)

    # ETHNIC coding is not fully specified in excerpt; use minimal assumption:
    # treat 1/2 as valid and code hispanic=1 if ETHNIC==2, else 0 if ==1.
    if "ethnic" in df.columns:
        eth = valid_in(df["ethnic"], [1, 2])
        df["hispanic"] = pd.Series(np.where(eth.isna(), np.nan, (eth == 2).astype(float)), index=df.index)
    else:
        df["hispanic"] = np.nan

    # Religion
    relig = valid_in(df.get("relig", np.nan), list(range(1, 14)))  # conservative valid range
    denom = to_num(df.get("denom", np.nan))
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)  # GSS denom often includes 0

    df["norelig"] = pd.Series(np.where(relig.isna(), np.nan, (relig == 4).astype(float)), index=df.index)

    # Conservative Protestant:
    # Set to 0/1 for all respondents with nonmissing RELIG and DENOM.
    # Missing if RELIG or DENOM missing.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # approximation using documented denom categories
    known = relig.notna() & denom.notna()
    df["cons_prot"] = pd.Series(np.where(known, (is_prot & denom_cons).astype(float), np.nan), index=df.index)

    # Region: South
    region = valid_in(df.get("region", np.nan), list(range(1, 10)))
    df["south"] = pd.Series(np.where(region.isna(), np.nan, (region == 3).astype(float)), index=df.index)

    # Political intolerance (0-15)
    tol_items = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    missing_tol = [k for k in tol_items.keys() if k not in df.columns]
    if missing_tol:
        # Create missing columns if absent (will yield missing scale; but avoids KeyError)
        for k in missing_tol:
            df[k] = np.nan

    intoler = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        s = to_num(df[v])
        # Keep as-is but require exact match for intolerant code; other observed codes treated as tolerant(0)
        intoler[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    complete_tol = ~intoler.isna().any(axis=1)
    df["pol_intol"] = np.nan
    df.loc[complete_tol, "pol_intol"] = intoler.loc[complete_tol].sum(axis=1)

    # ----------------------------
    # Diagnostic output: missingness overview
    # ----------------------------
    diag_cols = [
        "music_exclusive", "educ_yrs", "inc_pc", "prestg80",
        "female", "age", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    diag_cols = [c for c in diag_cols if c in df.columns]
    with open("./output/table1_missingness_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("Missingness diagnostics (GSS 1993 only)\n")
        f.write("======================================\n\n")
        f.write("Non-missing counts:\n")
        f.write(df[diag_cols].notna().sum().sort_values(ascending=False).to_string())
        f.write("\n\n")
        for c in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
            if c in df.columns:
                f.write(f"{c} value counts (0/1/NA):\n")
                f.write(df[c].value_counts(dropna=False).to_string())
                f.write("\n\n")

        if "pol_intol" in df.columns:
            f.write("pol_intol descriptives (including NA):\n")
            f.write(df["pol_intol"].describe().to_string())
            f.write("\n\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    dv = "music_exclusive"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    models = [
        ("Model 1 (SES)", m1_x),
        ("Model 2 (Demographic)", m2_x),
        ("Model 3 (Political intolerance)", m3_x),
    ]

    fit_rows = []
    tables = {}
    model_frames = {}

    for model_name, x_cols in models:
        coef_table, fit, nonmissing_before, dropped, mframe = fit_table1_ols(df, dv, x_cols, model_name)
        tables[model_name] = coef_table
        model_frames[model_name] = mframe
        fit_rows.append(fit)

        write_model_output(model_name, coef_table, fit, nonmissing_before, dropped, mframe)

    fit_stats = pd.DataFrame(fit_rows)

    # Summary file aggregating fit stats + tables
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication: standardized OLS coefficients (predictors) with unstandardized constants\n")
        f.write("===========================================================================================\n\n")
        f.write("Fit statistics:\n")
        f.write(fit_stats.to_string(index=False))
        f.write("\n\n")
        for model_name, tab in tables.items():
            f.write(model_name + "\n")
            f.write("-" * len(model_name) + "\n")
            out = tab.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce").round(6)
            out["p"] = pd.to_numeric(out["p"], errors="coerce").round(6)
            f.write(out.to_string(index=False))
            f.write("\n\n")

    return {"fit_stats": fit_stats, "tables": tables, "model_frames": model_frames}