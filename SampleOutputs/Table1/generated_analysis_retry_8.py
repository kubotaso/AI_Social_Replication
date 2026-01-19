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

    def valid_codes(series, codes):
        s = to_num(series)
        return s.where(s.isin(codes), np.nan)

    def make_dummy(series, one, valid=None):
        s = to_num(series)
        if valid is not None:
            s = s.where(s.isin(valid), np.nan)
        return pd.Series(np.where(s.isna(), np.nan, (s == one).astype(float)), index=s.index)

    def standardized_betas_from_raw(res, X, y):
        # beta_j = b_j * sd(X_j)/sd(y), intercept stays unstandardized
        y_sd = float(np.nanstd(y, ddof=0))
        betas = {}
        for c in X.columns:
            x_sd = float(np.nanstd(X[c], ddof=0))
            b = float(res.params.get(c, np.nan))
            if y_sd == 0 or pd.isna(y_sd) or x_sd == 0 or pd.isna(x_sd):
                betas[c] = np.nan
            else:
                betas[c] = b * (x_sd / y_sd)
        return betas

    def fit_table1_style(df, dv, x_cols, model_name):
        # Model frame
        needed = [dv] + x_cols
        d = df[needed].copy()
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum().sort_index()

        # Listwise deletion per model
        d = d.dropna(axis=0, how="any").copy()

        # Drop predictors that became constant
        kept, dropped = [], []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            tab = pd.DataFrame([{"term": "Constant", "beta": np.nan, "p": np.nan, "sig": ""}] +
                               [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols])
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            return tab, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)

        # Raw OLS (for intercept, fit stats, and p-values)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        # Standardized betas computed from raw regression
        betas = standardized_betas_from_raw(res, X, y)

        rows = []
        rows.append({
            "term": "Constant",
            "beta": float(res.params.get("const", np.nan)),
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": star(float(res.pvalues.get("const", np.nan)))
        })

        for c in x_cols:
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({"term": c, "beta": float(betas.get(c, np.nan)), "p": p, "sig": star(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return tab, fit, nonmissing_before, dropped, d

    def write_model_txt(path, model_name, fit, coef_table, nonmissing_before, dropped_predictors):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("=" * len(model_name) + "\n\n")
            f.write("Non-missing counts BEFORE listwise deletion (model variables):\n")
            f.write(nonmissing_before.to_string())
            f.write("\n\n")
            if dropped_predictors:
                f.write("Dropped predictors due to zero variance AFTER listwise deletion:\n")
                f.write(", ".join(dropped_predictors) + "\n\n")

            f.write("Fit statistics:\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.6f}\n")
            f.write(f"Adj R^2 = {fit['adj_r2']:.6f}\n\n")

            f.write("Table 1-style coefficients:\n")
            f.write("- Constant is unstandardized (raw DV units)\n")
            f.write("- Predictors are standardized coefficients (β), computed from the raw OLS fit as b*sd(x)/sd(y)\n")
            f.write("- Stars from two-tailed p-values: * <.05, ** <.01, *** <.001\n\n")
            out = coef_table.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce").round(6)
            out["p"] = pd.to_numeric(out["p"], errors="coerce").round(6)
            f.write(out.to_string(index=False))
            f.write("\n")

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to year 1993 (if present)
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: Number of music genres disliked (0-18), listwise across 18 items
    # Each item valid 1..5; disliked = 4 or 5; other codes => missing
    # If ANY of 18 missing => DV missing
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

    df["num_genres_disliked"] = np.nan
    complete_music = ~disliked.isna().any(axis=1)
    df.loc[complete_music, "num_genres_disliked"] = disliked.loc[complete_music].sum(axis=1)

    # DV descriptives
    dv = df["num_genres_disliked"]
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Number of music genres disliked (0-18)\n")
        f.write("Count of 18 genre items rated 4 ('dislike') or 5 ('dislike very much').\n")
        f.write("Responses outside 1..5 treated as missing; listwise across all 18 items.\n\n")
        f.write(dv.describe().to_string())
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
    df["female"] = make_dummy(df.get("sex", np.nan), one=2, valid=[1, 2])
    df["age"] = to_num(df.get("age", np.nan))

    # Race: 1 white, 2 black, 3 other
    race = valid_codes(df.get("race", np.nan), [1, 2, 3])
    df["black"] = pd.Series(np.where(race.isna(), np.nan, (race == 2).astype(float)), index=df.index)
    df["otherrace"] = pd.Series(np.where(race.isna(), np.nan, (race == 3).astype(float)), index=df.index)

    # Hispanic: use ETHNIC where available; treat 1/2 as valid (1=not hispanic, 2=hispanic)
    # If ETHNIC is present but coded differently, this will show up in missingness diagnostics.
    if "ethnic" in df.columns:
        eth = valid_codes(df["ethnic"], [1, 2])
        df["hispanic"] = pd.Series(np.where(eth.isna(), np.nan, (eth == 2).astype(float)), index=df.index)
    else:
        df["hispanic"] = np.nan

    # Religion: RELIG (commonly 1=Protestant, 2=Catholic, 3=Jewish, 4=None, 5=Other)
    relig = valid_codes(df.get("relig", np.nan), [1, 2, 3, 4, 5])
    df["norelig"] = pd.Series(np.where(relig.isna(), np.nan, (relig == 4).astype(float)), index=df.index)

    # Conservative Protestant: derived from RELIG and DENOM.
    # Keep it simple and documentation-consistent: among Protestants (RELIG==1),
    # define conservative as DENOM in {1 (Baptist), 6 (Other Protestant)}; else 0.
    denom = to_num(df.get("denom", np.nan))
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)

    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    known_both = relig.notna() & denom.notna()
    df["cons_prot"] = pd.Series(np.where(known_both, (is_prot & denom_cons).astype(float), np.nan), index=df.index)

    # South: REGION==3 (valid 1..9 assumed)
    region = valid_codes(df.get("region", np.nan), list(range(1, 10)))
    df["south"] = pd.Series(np.where(region.isna(), np.nan, (region == 3).astype(float)), index=df.index)

    # Political intolerance (0-15): sum of 15 intolerant indicators; listwise across 15
    tol_items = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    for v in tol_items.keys():
        if v not in df.columns:
            df[v] = np.nan
        df[v] = to_num(df[v]).replace([np.inf, -np.inf], np.nan)

    tol_df = df[list(tol_items.keys())].copy()
    intoler = pd.DataFrame(index=df.index, columns=list(tol_items.keys()), dtype=float)
    for v, intoler_code in tol_items.items():
        s = tol_df[v]
        intoler[v] = np.where(s.isna(), np.nan, (s == intoler_code).astype(float))

    df["pol_intol"] = np.nan
    complete_tol = ~intoler.isna().any(axis=1)
    df.loc[complete_tol, "pol_intol"] = intoler.loc[complete_tol].sum(axis=1)

    # Political intolerance descriptives
    with open("./output/table1_polintol_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("Political intolerance scale (0-15)\n")
        f.write("Sum of 15 intolerant responses (5 target groups x 3 contexts).\n")
        f.write("Item-level missing treated as missing; listwise across 15 items.\n\n")
        f.write(df["pol_intol"].describe().to_string())
        f.write("\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    dv_name = "num_genres_disliked"

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

    # Write a simple missingness audit for all analysis variables
    audit_vars = [dv_name] + sorted(set(m1_x + m2_x + m3_x))
    audit = pd.DataFrame({
        "nonmissing": df[audit_vars].notna().sum(),
        "missing": df[audit_vars].isna().sum(),
        "n_unique_nonmissing": df[audit_vars].nunique(dropna=True),
    }).sort_index()
    with open("./output/table1_missingness_audit.txt", "w", encoding="utf-8") as f:
        f.write("Missingness audit (after recodes; before model-specific listwise deletion)\n\n")
        f.write(audit.to_string())
        f.write("\n")

    for model_name, x_cols in models:
        tab, fit, nonmissing_before, dropped, mframe = fit_table1_style(df, dv_name, x_cols, model_name)
        tables[model_name] = tab
        model_frames[model_name] = mframe
        fit_rows.append(fit)

        # Save per-model text file
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        write_model_txt(f"./output/table1_{safe_name}.txt", model_name, fit, tab, nonmissing_before, dropped)

    fit_stats = pd.DataFrame(fit_rows)

    # Save combined summary
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication summary (computed from provided GSS 1993 extract)\n")
        f.write("DV: number of music genres disliked (0-18)\n")
        f.write("Models: OLS with standardized coefficients (β) for predictors; intercept unstandardized.\n\n")
        f.write("Fit statistics:\n")
        f.write(fit_stats.to_string(index=False))
        f.write("\n\n")
        for model_name in [m[0] for m in models]:
            f.write(model_name + "\n")
            f.write("-" * len(model_name) + "\n")
            f.write(tables[model_name].to_string(index=False))
            f.write("\n\n")

    # Return results
    return {
        "fit_stats": fit_stats,
        "tables": tables,
        "model_frames": model_frames,
        "missingness_audit": audit,
    }