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

    def zscore_series(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    # More conservative missing-code handling. Dataset appears already mostly numeric/NaN,
    # but some GSS extracts use negative or 8/9/98/99 codes. We only coerce the most common.
    def clean_gss_like_missing(s):
        s = to_num(s)
        # Common "not asked/refused/dk/na" style codes across many GSS extracts:
        # negative codes and 8/9/98/99/998/999.
        bad = set([-1, -2, -3, -4, -5, -6, -7, -8, -9, 8, 9, 98, 99, 998, 999])
        return s.where(~s.isin(bad), np.nan)

    # Standardized beta from unstandardized slope: beta = b * sd(x) / sd(y)
    # Uses sample SD (ddof=0) to match zscore_series.
    def standardized_betas_from_raw(res, y, X_cols):
        y_sd = to_num(y).std(skipna=True, ddof=0)
        out = {}
        for c in X_cols:
            if c in res.params.index:
                x_sd = to_num(res.model.exog[:, list(res.params.index).index(c)]).std(ddof=0) if False else None
            # We'll compute from the original data columns instead (safer).
        # implemented below where we have df frame and columns

    def fit_table1_ols(df, dv, x_cols, model_name):
        needed = [dv] + x_cols
        d = df[needed].copy()

        # force numeric + finite
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum()

        # listwise deletion for this model
        d = d.dropna(axis=0, how="any").copy()

        # drop constant/zero-variance predictors (avoid NaN coef rows)
        kept = []
        dropped = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0 or len(kept) == 0:
            coef = pd.DataFrame([{"term": "Constant", "beta": np.nan, "sig": ""}] +
                                [{"term": c, "beta": np.nan, "sig": ""} for c in x_cols])
            fit = {"model": model_name, "n": int(len(d)), "r2": np.nan, "adj_r2": np.nan}
            return coef, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        # standardized betas for predictors (NOT for constant)
        y_sd = y.std(ddof=0)
        betas = {}
        for c in kept:
            x_sd = X[c].std(ddof=0)
            if pd.isna(x_sd) or x_sd == 0 or pd.isna(y_sd) or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(res.params[c] * (x_sd / y_sd))

        rows = []
        # Constant: unstandardized
        p_const = float(res.pvalues.get("const", np.nan))
        rows.append({"term": "Constant", "beta": float(res.params.get("const", np.nan)), "sig": stars(p_const)})

        # Predictors: standardized betas, stars from the raw-model p-values (same t-tests)
        for c in x_cols:
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({"term": c, "beta": betas.get(c, np.nan), "sig": stars(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "sig": ""})

        coef = pd.DataFrame(rows)
        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return coef, fit, nonmissing_before, dropped, d

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[clean_gss_like_missing(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: Number of genres disliked (0-18), listwise across 18 items
    # "disliked" = 4 or 5; valid = 1..5; other -> missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = df[music_items].apply(clean_gss_like_missing)
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1)
    # listwise requirement for DV construction
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Constructed as count of 18 genres rated 4 ('dislike') or 5 ('dislike very much').\n"
        "Non-1..5 responses treated as missing; listwise across 18 genre items.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss_like_missing(df.get("educ", np.nan))
    df["prestg80"] = clean_gss_like_missing(df.get("prestg80", np.nan))

    realinc = clean_gss_like_missing(df.get("realinc", np.nan))
    hompop = clean_gss_like_missing(df.get("hompop", np.nan))
    hompop = hompop.where(hompop > 0, np.nan)
    df["inc_pc"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Demographics / group identity
    sex = clean_gss_like_missing(df.get("sex", np.nan))
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age"] = clean_gss_like_missing(df.get("age", np.nan))

    race = clean_gss_like_missing(df.get("race", np.nan)).where(lambda s: s.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if available. In many GSS extracts, ETHNIC is asked broadly.
    # Treat values 1/2 as not/yes; otherwise missing.
    if "ethnic" in df.columns:
        eth = clean_gss_like_missing(df["ethnic"]).where(lambda s: s.isin([1, 2]), np.nan)
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
    else:
        df["hispanic"] = np.nan

    # Religion dummies
    relig = clean_gss_like_missing(df.get("relig", np.nan))
    # Some extracts code RELIG: 1=Protestant, 2=Catholic, 3=Jewish, 4=None, 5=Other.
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant (approximation using DENOM among Protestants):
    # Define conservative as Baptist (1) or other Protestant (6) among RELIG==1.
    denom = clean_gss_like_missing(df.get("denom", np.nan))
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)
    known_rel = relig.notna()
    is_prot = relig == 1
    is_cons_denom = denom.isin([1, 6])
    # If RELIG is known but DENOM missing (e.g., non-Protestant), set 0.
    # If RELIG missing, keep missing.
    df["cons_prot"] = np.where(
        known_rel,
        np.where(is_prot & denom.notna(), (is_cons_denom).astype(float), 0.0),
        np.nan
    )
    # For Protestants with DENOM missing, keep missing (can't classify)
    df.loc[is_prot & denom.isna(), "cons_prot"] = np.nan

    region = clean_gss_like_missing(df.get("region", np.nan))
    region = region.where(region.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15) from 15 items; listwise across the 15
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
        df[v] = clean_gss_like_missing(df[v])

    tol = df[list(tol_items.keys())].copy()
    intoler = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        s = tol[v]
        intoler[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    df["pol_intol"] = intoler.sum(axis=1)
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    dv = "num_genres_disliked"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    t1_m1, fit1, before1, dropped1, frame1 = fit_table1_ols(df, dv, m1_x, "Model 1 (SES)")
    t1_m2, fit2, before2, dropped2, frame2 = fit_table1_ols(df, dv, m2_x, "Model 2 (Demographic)")
    t1_m3, fit3, before3, dropped3, frame3 = fit_table1_ols(df, dv, m3_x, "Model 3 (Political intolerance)")

    # ----------------------------
    # Write human-readable outputs
    # ----------------------------
    def format_model_txt(model_name, fit, nonmissing_before, dropped, table, x_cols):
        # Use 3 decimals like typical published tables
        tab = table.copy()
        tab["beta"] = pd.to_numeric(tab["beta"], errors="coerce")
        tab["beta"] = tab["beta"].round(3)
        tab["beta_star"] = tab["beta"].astype(object)
        tab.loc[tab["term"] != "Constant", "beta_star"] = tab.loc[tab["term"] != "Constant", "beta"].astype(object) + tab.loc[tab["term"] != "Constant", "sig"]
        tab.loc[tab["term"] == "Constant", "beta_star"] = tab.loc[tab["term"] == "Constant", "beta"].astype(object) + tab.loc[tab["term"] == "Constant", "sig"]

        # Keep just the display columns
        disp = tab[["term", "beta_star"]].copy()
        disp.columns = ["term", "beta (std.) / constant (unstd.)"]

        txt = []
        txt.append(model_name)
        txt.append("=" * len(model_name))
        txt.append("")
        txt.append("Fit statistics:")
        txt.append(f"N = {fit['n']}")
        txt.append(f"R^2 = {fit['r2']:.3f}" if pd.notna(fit["r2"]) else "R^2 = NA")
        txt.append(f"Adj R^2 = {fit['adj_r2']:.3f}" if pd.notna(fit["adj_r2"]) else "Adj R^2 = NA")
        txt.append("")
        txt.append("Notes:")
        txt.append("- Predictors reported as standardized coefficients (beta = b * SD(X)/SD(Y)).")
        txt.append("- Constant is unstandardized intercept from raw OLS.")
        txt.append("- Stars from two-tailed p-values: * p<.05, ** p<.01, *** p<.001.")
        txt.append("")
        txt.append("Non-missing counts BEFORE listwise deletion (DV + predictors):")
        txt.append(nonmissing_before.to_string())
        txt.append("")
        if dropped:
            txt.append("Dropped predictors due to no variance AFTER listwise deletion:")
            txt.append(", ".join(dropped))
            txt.append("")
        txt.append("Regression table:")
        txt.append(disp.to_string(index=False))
        txt.append("")
        return "\n".join(txt)

    write_text("./output/table1_model1_ses.txt", format_model_txt("Model 1 (SES)", fit1, before1, dropped1, t1_m1, m1_x))
    write_text("./output/table1_model2_demographic.txt", format_model_txt("Model 2 (Demographic)", fit2, before2, dropped2, t1_m2, m2_x))
    write_text("./output/table1_model3_political_intolerance.txt", format_model_txt("Model 3 (Political intolerance)", fit3, before3, dropped3, t1_m3, m3_x))

    # Overall summary file
    fit_df = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2"]]
    write_text(
        "./output/table1_fit_summary.txt",
        "Table 1 fit summary\n"
        "===================\n\n"
        + fit_df.to_string(index=False)
        + "\n"
    )

    # Return results as dict of DataFrames
    return {
        "fit_stats": fit_df,
        "Model 1 (SES)": t1_m1,
        "Model 2 (Demographic)": t1_m2,
        "Model 3 (Political intolerance)": t1_m3,
        "model_frames": {
            "Model 1 (SES)": frame1,
            "Model 2 (Demographic)": frame2,
            "Model 3 (Political intolerance)": frame3,
        },
    }