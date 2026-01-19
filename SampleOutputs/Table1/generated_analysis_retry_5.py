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

    def dummy_from(s, one_code, valid_codes=None):
        s = to_num(s)
        if valid_codes is not None:
            s = s.where(s.isin(valid_codes), np.nan)
        return pd.Series(np.where(s.isna(), np.nan, (s == one_code).astype(float)), index=s.index)

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
            f.write("- Predictors are standardized betas (β)\n")
            f.write("- Stars from two-tailed p-values: * <.05, ** <.01, *** <.001\n\n")
            out = coef_table.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
            out["p"] = pd.to_numeric(out["p"], errors="coerce")
            out["beta"] = out["beta"].round(6)
            out["p"] = out["p"].round(6)
            f.write(out.to_string(index=False))
            f.write("\n")

    def fit_standardized_ols(df, dv, x_cols, model_name):
        # Model frame and numeric conversion
        needed = [dv] + x_cols
        d = df[needed].copy()
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum().sort_index()

        # Listwise deletion (per-model)
        d = d.dropna(axis=0, how="any").copy()

        # Drop any predictors that became constant
        kept = []
        dropped = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        # If no data or no predictors, return empty-ish results safely
        if len(d) == 0 or len(kept) == 0:
            coef_table = pd.DataFrame(
                [{"term": "Constant", "beta": np.nan, "p": np.nan, "sig": ""}]
                + [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols]
            )
            fit = {"model": model_name, "n": int(len(d)), "r2": np.nan, "adj_r2": np.nan}
            return coef_table, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)

        # Raw OLS (for intercept + fit stats)
        X_raw = sm.add_constant(X, has_constant="add")
        res_raw = sm.OLS(y, X_raw).fit()

        # Standardized betas via standardized regression on y and X (no intercept)
        y_std = zscore(y)
        X_std = pd.DataFrame({c: zscore(X[c]) for c in kept}, index=X.index)

        # Ensure no missing in standardized arrays
        mm = pd.concat([y_std.rename("_y"), X_std], axis=1).dropna(axis=0, how="any")
        if len(mm) == 0 or mm.shape[1] <= 1:
            coef_table = pd.DataFrame(
                [{"term": "Constant", "beta": float(res_raw.params.get("const", np.nan)),
                  "p": float(res_raw.pvalues.get("const", np.nan)), "sig": star(res_raw.pvalues.get("const", np.nan))}]
                + [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols]
            )
            fit = {"model": model_name, "n": int(res_raw.nobs), "r2": float(res_raw.rsquared), "adj_r2": float(res_raw.rsquared_adj)}
            return coef_table, fit, nonmissing_before, dropped, d

        y_std2 = mm["_y"]
        X_std2 = mm.drop(columns=["_y"])

        # This should now be NaN-free; still guard
        if not np.isfinite(X_std2.to_numpy()).all() or not np.isfinite(y_std2.to_numpy()).all():
            coef_table = pd.DataFrame(
                [{"term": "Constant", "beta": float(res_raw.params.get("const", np.nan)),
                  "p": float(res_raw.pvalues.get("const", np.nan)), "sig": star(res_raw.pvalues.get("const", np.nan))}]
                + [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols]
            )
            fit = {"model": model_name, "n": int(res_raw.nobs), "r2": float(res_raw.rsquared), "adj_r2": float(res_raw.rsquared_adj)}
            return coef_table, fit, nonmissing_before, dropped, d

        res_std = sm.OLS(y_std2, X_std2).fit()

        rows = []
        rows.append({
            "term": "Constant",
            "beta": float(res_raw.params.get("const", np.nan)),
            "p": float(res_raw.pvalues.get("const", np.nan)),
            "sig": star(res_raw.pvalues.get("const", np.nan)),
        })
        for c in x_cols:
            if c in kept and c in res_std.params.index:
                b = float(res_std.params[c])
                p = float(res_std.pvalues[c])
                rows.append({"term": c, "beta": b, "p": p, "sig": star(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})

        coef_table = pd.DataFrame(rows)
        fit = {"model": model_name, "n": int(res_raw.nobs), "r2": float(res_raw.rsquared), "adj_r2": float(res_raw.rsquared_adj)}
        return coef_table, fit, nonmissing_before, dropped, d

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: Musical exclusiveness (# genres disliked), listwise across 18 items
    # disliked = 4 or 5; valid responses 1..5; other codes -> missing
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
    disliked_ind = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    # Require complete responses on all 18 items
    df["music_exclusive"] = disliked_ind.sum(axis=1, min_count=len(music_items))
    df.loc[disliked_ind.isna().any(axis=1), "music_exclusive"] = np.nan

    # DV descriptives
    dv = df["music_exclusive"]
    dv_desc = dv.describe()
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Musical exclusiveness (# of music genres disliked)\n")
        f.write("Count of 18 genres rated 4 ('dislike') or 5 ('dislike very much').\n")
        f.write("Listwise across 18 genre items; non-1..5 codes treated as missing.\n\n")
        f.write(dv_desc.to_string())
        f.write("\n")

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = to_num(df.get("educ", np.nan)).replace([np.inf, -np.inf], np.nan)
    df["prestg80"] = to_num(df.get("prestg80", np.nan)).replace([np.inf, -np.inf], np.nan)

    df["realinc"] = to_num(df.get("realinc", np.nan)).replace([np.inf, -np.inf], np.nan)
    df["hompop"] = to_num(df.get("hompop", np.nan)).replace([np.inf, -np.inf], np.nan)
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics / group identity
    # female: SEX 1=male 2=female
    df["female"] = dummy_from(df.get("sex", np.nan), one_code=2, valid_codes=[1, 2])

    df["age"] = to_num(df.get("age", np.nan)).replace([np.inf, -np.inf], np.nan)

    # race: 1 white, 2 black, 3 other
    race = valid_in(df.get("race", np.nan), [1, 2, 3])
    df["black"] = pd.Series(np.where(race.isna(), np.nan, (race == 2).astype(float)), index=df.index)
    df["otherrace"] = pd.Series(np.where(race.isna(), np.nan, (race == 3).astype(float)), index=df.index)

    # hispanic: use ETHNIC if present; accept 1/2 (not/yes); otherwise missing
    if "ethnic" in df.columns:
        eth = valid_in(df["ethnic"], [1, 2])
        df["hispanic"] = pd.Series(np.where(eth.isna(), np.nan, (eth == 2).astype(float)), index=df.index)
    else:
        df["hispanic"] = np.nan

    # religion: norelig from RELIG==4 among valid [1..13] (conservative); others -> missing
    relig = valid_in(df.get("relig", np.nan), list(range(1, 14)))
    df["norelig"] = pd.Series(np.where(relig.isna(), np.nan, (relig == 4).astype(float)), index=df.index)

    # conservative protestant: only defined when RELIG and DENOM observed.
    # Conservative denom approximation (documentation-supported): Baptist (1) or Other Protestant (6) among Protestants.
    denom = to_num(df.get("denom", np.nan))
    denom = denom.where(np.isfinite(denom), np.nan)
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)

    is_prot = relig == 1
    denom_cons = denom.isin([1, 6])
    known = relig.notna() & denom.notna()
    df["cons_prot"] = pd.Series(np.where(known, (is_prot & denom_cons).astype(float), np.nan), index=df.index)

    # south: REGION==3, valid 1..9
    region = valid_in(df.get("region", np.nan), list(range(1, 10)))
    df["south"] = pd.Series(np.where(region.isna(), np.nan, (region == 3).astype(float)), index=df.index)

    # Political intolerance (0-15): sum of 15 intolerant indicators; listwise across 15
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

    tol_frame = df[list(tol_items.keys())].copy()
    intoler = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        s = tol_frame[v]
        intoler[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))
    df["pol_intol"] = intoler.sum(axis=1, min_count=len(tol_items))
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    dv_name = "music_exclusive"

    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    # Fit
    coef1, fit1, nonmiss1, dropped1, mf1 = fit_standardized_ols(df, dv_name, m1_x, "Model 1 (SES)")
    coef2, fit2, nonmiss2, dropped2, mf2 = fit_standardized_ols(df, dv_name, m2_x, "Model 2 (Demographic)")
    coef3, fit3, nonmiss3, dropped3, mf3 = fit_standardized_ols(df, dv_name, m3_x, "Model 3 (Political intolerance)")

    # Save model outputs
    write_model_txt("./output/table1_model1.txt", "Model 1 (SES)", fit1, coef1, nonmiss1, dropped1)
    write_model_txt("./output/table1_model2.txt", "Model 2 (Demographic)", fit2, coef2, nonmiss2, dropped2)
    write_model_txt("./output/table1_model3.txt", "Model 3 (Political intolerance)", fit3, coef3, nonmiss3, dropped3)

    # Overall summary file
    fit_stats = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2"]]
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication summary (computed from raw data)\n")
        f.write("====================================================\n\n")
        f.write("Fit statistics:\n")
        f.write(fit_stats.to_string(index=False))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("- Predictor coefficients are standardized betas (β) from OLS on z-scored DV and X's (no intercept).\n")
        f.write("- Constant is unstandardized intercept from OLS on raw DV with intercept.\n")
        f.write("- Each model uses listwise deletion on exactly the variables included in that model.\n")

    # Also save machine-readable CSVs
    fit_stats.to_csv("./output/table1_fit_stats.csv", index=False)
    coef1.to_csv("./output/table1_model1_coefs.csv", index=False)
    coef2.to_csv("./output/table1_model2_coefs.csv", index=False)
    coef3.to_csv("./output/table1_model3_coefs.csv", index=False)

    # Missingness audit (helps diagnose N collapse)
    audit_cols = [dv_name] + sorted(set(m3_x))
    audit = df[audit_cols].notna().sum().sort_values(ascending=False)
    with open("./output/table1_missingness_audit.txt", "w", encoding="utf-8") as f:
        f.write("Non-missing counts per variable (1993 sample; before any model-specific listwise deletion)\n")
        f.write("-----------------------------------------------------------------------------\n\n")
        f.write(audit.to_string())
        f.write("\n")

    return {
        "fit_stats": fit_stats,
        "tables": {
            "Model 1 (SES)": coef1,
            "Model 2 (Demographic)": coef2,
            "Model 3 (Political intolerance)": coef3,
        },
        "model_frames": {
            "Model 1 (SES)": mf1,
            "Model 2 (Demographic)": mf2,
            "Model 3 (Political intolerance)": mf3,
        },
    }