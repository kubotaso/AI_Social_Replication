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

    def zscore_series(s):
        s = s.astype(float)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def make_binary_from_codes(s, valid_codes, one_code):
        """
        For GSS-style numeric variables:
        - keep only values in valid_codes; others -> NaN
        - return 1.0 if == one_code else 0.0
        """
        s = to_num(s)
        s = s.where(s.isin(valid_codes), np.nan)
        return np.where(s.isna(), np.nan, (s == one_code).astype(float))

    def fit_table1_style(df_in, dv, x_cols, model_name):
        """
        Fit OLS and output:
        - standardized coefficients (beta) for predictors (as in Table 1)
        - constant unstandardized
        - R2, Adj R2, N
        Also safeguards:
        - listwise deletion on dv + x_cols
        - drops any predictor that becomes constant after deletion (records it)
        """
        needed = [dv] + x_cols
        d = df_in[needed].copy()

        # Ensure numeric, convert inf -> nan
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        # Listwise deletion for the model
        d = d.dropna(axis=0, how="any").copy()

        dropped_constant = []
        kept = []
        for c in x_cols:
            v = d[c].astype(float)
            if v.nunique(dropna=True) <= 1:
                dropped_constant.append(c)
            else:
                kept.append(c)

        # If nothing left, fail gracefully
        if len(d) == 0 or len(kept) == 0:
            tab = pd.DataFrame(
                [{"term": "Constant", "beta": np.nan, "p": np.nan, "sig": ""}]
                + [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols]
            )
            fit = {"model": model_name, "n": int(len(d)), "r2": np.nan, "adj_r2": np.nan}
            return tab, fit, d, dropped_constant

        y = d[dv].astype(float)
        X = d[kept].astype(float)

        # Raw model (for intercept and fit stats)
        X_raw = sm.add_constant(X, has_constant="add")
        res_raw = sm.OLS(y, X_raw).fit()

        # Standardized model: standardize y and all X; no intercept
        y_std = zscore_series(y)
        X_std = pd.DataFrame({c: zscore_series(X[c]) for c in kept}, index=X.index)

        # Safety: if any NaNs introduced by zero-variance, drop those columns (shouldn't happen due to check)
        X_std = X_std.replace([np.inf, -np.inf], np.nan)
        bad_cols = [c for c in X_std.columns if X_std[c].isna().any()]
        if bad_cols:
            # If z-scoring introduced NaNs (should only be if sd==0), drop those predictors
            X_std = X_std.drop(columns=bad_cols)
            kept = [c for c in kept if c not in bad_cols]

        # Still need complete cases (zscore doesn't add missing, but keep safe)
        mm = pd.concat([y_std, X_std], axis=1).dropna()
        y_std2 = mm.iloc[:, 0]
        X_std2 = mm.iloc[:, 1:]

        if len(mm) == 0 or X_std2.shape[1] == 0:
            tab = pd.DataFrame(
                [{"term": "Constant", "beta": float(res_raw.params.get("const", np.nan)), "p": float(res_raw.pvalues.get("const", np.nan)), "sig": star(res_raw.pvalues.get("const", np.nan))}]
                + [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols]
            )
            fit = {"model": model_name, "n": int(res_raw.nobs), "r2": float(res_raw.rsquared), "adj_r2": float(res_raw.rsquared_adj)}
            return tab, fit, d, dropped_constant

        res_std = sm.OLS(y_std2, X_std2).fit()

        # Build Table 1-style output: constant (unstandardized), predictors (standardized betas)
        rows = []
        rows.append(
            {
                "term": "Constant",
                "beta": float(res_raw.params.get("const", np.nan)),
                "p": float(res_raw.pvalues.get("const", np.nan)),
                "sig": star(res_raw.pvalues.get("const", np.nan)),
            }
        )

        # For predictors: use standardized coefficient estimate from standardized regression
        for c in x_cols:
            if c in kept:
                b = float(res_std.params.get(c, np.nan))
                p = float(res_std.pvalues.get(c, np.nan))
                rows.append({"term": c, "beta": b, "p": p, "sig": star(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        fit = {
            "model": model_name,
            "n": int(res_raw.nobs),
            "r2": float(res_raw.rsquared),
            "adj_r2": float(res_raw.rsquared_adj),
        }
        return tab, fit, d, dropped_constant

    def write_text_report(path, model_name, fit, tab, missing_report=None, dropped_constant=None):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("=" * len(model_name) + "\n\n")
            f.write("Fit:\n")
            f.write(f"N = {fit.get('n', np.nan)}\n")
            f.write(f"R^2 = {fit.get('r2', np.nan):.4f}\n" if pd.notna(fit.get("r2", np.nan)) else "R^2 = NA\n")
            f.write(f"Adj R^2 = {fit.get('adj_r2', np.nan):.4f}\n\n" if pd.notna(fit.get("adj_r2", np.nan)) else "Adj R^2 = NA\n\n")

            if missing_report is not None:
                f.write("Non-missing counts (before listwise deletion):\n")
                f.write(missing_report.to_string())
                f.write("\n\n")

            if dropped_constant:
                f.write("Dropped predictors due to zero variance after listwise deletion:\n")
                f.write(", ".join(dropped_constant) + "\n\n")

            out = tab.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce").round(4)
            out["p"] = pd.to_numeric(out["p"], errors="coerce").round(4)
            f.write("Table 1-style coefficients:\n")
            f.write("(Constant shown in raw DV units; predictors shown as standardized betas)\n\n")
            f.write(out.to_string(index=False))
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
    # DV: musical exclusiveness (# genres disliked), listwise across 18 items
    # Codes: 1..5 valid; disliked = 4 or 5; DK/others missing
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

    disliked = (music.isin([4, 5])).astype(float)
    disliked = disliked.where(music.notna(), np.nan)

    df["music_exclusive"] = disliked.sum(axis=1, min_count=len(music_items))
    df.loc[disliked.isna().any(axis=1), "music_exclusive"] = np.nan

    # ----------------------------
    # SES predictors
    # ----------------------------
    df["educ"] = to_num(df.get("educ", np.nan)).replace([np.inf, -np.inf], np.nan)
    df["prestg80"] = to_num(df.get("prestg80", np.nan)).replace([np.inf, -np.inf], np.nan)

    df["realinc"] = to_num(df.get("realinc", np.nan)).replace([np.inf, -np.inf], np.nan)
    df["hompop"] = to_num(df.get("hompop", np.nan)).replace([np.inf, -np.inf], np.nan)
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # ----------------------------
    # Demographics / group identity
    # ----------------------------
    # Female: SEX 1=male 2=female
    df["female"] = make_binary_from_codes(df.get("sex", np.nan), valid_codes=[1, 2], one_code=2)

    # Age
    df["age"] = to_num(df.get("age", np.nan)).replace([np.inf, -np.inf], np.nan)

    # Race dummies from RACE: 1=white 2=black 3=other
    race = to_num(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: use ETHNIC if available; common coding 1=not hispanic, 2=hispanic
    if "ethnic" in df.columns:
        eth = to_num(df["ethnic"])
        eth = eth.where(eth.isin([1, 2]), np.nan)
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
    else:
        df["hispanic"] = np.nan

    # No religion: RELIG==4 (none). Keep only common valid codes; other -> NaN
    relig = to_num(df.get("relig", np.nan))
    # Conservative valid set; allow 0? not sure; keep 1..13-ish if present, but at least 1..9 from sample mapping
    relig = relig.where(relig.isin(list(range(1, 14))), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant (best-effort given available vars):
    # Protestant if RELIG==1; then "conservative" denom approximated as Baptist (1) or Other Protestant (6).
    denom = to_num(df.get("denom", np.nan))
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)  # allow some reasonable range
    cons_mask = (relig == 1) & denom.isin([1, 6])
    known_mask = relig.notna() & denom.notna()
    df["cons_prot"] = np.where(known_mask, cons_mask.astype(float), np.nan)

    # South: REGION==3
    region = to_num(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance scale (0-15), listwise across 15 items
    # ----------------------------
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

    tol = df[list(tol_items.keys())].copy()
    # Construct intolerant indicators; keep missing as NaN
    intoler = pd.DataFrame(index=df.index)
    for v, code_bad in tol_items.items():
        s = tol[v]
        intoler[v] = np.where(s.isna(), np.nan, (s == code_bad).astype(float))

    df["pol_intol"] = intoler.sum(axis=1, min_count=len(tol_items))
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # DV descriptives output
    # ----------------------------
    dv_desc = df["music_exclusive"].describe()
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Musical exclusiveness (# of music genres disliked)\n")
        f.write("Construction: count of 18 genres rated 4 or 5; listwise across 18 items; DK/NA treated as missing.\n\n")
        f.write(dv_desc.to_string())
        f.write("\n")

    # ----------------------------
    # Model specs (Table 1)
    # ----------------------------
    dv = "music_exclusive"
    models = {
        "Model 1 (SES)": ["educ", "inc_pc", "prestg80"],
        "Model 2 (Demographic)": [
            "educ", "inc_pc", "prestg80",
            "female", "age", "black", "hispanic", "otherrace",
            "cons_prot", "norelig", "south",
        ],
        "Model 3 (Political intolerance)": [
            "educ", "inc_pc", "prestg80",
            "female", "age", "black", "hispanic", "otherrace",
            "cons_prot", "norelig", "south",
            "pol_intol",
        ],
    }

    # ----------------------------
    # Fit models, save outputs
    # ----------------------------
    all_tabs = {}
    fit_rows = []

    for model_name, xcols in models.items():
        # Missingness audit (before listwise deletion)
        audit_cols = [dv] + xcols
        nonmissing = df[audit_cols].notna().sum().sort_values(ascending=False)
        missing_report = pd.DataFrame(
            {"nonmissing": nonmissing, "missing": (len(df) - nonmissing)}
        )

        tab, fit, mframe, dropped_constant = fit_table1_style(df, dv, xcols, model_name)
        all_tabs[model_name] = tab
        fit_rows.append(fit)

        safe = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        write_text_report(
            f"./output/table1_{safe}.txt",
            model_name=model_name,
            fit=fit,
            tab=tab,
            missing_report=missing_report,
            dropped_constant=dropped_constant,
        )

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv("./output/table1_fit_stats.csv", index=False)

    # Also write a compact combined text summary
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication (computed from data)\n")
        f.write("====================================\n\n")
        f.write("Fit statistics:\n")
        f.write(fit_df.to_string(index=False))
        f.write("\n\n")
        for model_name, tab in all_tabs.items():
            f.write(model_name + "\n")
            f.write("-" * len(model_name) + "\n")
            out = tab.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce").round(4)
            out["p"] = pd.to_numeric(out["p"], errors="coerce").round(4)
            f.write(out.to_string(index=False))
            f.write("\n\n")

    return {"fit_stats": fit_df, "tables": all_tabs}