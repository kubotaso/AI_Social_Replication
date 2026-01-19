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

    def zscore(series):
        s = series.astype(float)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd == 0 or np.isnan(sd):
            return s * np.nan
        return (s - mu) / sd

    def star(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def regression_table_ols(y, X, model_name):
        # Fit OLS with statsmodels; report standardized betas (including dummies) per Table 1 style
        Xc = X.copy()
        Xc = sm.add_constant(Xc, has_constant="add")
        mod = sm.OLS(y, Xc, missing="drop")
        res = mod.fit()

        # Standardized coefficients: beta_j = b_j * sd(x_j) / sd(y)
        y_sd = np.std(res.model.endog, ddof=0)
        betas = {}
        for col in X.columns:
            x = res.model.exog[:, list(res.model.exog_names).index(col)]
            x_sd = np.std(x, ddof=0)
            b = res.params.get(col, np.nan)
            betas[col] = (b * x_sd / y_sd) if (y_sd and x_sd) else np.nan

        rows = []
        for col in ["const"] + list(X.columns):
            if col == "const":
                rows.append(
                    {
                        "term": "Constant",
                        "b": res.params.get("const", np.nan),
                        "se": res.bse.get("const", np.nan),
                        "t": res.tvalues.get("const", np.nan),
                        "p": res.pvalues.get("const", np.nan),
                        "beta_std": np.nan,
                        "sig": star(res.pvalues.get("const", np.nan)),
                    }
                )
            else:
                rows.append(
                    {
                        "term": col,
                        "b": res.params.get(col, np.nan),
                        "se": res.bse.get(col, np.nan),
                        "t": res.tvalues.get(col, np.nan),
                        "p": res.pvalues.get(col, np.nan),
                        "beta_std": betas.get(col, np.nan),
                        "sig": star(res.pvalues.get(col, np.nan)),
                    }
                )

        out = pd.DataFrame(rows)
        fit = {
            "model": model_name,
            "n": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "aic": float(res.aic),
            "bic": float(res.bic),
        }
        return res, out, fit

    def write_model_text(path, model_name, res, table_df, fit):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("=" * len(model_name) + "\n\n")
            f.write("Fit statistics:\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.4f}\n")
            f.write(f"Adj. R^2 = {fit['adj_r2']:.4f}\n")
            f.write(f"AIC = {fit['aic']:.2f}\n")
            f.write(f"BIC = {fit['bic']:.2f}\n\n")
            f.write("Coefficients (unstandardized b, SE, p; standardized beta shown as beta_std):\n\n")
            show = table_df.copy()
            show["b"] = show["b"].astype(float).round(4)
            show["se"] = show["se"].astype(float).round(4)
            show["t"] = show["t"].astype(float).round(3)
            show["p"] = show["p"].astype(float).round(4)
            show["beta_std"] = show["beta_std"].astype(float).round(4)
            f.write(show.to_string(index=False))
            f.write("\n\n")
            f.write("Statsmodels summary:\n\n")
            f.write(res.summary().as_text())

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)

    # Normalize column names to lowercase for robustness
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993 if year exists
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: musical exclusiveness (# genres disliked) from 18 items
    # Listwise requirement on the 18 music items.
    # Disliked = 4 or 5; Like/neutral = 1,2,3; missing otherwise.
    # ----------------------------
    music_items = [
        "bigband",
        "blugrass",
        "country",
        "blues",
        "musicals",
        "classicl",
        "folk",
        "gospel",
        "jazz",
        "latin",
        "moodeasy",
        "newage",
        "opera",
        "rap",
        "reggae",
        "conrock",
        "oldies",
        "hvymetal",
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = df[music_items].apply(to_num)

    # Keep only 1..5; treat everything else as missing
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.apply(lambda s: np.where(s.isna(), np.nan, np.where(s.isin([4, 5]), 1.0, 0.0)))
    disliked = pd.DataFrame(disliked, columns=music_items, index=df.index)

    # Listwise requirement across all 18: any missing -> DV missing
    any_missing_music = disliked.isna().any(axis=1)
    df["music_exclusive"] = disliked.sum(axis=1, min_count=len(music_items))
    df.loc[any_missing_music, "music_exclusive"] = np.nan

    # ----------------------------
    # IVs: SES
    # ----------------------------
    df["educ"] = to_num(df.get("educ", np.nan))
    df["prestg80"] = to_num(df.get("prestg80", np.nan))

    # Per-capita income: realinc / hompop (as instructed)
    df["realinc"] = to_num(df.get("realinc", np.nan))
    df["hompop"] = to_num(df.get("hompop", np.nan))
    df["inc_pc"] = np.where(
        (df["realinc"].notna()) & (df["hompop"].notna()) & (df["hompop"] > 0),
        df["realinc"] / df["hompop"],
        np.nan,
    )

    # ----------------------------
    # Demographics / group identity
    # ----------------------------
    df["sex"] = to_num(df.get("sex", np.nan))
    df["female"] = np.where(df["sex"] == 2, 1.0, np.where(df["sex"] == 1, 0.0, np.nan))

    df["age"] = to_num(df.get("age", np.nan))

    df["race"] = to_num(df.get("race", np.nan))
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["otherrace"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: use 'ethnic' if present (common in provided sample)
    # Conservative choice: treat ethnic == 2 as Hispanic if coded that way; otherwise leave missing.
    # If the coding differs, this at least runs without error and flags via missingness.
    if "ethnic" in df.columns:
        df["ethnic"] = to_num(df["ethnic"])
        # Best-effort: many GSS files use 1=not hispanic, 2=hispanic. Keep only 1/2.
        df["hispanic"] = np.where(df["ethnic"].isin([1, 2]), (df["ethnic"] == 2).astype(float), np.nan)
    else:
        df["hispanic"] = np.nan

    df["relig"] = to_num(df.get("relig", np.nan))
    df["denom"] = to_num(df.get("denom", np.nan))
    df["region"] = to_num(df.get("region", np.nan))

    # No religion: relig == 4
    df["norelig"] = np.where(df["relig"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: derived from relig==1 and denom in a "conservative" set.
    # NOTE: GSS DENOM coding varies; keep a defensible, documented approach:
    # Treat Baptists and "other Protestant" as conservative-ish; exclude mainline categories.
    # Here: denom==1 (Baptist) or denom==6 (other) as conservative; denom==7 (no denom) excluded.
    df["cons_prot"] = np.nan
    is_prot = df["relig"] == 1
    denom_cons = df["denom"].isin([1, 6])
    denom_known = df["denom"].notna()
    relig_known = df["relig"].notna()
    idx = relig_known & denom_known
    # FIX for previous crash: assign scalar series aligned to index, not a mismatched ndarray
    df.loc[idx, "cons_prot"] = ((is_prot & denom_cons).loc[idx]).astype(float)

    # Southern: region == 3
    df["south"] = np.where(df["region"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), (df["region"] == 3).astype(float), np.nan)

    # ----------------------------
    # Political intolerance scale (0-15)
    # ----------------------------
    tol_items = {
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

    for v in tol_items.keys():
        if v not in df.columns:
            df[v] = np.nan
        df[v] = to_num(df[v])

    tol_df = df[list(tol_items.keys())].copy()
    # Keep only expected codes (1..n); anything else missing
    # (We don't know full ranges per item; but equality checks below will handle.)
    intolerant_ind = pd.DataFrame(index=df.index)
    for v, (_, bad_code) in tol_items.items():
        s = tol_df[v]
        intolerant_ind[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    df["pol_intol"] = intolerant_ind.sum(axis=1, min_count=len(tol_items))

    # ----------------------------
    # Build models (listwise per model)
    # ----------------------------
    dv = "music_exclusive"

    model_specs = {
        "Model 1 (SES)": ["educ", "inc_pc", "prestg80"],
        "Model 2 (Demographic)": [
            "educ",
            "inc_pc",
            "prestg80",
            "female",
            "age",
            "black",
            "hispanic",
            "otherrace",
            "cons_prot",
            "norelig",
            "south",
        ],
        "Model 3 (Political intolerance)": [
            "educ",
            "inc_pc",
            "prestg80",
            "female",
            "age",
            "black",
            "hispanic",
            "otherrace",
            "cons_prot",
            "norelig",
            "south",
            "pol_intol",
        ],
    }

    results_tables = {}
    fit_rows = []
    text_lines = []

    # Basic DV descriptives (computed)
    dv_desc = df[dv].describe()
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Musical exclusiveness (# genres disliked)\n")
        f.write("Constructed as count of 18 genres rated 4 or 5; listwise across 18 items.\n\n")
        f.write(dv_desc.to_string())
        f.write("\n")

    for name, predictors in model_specs.items():
        needed = [dv] + predictors
        dsub = df[needed].copy()

        # Ensure numeric
        for c in needed:
            dsub[c] = to_num(dsub[c])

        # Listwise delete for the model
        dsub = dsub.dropna(axis=0, how="any")
        y = dsub[dv].astype(float)
        X = dsub[predictors].astype(float)

        res, coef_table, fit = regression_table_ols(y, X, name)
        results_tables[name] = coef_table
        fit_rows.append(fit)

        # Save per-model text
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        write_model_text(f"./output/table1_{safe_name}.txt", name, res, coef_table, fit)

        # Quick summary line
        text_lines.append(
            f"{name}: N={fit['n']}, R2={fit['r2']:.4f}, AdjR2={fit['adj_r2']:.4f}"
        )

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv("./output/table1_fit_stats.csv", index=False)

    # Combined coefficient tables as CSV for convenience
    for name, tab in results_tables.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        tab.to_csv(f"./output/table1_{safe_name}_coefs.csv", index=False)

    # Human-readable combined summary
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication (computed from data)\n")
        f.write("Standardized OLS coefficients reported as beta_std; significance stars are two-tailed.\n\n")
        f.write("\n".join(text_lines))
        f.write("\n\nFit statistics:\n")
        f.write(fit_df.to_string(index=False))
        f.write("\n")

    # Return a dict of DataFrames for downstream checks
    return {
        "fit_stats": fit_df,
        "Model 1 (SES)": results_tables["Model 1 (SES)"],
        "Model 2 (Demographic)": results_tables["Model 2 (Demographic)"],
        "Model 3 (Political intolerance)": results_tables["Model 3 (Political intolerance)"],
    }