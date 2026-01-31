def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)

    # ---- Utilities ----
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def safe_zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd is None or np.isnan(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

    def fit_std_ols(data, y, xvars):
        # Subset and drop missing
        cols = [y] + xvars
        d = data.loc[:, cols].copy()
        d = d.dropna(axis=0, how="any")

        # Standardize within estimation sample
        y_std = safe_zscore(d[y])

        X_std = pd.DataFrame(index=d.index)
        dropped = []
        for v in xvars:
            z = safe_zscore(d[v])
            if z.notna().sum() == 0:
                dropped.append(v)
                continue
            # If constant after standardization (all NaN would already be caught), drop
            if np.isclose(z.var(ddof=0), 0) or np.isnan(z.var(ddof=0)):
                dropped.append(v)
                continue
            X_std[v] = z

        # Ensure nothing weird sneaks in
        X = sm.add_constant(X_std, has_constant="add")
        X = X.replace([np.inf, -np.inf], np.nan)
        ok = y_std.notna() & X.notna().all(axis=1)
        y_std = y_std.loc[ok]
        X = X.loc[ok]

        if X.shape[0] == 0 or X.shape[1] <= 1:
            raise ValueError(f"Model has no usable rows/predictors after cleaning. Dropped predictors: {dropped}")

        model = sm.OLS(y_std, X).fit()
        return model, int(X.shape[0]), dropped

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

    # ---- Year filter (defensive) ----
    if "year" in df.columns:
        df = df.loc[df["year"] == 1993].copy()
    elif "YEAR" in df.columns:
        df = df.loc[df["YEAR"] == 1993].copy()

    # Lowercase column names for robustness
    df.columns = [c.lower() for c in df.columns]

    # ---- DV: Musical exclusiveness ----
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music columns: {missing_music}")

    for c in music_items:
        df[c] = to_num(df[c])

    # Require complete cases on all 18 items
    df_music = df.dropna(subset=music_items).copy()

    # Disliked = 4 or 5
    for c in music_items:
        df_music[f"dislike_{c}"] = df_music[c].isin([4, 5]).astype(int)

    df_music["exclusiveness"] = df_music[[f"dislike_{c}" for c in music_items]].sum(axis=1).astype(float)

    # ---- Covariates ----
    # SES
    for c in ["educ", "realinc", "hompop", "prestg80"]:
        if c in df_music.columns:
            df_music[c] = to_num(df_music[c])
        else:
            df_music[c] = np.nan

    # Income per capita; avoid inf
    df_music["inc_pc"] = np.nan
    valid_inc = df_music["realinc"].notna() & df_music["hompop"].notna() & (df_music["hompop"] > 0)
    df_music.loc[valid_inc, "inc_pc"] = (df_music.loc[valid_inc, "realinc"] / df_music.loc[valid_inc, "hompop"]).astype(float)
    df_music["inc_pc"] = df_music["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Demographics/identities
    if "sex" in df_music.columns:
        df_music["sex"] = to_num(df_music["sex"])
        df_music["female"] = np.where(df_music["sex"].isin([1, 2]), (df_music["sex"] == 2).astype(float), np.nan)
    else:
        df_music["female"] = np.nan

    if "age" in df_music.columns:
        df_music["age"] = to_num(df_music["age"])
    else:
        df_music["age"] = np.nan

    if "race" in df_music.columns:
        df_music["race"] = to_num(df_music["race"])
        df_music["black"] = np.where(df_music["race"].isin([1, 2, 3]), (df_music["race"] == 2).astype(float), np.nan)
        df_music["other_race"] = np.where(df_music["race"].isin([1, 2, 3]), (df_music["race"] == 3).astype(float), np.nan)
    else:
        df_music["black"] = np.nan
        df_music["other_race"] = np.nan

    # Hispanic not available in provided dataset; omitted from models (no placeholder column used in regression)
    if "relig" in df_music.columns:
        df_music["relig"] = to_num(df_music["relig"])
        df_music["no_religion"] = np.where(df_music["relig"].isin([1, 2, 3, 4, 5]), (df_music["relig"] == 4).astype(float), np.nan)
    else:
        df_music["relig"] = np.nan
        df_music["no_religion"] = np.nan

    if "denom" in df_music.columns:
        df_music["denom"] = to_num(df_music["denom"])
    else:
        df_music["denom"] = np.nan

    # Conservative Protestant approximation: RELIG==1 and DENOM in {1,6,7}
    df_music["conserv_prot"] = np.nan
    rel_denom_ok = df_music["relig"].notna() & df_music["denom"].notna()
    df_music.loc[rel_denom_ok, "conserv_prot"] = 0.0
    df_music.loc[rel_denom_ok & (df_music["relig"] == 1) & (df_music["denom"].isin([1, 6, 7])), "conserv_prot"] = 1.0

    if "region" in df_music.columns:
        df_music["region"] = to_num(df_music["region"])
        df_music["south"] = np.where(df_music["region"].isin([1, 2, 3, 4]), (df_music["region"] == 3).astype(float), np.nan)
    else:
        df_music["south"] = np.nan

    # ---- Political intolerance scale ----
    tol_items = [
        "spkath", "colath", "libath",
        "spkrac", "colrac", "librac",
        "spkcom", "colcom", "libcom",
        "spkmil", "colmil", "libmil",
        "spkhomo", "colhomo", "libhomo"
    ]
    for c in tol_items:
        if c in df_music.columns:
            df_music[c] = to_num(df_music[c])
        else:
            df_music[c] = np.nan

    def intolerance_indicator(col, s):
        s = to_num(s)
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove=1 intolerant
        elif col.startswith("col"):
            if col == "colcom":
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired=4 intolerant
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed=5 intolerant
        return out

    intoler_df = pd.DataFrame({c: intolerance_indicator(c, df_music[c]) for c in tol_items})

    complete_tol = intoler_df.notna().all(axis=1)
    df_music["polintol"] = np.nan
    df_music.loc[complete_tol, "polintol"] = intoler_df.loc[complete_tol].sum(axis=1).astype(float)

    # ---- Models ----
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    # Ensure predictors are numeric and finite
    for c in set([y] + x_m3):
        if c in df_music.columns:
            df_music[c] = to_num(df_music[c]).replace([np.inf, -np.inf], np.nan)

    results = {}
    fit_rows = []
    dropped_predictors = {}

    m1, n1, drop1 = fit_std_ols(df_music, y, x_m1)
    results["Model1_SES"] = m1
    fit_rows.append(("Model1_SES", n1, m1.rsquared, m1.rsquared_adj))
    dropped_predictors["Model1_SES"] = drop1

    m2, n2, drop2 = fit_std_ols(df_music, y, x_m2)
    results["Model2_Demographic"] = m2
    fit_rows.append(("Model2_Demographic", n2, m2.rsquared, m2.rsquared_adj))
    dropped_predictors["Model2_Demographic"] = drop2

    m3, n3, drop3 = fit_std_ols(df_music, y, x_m3)
    results["Model3_PolIntolerance"] = m3
    fit_rows.append(("Model3_PolIntolerance", n3, m3.rsquared, m3.rsquared_adj))
    dropped_predictors["Model3_PolIntolerance"] = drop3

    fitstats = pd.DataFrame(fit_rows, columns=["model", "N", "R2", "Adj_R2"])

    # ---- Coefficient tables ----
    def coef_table(model, name):
        t = pd.DataFrame({
            "term": model.params.index,
            "beta_std": model.params.values,
            "se": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values
        })
        t.insert(0, "model", name)
        t["sig"] = t["p"].map(stars)
        return t

    regtab = pd.concat([coef_table(m, name) for name, m in results.items()], ignore_index=True)

    # Pivot (exclude const)
    reg_noconst = regtab.loc[regtab["term"] != "const"].copy()
    reg_noconst["beta_star"] = reg_noconst["beta_std"].map(lambda x: f"{x: .4f}") + reg_noconst["sig"]

    pivot = reg_noconst.pivot(index="term", columns="model", values="beta_star")

    # Order terms by fullest model, then append any extras
    term_order = [t for t in x_m3 if t in pivot.index]
    extras = [t for t in pivot.index.tolist() if t not in term_order]
    pivot = pivot.reindex(index=term_order + extras)

    # ---- Save outputs ----
    # 1) Summary
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS with standardized coefficients (DV = # music genres disliked)\n")
    summary_lines.append("DV construction: sum over 18 genres of I(response in {4,5}); listwise complete cases on all 18 items.\n")
    summary_lines.append("Note: Hispanic indicator not available in provided dataset excerpt; omitted.\n")
    summary_lines.append("Model fit statistics:\n")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("\n\nDropped predictors due to no variance / unusable after cleaning (if any):\n")
    for k, v in dropped_predictors.items():
        summary_lines.append(f"{k}: {v if v else 'None'}")
    summary_lines.append("\n")

    for name, model in results.items():
        summary_lines.append(f"\n==== {name} ====\n")
        summary_lines.append(model.summary().as_text())

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 2) Regression table
    with open("./output/regression_table_std_betas.txt", "w", encoding="utf-8") as f:
        f.write("Standardized coefficients (beta) from OLS; stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(pivot.to_string())
        f.write("\n\nModel fit stats:\n")
        f.write(fitstats.to_string(index=False))
        f.write("\n")

    # 3) Long-form coefficients
    regtab_out = regtab.loc[:, ["model", "term", "beta_std", "se", "t", "p", "sig"]].copy()
    regtab_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)

    return {
        "fit_stats": fitstats,
        "regression_table": pivot,
        "coefficients_long": regtab_out
    }