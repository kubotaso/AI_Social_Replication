def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)

    # --- Helpers ---
    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd == 0 or np.isnan(sd):
            return s * np.nan
        return (s - mu) / sd

    def to_num(series):
        return pd.to_numeric(series, errors="coerce")

    def fit_std_ols(data, y, xvars):
        cols = [y] + xvars
        d = data[cols].copy()
        d = d.dropna()

        y_std = zscore(d[y])
        X_std = pd.DataFrame({v: zscore(d[v]) for v in xvars})
        X = sm.add_constant(X_std, has_constant="add")

        model = sm.OLS(y_std, X).fit()
        return model, d.shape[0]

    # --- Year filter (defensive; file is already 1993) ---
    if "year" in df.columns:
        df = df.loc[df["year"] == 1993].copy()
    elif "YEAR" in df.columns:
        df = df.loc[df["YEAR"] == 1993].copy()

    # --- DV: musical exclusiveness (count of 18 disliked genres; complete cases required) ---
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing music columns: {missing_music}")

    for c in music_items:
        df[c] = to_num(df[c])

    # Drop cases with any missing in the 18 items (per paper rule)
    df_music = df.dropna(subset=music_items).copy()

    # Dislike indicator: 1 if 4 or 5; 0 if 1-3; otherwise NaN (but we already required non-missing)
    dislike = {}
    for c in music_items:
        dislike[c] = df_music[c].isin([4, 5]).astype(int)

    df_music["exclusiveness"] = pd.DataFrame(dislike).sum(axis=1).astype(float)

    # --- Covariates ---
    # SES
    for c in ["educ", "realinc", "hompop", "prestg80"]:
        if c in df_music.columns:
            df_music[c] = to_num(df_music[c])

    df_music["inc_pc"] = np.where(
        (df_music["hompop"].notna()) & (df_music["realinc"].notna()) & (df_music["hompop"] > 0),
        df_music["realinc"] / df_music["hompop"],
        np.nan
    )

    # Demographics / identities
    if "sex" in df_music.columns:
        df_music["sex"] = to_num(df_music["sex"])
        df_music["female"] = np.where(df_music["sex"].isin([1, 2]), (df_music["sex"] == 2).astype(float), np.nan)
    else:
        df_music["female"] = np.nan

    if "age" in df_music.columns:
        df_music["age"] = to_num(df_music["age"])

    if "race" in df_music.columns:
        df_music["race"] = to_num(df_music["race"])
        df_music["black"] = np.where(df_music["race"].isin([1, 2, 3]), (df_music["race"] == 2).astype(float), np.nan)
        df_music["other_race"] = np.where(df_music["race"].isin([1, 2, 3]), (df_music["race"] == 3).astype(float), np.nan)
    else:
        df_music["black"] = np.nan
        df_music["other_race"] = np.nan

    # Hispanic not available in provided columns; omit (keep placeholder for reporting)
    df_music["hispanic"] = np.nan

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

    # Conservative Protestant approximation per mapping instruction (coarse DENOM recode)
    # conserv_prot = 1 if RELIG==1 and DENOM in {1,6,7}; else 0; missing if RELIG/DENOM missing.
    df_music["conserv_prot"] = np.nan
    mask_rel_denom = df_music["relig"].notna() & df_music["denom"].notna()
    df_music.loc[mask_rel_denom, "conserv_prot"] = 0.0
    df_music.loc[mask_rel_denom & (df_music["relig"] == 1) & (df_music["denom"].isin([1, 6, 7])), "conserv_prot"] = 1.0

    if "region" in df_music.columns:
        df_music["region"] = to_num(df_music["region"])
        df_music["south"] = np.where(df_music["region"].isin([1, 2, 3, 4]), (df_music["region"] == 3).astype(float), np.nan)
    else:
        df_music["south"] = np.nan

    # --- Political intolerance scale (0-15): complete cases on 15 tolerance items ---
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

    def intolerance_indicator(col, series):
        s = series
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            out.loc[s.isin([1, 2])] = (s.loc[s.isin([1, 2])] == 2).astype(float)
        elif col.startswith("lib"):
            out.loc[s.isin([1, 2])] = (s.loc[s.isin([1, 2])] == 1).astype(float)  # remove=1 intolerant
        elif col.startswith("col"):
            if col == "colcom":
                out.loc[s.isin([4, 5])] = (s.loc[s.isin([4, 5])] == 4).astype(float)  # fired=4 intolerant
            else:
                out.loc[s.isin([4, 5])] = (s.loc[s.isin([4, 5])] == 5).astype(float)  # not allowed=5 intolerant
        return out

    intoler = {}
    for c in tol_items:
        intoler[c] = intolerance_indicator(c, df_music[c])
    intoler_df = pd.DataFrame(intoler)

    df_music["polintol"] = np.nan
    complete_tol = intoler_df.notna().all(axis=1)
    df_music.loc[complete_tol, "polintol"] = intoler_df.loc[complete_tol].sum(axis=1).astype(float)

    # --- Model specs ---
    y = "exclusiveness"
    x_m1 = ["educ", "inc_pc", "prestg80"]
    x_m2 = x_m1 + ["female", "age", "black", "other_race", "conserv_prot", "no_religion", "south"]
    x_m3 = x_m2 + ["polintol"]

    # Ensure core numeric coercion for predictors
    for c in set(x_m1 + x_m2 + x_m3 + [y]):
        if c in df_music.columns:
            df_music[c] = to_num(df_music[c])

    results = {}
    rows = []

    # Fit models
    m1, n1 = fit_std_ols(df_music, y, x_m1)
    results["Model1_SES"] = m1
    rows.append(("Model1_SES", n1, m1.rsquared, m1.rsquared_adj))

    m2, n2 = fit_std_ols(df_music, y, x_m2)
    results["Model2_Demographic"] = m2
    rows.append(("Model2_Demographic", n2, m2.rsquared, m2.rsquared_adj))

    m3, n3 = fit_std_ols(df_music, y, x_m3)
    results["Model3_PolIntolerance"] = m3
    rows.append(("Model3_PolIntolerance", n3, m3.rsquared, m3.rsquared_adj))

    # --- Build regression table (standardized betas) ---
    def coef_table(model, name):
        t = pd.DataFrame({
            "term": model.params.index,
            "beta_std": model.params.values,
            "se": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values
        })
        t.insert(0, "model", name)
        return t

    regtab = pd.concat([coef_table(results[k], k) for k in results], ignore_index=True)

    # Add star notation (two-tailed)
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

    regtab["sig"] = regtab["p"].map(stars)
    regtab["beta_std_fmt"] = regtab["beta_std"].map(lambda x: f"{x: .4f}" if pd.notna(x) else "")
    regtab["p_fmt"] = regtab["p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")

    # Fit stats
    fitstats = pd.DataFrame(rows, columns=["model", "N", "R2", "Adj_R2"])

    # --- Save human-readable text outputs ---
    # 1) Summary file
    summary_lines = []
    summary_lines.append("Replication output: Table 1-style OLS with standardized coefficients (DV = # music genres disliked)\n")
    summary_lines.append("DV construction: sum(dislike==4/5) across 18 genres; listwise complete on all 18 items.\n")
    summary_lines.append("Note: Hispanic indicator not available in provided data; omitted from models.\n")

    summary_lines.append("Model fit statistics:\n")
    summary_lines.append(fitstats.to_string(index=False))
    summary_lines.append("\n\n")

    for k, model in results.items():
        summary_lines.append(f"==== {k} ====\n")
        summary_lines.append(model.summary().as_text())
        summary_lines.append("\n\n")

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 2) Regression table file (compact)
    # Pivot to term x model with beta(stars)
    reg_noconst = regtab.loc[regtab["term"] != "const"].copy()
    reg_noconst["beta_star"] = reg_noconst["beta_std"].map(lambda x: f"{x: .4f}" if pd.notna(x) else "") + reg_noconst["sig"]
    pivot = reg_noconst.pivot_table(index="term", columns="model", values="beta_star", aggfunc="first")
    pivot = pivot.reindex(index=[t for t in x_m3 if t in pivot.index])  # order terms as in fullest model where present

    with open("./output/regression_table_std_betas.txt", "w", encoding="utf-8") as f:
        f.write("Standardized coefficients (beta) from OLS; stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(pivot.to_string())
        f.write("\n\n")
        f.write("Model fit stats:\n")
        f.write(fitstats.to_string(index=False))
        f.write("\n")

    # 3) Save the long-form coefficient table as TSV for auditing
    regtab_out = regtab[["model", "term", "beta_std", "se", "t", "p", "sig"]].copy()
    regtab_out.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)

    return {
        "fit_stats": fitstats,
        "regression_table": pivot,
        "coefficients_long": regtab_out
    }