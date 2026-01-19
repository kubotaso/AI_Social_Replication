def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative GSS missing handling:
        - Coerce to numeric
        - Treat common special codes as missing
        Note: This dataset is a selected extract; we avoid overly aggressive rules.
        """
        x = to_num(x).copy()
        special = {
            8, 9, 98, 99, 998, 999, 9998, 9999,
            0,  # often "not applicable" for some items; safer to set missing then validate per-variable
        }
        x = x.mask(x.isin(list(special)))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items: 1-5 where 4/5 = dislike; 1/2/3 = not-dislike.
        Missing if not in 1..5 after cleaning.
        """
        x = clean_gss_missing(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_gss_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_series(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_dislike_count(df_, items, require_all=True):
        mat = pd.concat([likert_dislike_indicator(df_[c]).rename(c) for c in items], axis=1)
        if require_all:
            return mat.sum(axis=1, min_count=len(items))
        else:
            # not used; kept for transparency
            return mat.sum(axis=1, min_count=1)

    def fit_ols_with_standardized_betas(df_model, y_col, x_cols, model_name):
        """
        Fit OLS on raw DV, compute standardized betas as:
            beta_j = b_j * sd(x_j) / sd(y)
        using the estimation sample (listwise deletion on y and X).
        """
        needed = [y_col] + x_cols
        d = df_model[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if d.shape[0] < (len(x_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = to_num(d[y_col])
        X = d[x_cols].apply(to_num)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        # standardized betas from unstandardized coefficients
        y_sd = y.std(ddof=1)
        betas = {}
        for c in x_cols:
            x_sd = X[c].std(ddof=1)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = res.params[c] * (x_sd / y_sd)

        beta_table = pd.DataFrame(
            {
                "beta_std": pd.Series(betas),
                "b_unstd": res.params[x_cols],
                "std_err": res.bse[x_cols],
                "t": res.tvalues[x_cols],
                "p_value": res.pvalues[x_cols],
            }
        )

        def star(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        beta_table["stars"] = beta_table["p_value"].map(star)
        beta_table["beta_std_with_stars"] = beta_table["beta_std"].map(lambda v: np.nan if pd.isna(v) else float(v))
        beta_table["beta_std_with_stars"] = beta_table.apply(
            lambda r: ("" if pd.isna(r["beta_std"]) else f"{r['beta_std']:.3f}{r['stars']}"), axis=1
        )

        fit = {
            "model": model_name,
            "n": int(res.nobs),
            "k_including_const": int(res.df_model + 1),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "intercept_unstd": float(res.params["const"]),
        }

        # Save summary + tables
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
            f.write("\n\nStandardized betas computed as b * sd(x)/sd(y) on estimation sample.\n")
            f.write("\nFit:\n")
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(beta_table[["beta_std", "stars", "beta_std_with_stars", "b_unstd", "std_err", "t", "p_value"]].to_string())

        return res, beta_table, pd.DataFrame([fit])

    # -------------------------
    # Filter to 1993
    # -------------------------
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    # Bryson: DK treated as missing; cases with missing excluded -> require all component items
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all=True)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing required column: educ")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing required columns: realinc and/or hompop")
    realinc = clean_gss_missing(df["realinc"]).where(clean_gss_missing(df["realinc"]) > 0)
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing required column: prestg80")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"]).where(clean_gss_missing(df["prestg80"]) > 0)

    if "sex" not in df.columns:
        raise ValueError("Missing required column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing required column: age")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing required column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: if not present, create from ethnic as a best-effort proxy.
    # NOTE: This dataset lacks a direct hispanic flag; we create a conservative proxy:
    # mark Hispanic=1 if ETHNIC indicates Mexican/Puerto Rican/other Spanish origin.
    # If ethnic coding is not compatible, this will be mostly missing and will be dropped by listwise deletion.
    if "hispanic" in df.columns:
        hisp = binary_from_codes(df["hispanic"], true_codes=[1], false_codes=[2, 0])
        df["hispanic_flag"] = hisp
    elif "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        # Common GSS ETHNIC codes (vary by year). We treat these as Hispanic/Latino-origin proxies if present.
        # Keep only if non-missing; otherwise remain NaN.
        # This is a best-effort; it may not reproduce the paper exactly without the proper hispanic variable.
        hisp_codes = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}
        df["hispanic_flag"] = np.where(eth.isna(), np.nan, eth.isin(list(hisp_codes)).astype(float))
    else:
        df["hispanic_flag"] = np.nan

    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing required columns: relig and/or denom")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    if "region" not in df.columns:
        raise ValueError("Missing required column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["southern"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Fit models
    # -------------------------
    # Keep exact RHS list (12 predictors)
    rhs = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic_flag",
        "other_race",
        "cons_protestant",
        "no_religion",
        "southern",
    ]

    # Diagnostics to help identify why N collapses
    diag = []
    for c in ["dislike_minority_genres", "dislike_other12_genres"] + rhs:
        diag.append(
            {
                "variable": c,
                "nonmissing_n": int(df[c].notna().sum()),
                "mean": float(to_num(df[c]).mean(skipna=True)) if df[c].notna().any() else np.nan,
                "std": float(to_num(df[c]).std(skipna=True, ddof=1)) if df[c].notna().sum() > 1 else np.nan,
            }
        )
    diag_df = pd.DataFrame(diag)
    diag_df.to_csv("./output/Table2_diagnostics.csv", index=False)
    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write(diag_df.to_string(index=False))

    resA, tabA, fitA = fit_ols_with_standardized_betas(
        df_model=df,
        y_col="dislike_minority_genres",
        x_cols=rhs,
        model_name="Table2_ModelA_dislike_minority6",
    )
    resB, tabB, fitB = fit_ols_with_standardized_betas(
        df_model=df,
        y_col="dislike_other12_genres",
        x_cols=rhs,
        model_name="Table2_ModelB_dislike_other12",
    )

    # Paper-style output: standardized betas + stars (no SEs)
    paperA = tabA[["beta_std_with_stars"]].rename(columns={"beta_std_with_stars": "ModelA_beta"})
    paperB = tabB[["beta_std_with_stars"]].rename(columns={"beta_std_with_stars": "ModelB_beta"})
    paper_tbl = paperA.join(paperB, how="outer")
    paper_tbl.index.name = "variable"
    with open("./output/Table2_paper_style_betas.txt", "w", encoding="utf-8") as f:
        f.write("Standardized OLS coefficients (beta) with stars computed from model p-values.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed).\n")
        f.write("Betas computed as b * sd(x)/sd(y) on the estimation sample.\n\n")
        f.write(paper_tbl.to_string())

    # Overview
    overview = pd.concat([fitA, fitB], axis=0, ignore_index=True)
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS)\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write("Model B DV: dislike_other12_genres (12 remaining genres)\n\n")
        f.write(overview.to_string(index=False))
        f.write("\n\nNotes:\n")
        f.write("- Intercept is unstandardized (raw DV units).\n")
        f.write("- Standardized betas computed post-estimation from unstandardized OLS.\n")
        f.write("- Hispanic flag: created as a best-effort proxy from available columns; if a true Hispanic indicator is present, supply it.\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "Paper_style_betas": paper_tbl,
        "Diagnostics": diag_df,
    }