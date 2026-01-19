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

    def clean_missing(series):
        """
        Conservative missing-code cleaning for this extract.
        - Keep legitimate 1-5 Likert, 0-20 educ, etc.
        - Treat common GSS sentinels as missing.
        """
        x = to_num(series).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(series):
        """
        Music taste items: 1-5 scale; dislike is 4/5.
        Returns float with {0,1} and NaN for invalid/missing.
        """
        x = clean_missing(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_missing(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, item_cols, require_all_items=True):
        """
        Sum of dislike indicators across items.
        - If require_all_items=True: listwise across the DV items (min_count=len(items)).
        - Else: sum available items (min_count=1).
        Paper note suggests DK treated as missing and cases excluded; we implement require_all_items=True.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in item_cols], axis=1)
        if require_all_items:
            return mat.sum(axis=1, min_count=len(item_cols))
        return mat.sum(axis=1, min_count=1)

    def standardize_for_beta(series, ddof=0):
        """
        Standardization used for standardized betas: z = (x-mean)/sd computed on estimation sample.
        ddof=0 to match many "population SD" standardizations used for betas.
        """
        x = to_num(series)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        return (x - mu) / sd

    def sig_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df, dv_col, x_cols, model_name):
        """
        Fit:
          - OLS on unstandardized DV to keep an interpretable intercept (as in paper table).
          - Report standardized betas for predictors (X and Y z-scored over estimation sample).
        Implementation:
          - Run OLS on (z_y ~ z_X + const) to get standardized betas directly.
          - Separately run OLS on (y ~ X + const) to get unstandardized intercept (and other b if desired).
        Return:
          - table with standardized beta + stars (and optional p-values from re-estimation),
          - intercept (unstandardized) with stars from unstandardized model,
          - fit (n, r2, adj_r2) from unstandardized model (matches usual reporting with unstd DV).
        """
        needed = [dv_col] + x_cols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_cols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        # Unstandardized model (for intercept and fit stats)
        y = to_num(d[dv_col])
        X = d[x_cols].apply(to_num)
        Xc = sm.add_constant(X, has_constant="add")
        mod_unstd = sm.OLS(y, Xc).fit()

        # Standardized-betas model
        zy = standardize_for_beta(y, ddof=0)
        zX = pd.DataFrame({c: standardize_for_beta(d[c], ddof=0) for c in x_cols}, index=d.index)

        # If any predictor has zero variance after listwise, drop it (but record)
        dropped = [c for c in zX.columns if zX[c].isna().all() or (np.isfinite(zX[c].std(ddof=0)) and zX[c].std(ddof=0) == 0)]
        zX = zX.drop(columns=dropped, errors="ignore")

        if zX.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors dropped (constants/NaNs).")

        zXc = sm.add_constant(zX, has_constant="add")
        mod_std = sm.OLS(zy, zXc).fit()

        # Build "paper-style" table: standardized betas for predictors only; intercept shown separately (unstandardized)
        rows = []
        for c in x_cols:
            if c in dropped or c not in mod_std.params.index:
                beta = np.nan
                p = np.nan
            else:
                beta = float(mod_std.params[c])
                p = float(mod_std.pvalues[c])
            rows.append({"term": c, "beta_std": beta, "stars": sig_stars(p), "p_value_reest": p})

        table = pd.DataFrame(rows).set_index("term")

        intercept_unstd = float(mod_unstd.params.get("const", np.nan))
        intercept_p = float(mod_unstd.pvalues.get("const", np.nan))
        intercept_row = pd.DataFrame(
            {"b_const_unstd": [intercept_unstd], "stars": [sig_stars(intercept_p)], "p_value_reest": [intercept_p]},
            index=["const"],
        )

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "n": int(mod_unstd.nobs),
                "k_predictors": int(mod_unstd.df_model),  # excludes intercept
                "r2": float(mod_unstd.rsquared),
                "adj_r2": float(mod_unstd.rsquared_adj),
                "dropped_predictors_post_listwise": ", ".join(dropped) if dropped else "",
            }]
        )

        # Save human-readable summaries
        with open(f"./output/{model_name}_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(mod_unstd.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        with open(f"./output/{model_name}_standardized_beta_summary.txt", "w", encoding="utf-8") as f:
            f.write(mod_std.summary().as_text())
            f.write("\n\nNOTE: This standardized model is run on z-scored Y and X over the estimation sample.\n")

        # Save paper-style table (standardized betas + stars) and intercept separately
        paper_style = table.copy()
        paper_style["beta_std_fmt"] = paper_style["beta_std"].map(lambda v: "" if pd.isna(v) else f"{v: .3f}")
        paper_style["beta_with_stars"] = paper_style["beta_std_fmt"] + paper_style["stars"]
        paper_style_out = paper_style[["beta_with_stars"]]

        with open(f"./output/{model_name}_Table2_like_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}: Standardized coefficients (betas) for predictors; intercept reported unstandardized.\n")
            f.write("Stars from two-tailed p-values of our re-estimation (* p<.05, ** p<.01, *** p<.001).\n")
            f.write("Table 2 in the paper prints stars but not SEs; p-values shown here are from re-estimation.\n\n")
            f.write("Standardized betas:\n")
            f.write(paper_style_out.to_string())
            f.write("\n\nConstant (unstandardized):\n")
            f.write(intercept_row.to_string())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        return {
            "std_beta_table": table,
            "const_unstd": intercept_row,
            "fit": fit,
            "model_unstd": mod_unstd,
            "model_std": mod_std,
        }

    # ----------------------------
    # Load + filter
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # Dependent variables (counts)
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Require complete responses across the DV's genre set (DK treated as missing; cases excluded)
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_items=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_items=True)

    # ----------------------------
    # Racism score (0-5)
    # ----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Controls (as available in this extract)
    # ----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    df["education_years"] = clean_missing(df["educ"]).where(clean_missing(df["educ"]).between(0, 20))

    # Income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income per capita: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(clean_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    # Female: SEX (1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    df["age_years"] = clean_missing(df["age"]).where(clean_missing(df["age"]).between(18, 89))

    # Race/ethnicity indicators
    # This extract does not contain a direct Hispanic flag. Best-available proxy: ETHNIC==1 (Mexican, American Mexican, Chicano)
    # IMPORTANT: This is a proxy only; it will not identify all Hispanics. We still include it to match Table 2 structure.
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_missing(df["race"]).where(clean_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    if "ethnic" not in df.columns:
        raise ValueError("Missing ethnic column (needed to build a Hispanic proxy).")
    ethnic = clean_missing(df["ethnic"])
    # proxy: mexican-origin in this extract's ETHNIC coding (per sample data: 1 appears)
    df["hispanic"] = np.where(ethnic.isna(), np.nan, (ethnic == 1).astype(float))

    # Religion and denomination
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_missing(df["relig"])
    denom = clean_missing(df["denom"])

    # Conservative Protestant proxy (as in mapping instruction)
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_missing(df["region"]).where(clean_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Fit models (Table 2 style)
    # ----------------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    resA = fit_table2_model(df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6")
    resB = fit_table2_model(df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12")

    # ----------------------------
    # Combined overview
    # ----------------------------
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS with standardized coefficients (betas) for predictors.\n")
        f.write("Intercept is reported from the unstandardized DV model (count outcome), not as a standardized beta.\n")
        f.write("Stars are computed from two-tailed p-values of this re-estimation.\n")
        f.write("NOTE: 'Hispanic' is built as a proxy from ETHNIC==1 in this extract (Mexican origin); it may not match paper coding.\n\n")

        f.write("Model A: DV = count disliked among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(resA["fit"].to_string(index=False))
        f.write("\n\nModel B: DV = count disliked among remaining 12 genres\n")
        f.write(resB["fit"].to_string(index=False))
        f.write("\n")

    return {
        "ModelA_std_beta": resA["std_beta_table"],
        "ModelA_const_unstd": resA["const_unstd"],
        "ModelA_fit": resA["fit"],
        "ModelB_std_beta": resB["std_beta_table"],
        "ModelB_const_unstd": resB["const_unstd"],
        "ModelB_fit": resB["fit"],
    }