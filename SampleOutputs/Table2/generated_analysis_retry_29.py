def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_na(x):
        """
        Conservative missing-code handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        """
        s = to_num(x).copy()
        s = s.mask(s.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return s

    def likert_dislike_indicator(series):
        """
        1-5 liking scale: 4/5 -> dislike indicator 1, 1/2/3 -> 0, else missing.
        """
        x = clean_gss_na(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_gss_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all_answered=False):
        """
        Sum of item-level dislike indicators.
        - require_all_answered=False: counts across answered items (DK treated as missing at item level);
          respondent can contribute even if some items missing.
        - require_all_answered=True: require complete responses to all items to compute count.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_all_answered:
            return mat.sum(axis=1, min_count=len(items))
        return mat.sum(axis=1, min_count=1)

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def fit_unstd_and_betas(df, y_col, x_cols, model_name):
        """
        Fit unstandardized OLS for y ~ X + const (to keep intercept meaningful),
        then compute standardized betas for each predictor:
            beta_j = b_j * sd(x_j) / sd(y)
        computed on the estimation sample used in the fit (listwise).
        """
        needed = [y_col] + x_cols
        d = df[needed].copy()

        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(x_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = to_num(d[y_col]).astype(float)
        X = d[x_cols].apply(to_num).astype(float)

        # Drop zero-variance predictors (after listwise deletion) to prevent singularities/NaN betas
        dropped_zero_var = []
        keep_cols = []
        for c in X.columns:
            v = X[c].values
            if np.nanstd(v, ddof=0) == 0:
                dropped_zero_var.append(c)
            else:
                keep_cols.append(c)
        X = X[keep_cols]

        if X.shape[1] == 0:
            raise ValueError(
                f"{model_name}: all predictors have zero variance after listwise deletion. "
                f"Dropped: {dropped_zero_var}"
            )

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (slopes only; constant gets NaN)
        sd_y = float(np.nanstd(y.values, ddof=0))
        betas = {}
        for term in model.params.index:
            if term == "const":
                betas[term] = np.nan
            else:
                sd_x = float(np.nanstd(X[term].values, ddof=0))
                betas[term] = float(model.params[term]) * (sd_x / sd_y) if sd_y > 0 and sd_x > 0 else np.nan

        tab = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": pd.Series(betas),
            }
        )

        fit = {
            "model": model_name,
            "n": int(model.nobs),
            "k_predictors": int(model.df_model),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "dropped_zero_variance_predictors": ", ".join(dropped_zero_var) if dropped_zero_var else "",
        }
        return model, tab, fit, d.index, keep_cols

    def add_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    # -------------------------
    # Load and filter to 1993
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.shape[0] == 0:
        raise ValueError("No rows found for YEAR == 1993.")

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
            raise ValueError(f"Missing music taste item column: {c}")

    # Use "counts across answered items" to avoid throwing away half the sample.
    # Then do listwise deletion for the regression itself, as in typical Table 2 replication.
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_answered=False)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_answered=False)

    # -------------------------
    # Racism score (0-5 additive)
    # -------------------------
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

    # -------------------------
    # Covariates
    # -------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    df["education_years"] = clean_gss_na(df["educ"]).where(clean_gss_na(df["educ"]).between(0, 20))

    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing columns needed for per-capita income: realinc, hompop")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(clean_gss_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = clean_gss_na(df["age"]).where(clean_gss_na(df["age"]).between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_na(df["race"]).where(clean_gss_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: not directly present; use ETHNIC as a pragmatic proxy only when clearly Hispanic-coded.
    # Many GSS extracts use ETHNIC=1 for Hispanic (varies by extract). We implement:
    # hispanic=1 if ethnic==1; 0 if ethnic in other valid positive codes; missing if ethnic missing.
    # If your extract uses a different coding, adjust this block.
    if "ethnic" in df.columns:
        eth = clean_gss_na(df["ethnic"])
        # treat any positive code as "known"; code 1 as Hispanic
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))
    else:
        df["hispanic"] = np.nan  # will be dropped by listwise deletion if included

    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing columns needed for religion dummies: relig, denom")
    relig = clean_gss_na(df["relig"])
    denom = clean_gss_na(df["denom"])

    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_na(df["region"]).where(clean_gss_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Fit Table 2 models
    # -------------------------
    x_order = [
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
    for c in x_order:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    def run_one(y_col, model_name):
        model, tab, fit, used_idx, kept_cols = fit_unstd_and_betas(df, y_col, x_order, model_name)

        # Reindex to desired display order (const last)
        display_terms = [c for c in x_order if c in kept_cols] + ["const"]
        tab = tab.reindex(display_terms)

        # Paper-style table: standardized betas + stars
        paper = pd.DataFrame(index=tab.index)
        paper["beta_std"] = tab["beta_std"]
        paper["sig"] = tab["p_value"].apply(add_stars)
        paper["beta_with_sig"] = paper["beta_std"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + paper["sig"]

        # Full table (for diagnostics)
        full = tab.copy()

        # Write files
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")
            f.write("\n\nKept predictors (post listwise + zero-var drop):\n")
            f.write(", ".join(kept_cols) + "\n")

        paper_out = paper.copy()
        paper_out.index.name = "term"
        paper_out.to_string(
            open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x:.6f}",
        )

        full_out = full.copy()
        full_out.index.name = "term"
        full_out.to_string(
            open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x:.6f}",
        )

        return paper_out, full_out, pd.DataFrame([fit])

    paperA, fullA, fitA = run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    paperB, fullB, fitB = run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    # Overall overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (GSS 1993) - OLS with standardized coefficients (betas)\n")
        f.write("Betas computed as: beta_j = b_j * sd(x_j) / sd(y), using the estimation sample.\n")
        f.write("DVs are counts of disliked genres (4/5 on 1-5 scale). Item-level DK treated as missing.\n")
        f.write("Model estimation uses listwise deletion on DV and all RHS variables.\n\n")
        f.write("Model A: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B: dislike_other12_genres (12 remaining genres)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table_paper_style": paperA,
        "ModelB_table_paper_style": paperB,
        "ModelA_table_full": fullA,
        "ModelB_table_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }