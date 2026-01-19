def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative GSS NA-code handling across this extract:
        - Treat common sentinel codes as missing.
        - Keep real values otherwise.
        """
        x = to_num(x).copy()
        # Common GSS-style sentinels (varies by item); keep conservative.
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999, -1, -2, -3, -4, -5}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 scale where 4/5 indicate dislike.
        Return: 1 dislike, 0 otherwise, NaN if missing/invalid.
        """
        x = clean_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Sum of dislike indicators across items; require all items observed (complete-case),
        matching "DK treated as missing; missing cases excluded" for index construction.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def wmean(x, w):
        x = np.asarray(x, dtype="float64")
        w = np.asarray(w, dtype="float64")
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return (x[m] * w[m]).sum() / w[m].sum()

    def wvar(x, w):
        x = np.asarray(x, dtype="float64")
        w = np.asarray(w, dtype="float64")
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() < 2:
            return np.nan
        mu = (x[m] * w[m]).sum() / w[m].sum()
        return (w[m] * (x[m] - mu) ** 2).sum() / w[m].sum()

    def wstd(x, w):
        v = wvar(x, w)
        if not np.isfinite(v) or v <= 0:
            return np.nan
        return np.sqrt(v)

    def standardize_series(s, w=None):
        s = to_num(s)
        if w is None:
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True, ddof=0)
        else:
            mu = wmean(s.values, w.values)
            sd = wstd(s.values, w.values)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_table2_model(df, dv, x_order, model_name, w_col=None):
        """
        - Listwise deletion on dv + all predictors
        - OLS (optionally WLS if w_col provided and valid)
        - Compute standardized coefficients (beta) using SDs from estimation sample
          (weighted SDs if WLS).
        - Output:
          * paper_style table: beta with stars
          * full table: b_unstd + beta + p_value (SEs are replication-only; table2 doesn't show them)
          * fit stats
        """
        cols = [dv] + x_order + ([w_col] if w_col else [])
        d = df[cols].copy()

        # Clean inf
        d = d.replace([np.inf, -np.inf], np.nan)

        # If weights provided, require positive finite weights; otherwise drop weight column
        weights = None
        if w_col is not None:
            d[w_col] = clean_na_codes(d[w_col])
            d.loc[~np.isfinite(d[w_col]) | (d[w_col] <= 0), w_col] = np.nan

        # Drop missing rows listwise
        d = d.dropna(axis=0, how="any")

        # Ensure we have enough cases
        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_order)}).")

        # Check predictor variance (do not silently drop; fail fast with useful info)
        zero_var = []
        for c in x_order:
            if d[c].nunique(dropna=True) <= 1:
                zero_var.append(c)
        if zero_var:
            # Save diagnostics
            diag_path = f"./output/{model_name}_diagnostics_zero_variance.txt"
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write(f"{model_name}: one or more predictors have zero variance after listwise deletion:\n")
                for c in zero_var:
                    f.write(f"- {c}: unique values = {sorted(d[c].dropna().unique().tolist())}\n")
                f.write("\nThis indicates a coding/filtering issue; fix before comparing to the paper.\n")
            raise ValueError(f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}")

        # Prepare y and X
        y = to_num(d[dv]).astype(float)
        X = d[x_order].apply(to_num).astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        # Fit OLS or WLS
        if w_col is not None:
            weights = d[w_col].astype(float)
            model = sm.WLS(y, Xc, weights=weights).fit()
        else:
            model = sm.OLS(y, Xc).fit()

        # Standardized betas: beta_j = b_j * sd(x_j) / sd(y)
        # Use estimation-sample SDs; weighted if weights provided.
        if weights is None:
            sd_y = y.std(ddof=0)
        else:
            sd_y = wstd(y.values, weights.values)

        beta = pd.Series(index=model.params.index, dtype="float64")
        beta.loc["const"] = np.nan

        for c in x_order:
            if weights is None:
                sd_x = X[c].std(ddof=0)
            else:
                sd_x = wstd(X[c].values, weights.values)
            if np.isfinite(sd_x) and np.isfinite(sd_y) and sd_x > 0 and sd_y > 0:
                beta.loc[c] = model.params.loc[c] * (sd_x / sd_y)
            else:
                beta.loc[c] = np.nan

        # Stars based on replication p-values (Table 2 itself doesn't provide SEs/p)
        def stars(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        # Build output tables in the paper's order
        order_with_const = ["const"] + x_order
        full = pd.DataFrame(
            {
                "b_unstd": model.params.reindex(order_with_const),
                "beta_std": beta.reindex(order_with_const),
                "p_value": model.pvalues.reindex(order_with_const),
            }
        )
        paper_style = pd.DataFrame(
            {
                "beta": full["beta_std"],
                "stars": full["p_value"].apply(stars),
            }
        )
        paper_style["beta_with_stars"] = paper_style["beta"].map(lambda v: np.nan if pd.isna(v) else float(v))
        paper_style["beta_with_stars"] = paper_style.apply(
            lambda r: ("" if pd.isna(r["beta"]) else f"{r['beta']:.3f}{r['stars']}"), axis=1
        )

        fit = {
            "model": model_name,
            "dv": dv,
            "n": int(model.nobs),
            "k_predictors": int(model.df_model),  # excludes intercept
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "weights_used": bool(weights is not None),
        }

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Table 2 in the paper reports standardized coefficients only; SEs are not shown in the paper.\n")
            f.write("Stars here are computed from replication-model p-values.\n\n")
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")

        with open(f"./output/{model_name}_paper_style.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name} (paper-style): standardized betas with stars from replication p-values\n\n")
            f.write(paper_style[["beta_with_stars"]].to_string())
            f.write("\n")

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name} (replication full): unstandardized b, standardized beta, p-value\n\n")
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        # Also return objects
        paper_style_out = paper_style[["beta", "stars", "beta_with_stars"]].copy()
        paper_style_out.index.name = "term"
        full_out = full.copy()
        full_out.index.name = "term"
        fit_out = pd.DataFrame([fit])

        return model, paper_style_out, full_out, fit_out

    # -----------------------------
    # Load data and filter to 1993
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct dependent variables
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------------
    # Construct racism score (0-5)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny edu chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    df["education_years"] = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))

    # Household income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column for income per capita: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    df["age_years"] = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(18, 89))

    # Race and ethnicity: use what's available; construct a Hispanic proxy from ETHNIC if present.
    # NOTE: The mapping instruction warns ETHNIC isn't a direct Hispanic identifier; however Table 2
    # requires a Hispanic dummy. Here we implement a transparent, minimal proxy:
    #   hispanic = 1 if ethnic in [20..29] (codes often used for Hispanic origins in many GSS extracts),
    #   else 0, missing if ETHNIC missing.
    # If this proxy doesn't match your extract's coding, replace the codes accordingly.
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    if "ethnic" in df.columns:
        eth = clean_na_codes(df["ethnic"])
        # Conservative proxy band; adjust if your ETHNIC scheme differs.
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[eth.notna()] = 0.0
        hisp.loc[eth.between(20, 29)] = 1.0
        df["hispanic"] = hisp
    else:
        # If not available, create missing; model will then drop all rows -> fail fast in variance check.
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing relig/denom columns.")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    m = relig.notna() & denom.notna()
    consprot.loc[m] = ((relig.loc[m] == 1) & (denom.loc[m].isin([1, 6, 7]))).astype(float)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = pd.Series(np.nan, index=df.index, dtype="float64")
    norelig.loc[relig.notna()] = (relig.loc[relig.notna()] == 4).astype(float)
    df["no_religion"] = norelig

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    south = pd.Series(np.nan, index=df.index, dtype="float64")
    south.loc[region.notna()] = (region.loc[region.notna()] == 3).astype(float)
    df["south"] = south

    # -----------------------------
    # Fit the two Table 2 models
    # -----------------------------
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
            raise ValueError(f"Constructed predictor missing: {c}")

    mA, paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6", w_col=None
    )
    mB, paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12", w_col=None
    )

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson Table 2 replication attempt (computed from microdata)\n")
        f.write("Outputs:\n")
        f.write("- *_paper_style.txt: standardized coefficients (beta) with stars from replication p-values\n")
        f.write("- *_full_table.txt: unstandardized b, standardized beta, p-values (SEs not shown because Table 2 doesn't report them)\n")
        f.write("\nIMPORTANT:\n")
        f.write("Table 2 in the paper reports standardized coefficients only; it does not report SE/t/p.\n")
        f.write("If coefficients/stars do not match, the likely causes are sample/module differences, weighting, or coding differences.\n\n")

        f.write("Model A DV: dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin (count)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(paperA[["beta_with_stars"]].to_string())
        f.write("\n\n")

        f.write("Model B DV: dislike of the 12 remaining genres (count)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(paperB[["beta_with_stars"]].to_string())
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }