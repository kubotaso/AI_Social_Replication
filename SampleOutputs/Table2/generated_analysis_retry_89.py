def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Helpers
    # -----------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing handling:
        - convert to numeric
        - treat common GSS sentinel codes as missing
        - keep other numeric values
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items on 1-5:
        1 like very much
        2 like
        3 neither
        4 dislike
        5 dislike very much
        Dislike = {4,5}; Not-dislike = {1,2,3}; else missing.
        """
        x = clean_gss_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all_answered=True):
        inds = []
        for c in items:
            inds.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(inds, axis=1)
        if require_all_answered:
            # Bryson: DK treated as missing; cases with missing excluded => complete-case on items
            return mat.sum(axis=1, min_count=len(items))
        # (Not used here, but left for clarity)
        return mat.sum(axis=1, min_count=1)

    def zscore_series(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def standardized_betas_via_posthoc(y_raw, X_raw, model_fit):
        """
        Compute standardized betas as: beta_j = b_j * sd(x_j) / sd(y)
        using the estimation sample (rows used in the fit).
        Intercept has no standardized beta (NaN).
        """
        y_sd = y_raw.std(ddof=0)
        betas = {}
        for col in X_raw.columns:
            if col == "const":
                betas[col] = np.nan
            else:
                x_sd = X_raw[col].std(ddof=0)
                b = model_fit.params.get(col, np.nan)
                if not np.isfinite(y_sd) or y_sd == 0 or not np.isfinite(x_sd) or x_sd == 0:
                    betas[col] = np.nan
                else:
                    betas[col] = float(b) * float(x_sd) / float(y_sd)
        return pd.Series(betas)

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

    def fit_model(df, dv_col, rhs_cols_order, model_name):
        """
        OLS on raw DV (so intercept is in DV units).
        Report standardized betas (post-hoc) + stars based on model p-values.
        """
        needed = [dv_col] + rhs_cols_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(rhs_cols_order) + 5):
            raise ValueError(
                f"{model_name}: not enough complete cases after listwise deletion "
                f"(n={d.shape[0]}, predictors={len(rhs_cols_order)})."
            )

        y = to_num(d[dv_col]).astype(float)
        X = d[rhs_cols_order].apply(to_num).astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        fit = sm.OLS(y, Xc).fit()

        # standardized betas
        beta_std = standardized_betas_via_posthoc(y_raw=y, X_raw=Xc, model_fit=fit)

        # Build "paper-style" table: standardized betas + stars (no SE column)
        # (Stars computed from our fitted model; Table 2 has no SEs/p-values.)
        rows = []
        order_with_const = ["const"] + rhs_cols_order
        for term in order_with_const:
            b = fit.params.get(term, np.nan)
            p = fit.pvalues.get(term, np.nan)
            beta = beta_std.get(term, np.nan)
            rows.append(
                {
                    "term": term,
                    "beta_std": beta,
                    "beta_std_star": ("" if pd.isna(beta) else f"{beta:.3f}{add_stars(p)}"),
                    "b_unstd": b,
                    "p_value_model": p,
                }
            )
        table = pd.DataFrame(rows).set_index("term")

        fit_stats = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(fit.nobs),
                    "k_including_const": int(fit.df_model + 1),
                    "r2": float(fit.rsquared),
                    "adj_r2": float(fit.rsquared_adj),
                    "const_unstd": float(fit.params.get("const", np.nan)),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n\nNOTE: The published Table 2 reports standardized coefficients and stars, not SEs.\n")
            f.write("Stars here are computed from this fitted microdata model (two-tailed).\n")

        # Save "paper-style" table (standardized betas + stars)
        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Standardized OLS coefficients (beta) with significance stars (computed from fitted model)\n")
            f.write("Table 2 in the paper does not report SEs; SEs/p-values are not taken from the paper.\n\n")
            out = table[["beta_std", "beta_std_star"]].copy()
            f.write(out.to_string(float_format=lambda x: f"{x: .6f}"))

        # Save full technical table (includes p-values for transparency, but clearly not from the paper)
        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write("Full fitted-model table (for transparency; not reported in the paper's Table 2)\n\n")
            f.write(table.to_string(float_format=lambda x: f"{x: .6f}"))

        return fit, table, fit_stats

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Missing required column: YEAR/year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Dependent variables
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Complete-case on the items for each DV (as described in the feedback)
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_answered=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_answered=True)

    # -----------------------
    # Racism score (0-5)
    # -----------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half Black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls (as available in this extract)
    # -----------------------
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC/educ.")
    educ = clean_gss_missing(df["educ"]).where(lambda s: s.between(0, 20))
    df["education_years"] = educ

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required income component: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(lambda s: s > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80/prestg80.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing SEX/sex.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing AGE/age.")
    df["age_years"] = clean_gss_missing(df["age"]).where(lambda s: s.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing RACE/race.")
    race = clean_gss_missing(df["race"]).where(lambda s: s.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not present in provided variables under the mapping instruction.
    # To keep the models runnable and faithful to the available extract, we omit hispanic.
    # (We DO NOT proxy using 'ethnic'.)
    # If a hispanic indicator exists in other extracts, add it here and include in RHS.
    hispanic_available = "hispanic" in df.columns

    if "relig" not in df.columns:
        raise ValueError("Missing RELIG/relig.")
    relig = clean_gss_missing(df["relig"])

    if "denom" not in df.columns:
        raise ValueError("Missing DENOM/denom.")
    denom = clean_gss_missing(df["denom"])

    # Conservative Protestant proxy from mapping instruction
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna(), np.nan)
    df["no_religion"] = norelig

    if "region" not in df.columns:
        raise ValueError("Missing REGION/region.")
    region = clean_gss_missing(df["region"]).where(lambda s: s.isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Fit Table 2 models
    # -----------------------
    rhs = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]
    if hispanic_available:
        # If present, include (but in this provided extract it typically isn't)
        rhs = rhs[:8] + ["hispanic"] + rhs[8:]

    for c in rhs:
        if c not in df.columns:
            raise ValueError(f"Missing RHS variable after construction: {c}")

    modelA, tableA, fitA = fit_model(
        df=df,
        dv_col="dislike_minority_genres",
        rhs_cols_order=rhs,
        model_name="Table2_ModelA_dislike_minority6",
    )
    modelB, tableB, fitB = fit_model(
        df=df,
        dv_col="dislike_other12_genres",
        rhs_cols_order=rhs,
        model_name="Table2_ModelB_dislike_other12",
    )

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993)\n")
        f.write("OLS on raw dislike-count DV; standardized betas computed post-hoc as b*sd(x)/sd(y).\n")
        f.write("Note: The paper's Table 2 does not report SEs; stars shown here come from fitted-model p-values.\n")
        if not hispanic_available:
            f.write("Important: 'Hispanic' indicator is not available in this provided extract, so it is omitted.\n")
        f.write("\nModel A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres (12 remaining genres)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results = {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }
    return results