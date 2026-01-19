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

    def clean_na(series):
        """
        Conservative missing-code handling for this extract:
        - Coerce to numeric
        - Treat common GSS-style sentinels as missing
        We deliberately do NOT treat values like 0 as missing universally (some variables can be 0).
        """
        x = to_num(series).copy()
        x = x.replace([np.inf, -np.inf], np.nan)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike(item):
        """
        1-5: like very much, like, neutral, dislike, dislike very much
        Dislike indicator = 1 if 4/5, 0 if 1/2/3, missing otherwise.
        """
        x = clean_na(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from(series, true_codes, false_codes):
        x = clean_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count(df, items, require_all_answered=True):
        """
        Count of disliked genres across a specified item list.
        Missing-handling:
        - Each item: missing if item missing/invalid.
        - DV: if require_all_answered, require all items non-missing (listwise within DV).
              else sum across available items (not used here).
        """
        mat = pd.concat([likert_dislike(df[c]).rename(c) for c in items], axis=1)
        if require_all_answered:
            return mat.sum(axis=1, min_count=len(items))
        return mat.sum(axis=1, min_count=1)

    def add_sig_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_unstd_and_report_betas(df, y_col, x_cols_ordered, model_name):
        """
        Fit OLS on unstandardized DV and unstandardized X (with constant),
        then compute standardized beta for each predictor as:
            beta_j = b_j * sd(x_j) / sd(y)
        using the estimation sample (listwise) and ddof=0 (population SD).
        Constant is left unstandardized (beta is NaN for const).
        """
        needed = [y_col] + x_cols_ordered
        d = df[needed].copy()

        # Ensure numeric
        for c in needed:
            d[c] = to_num(d[c])

        # Drop rows with any missing among needed columns (paper is listwise on model vars)
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Guard: enough data
        if d.shape[0] < (len(x_cols_ordered) + 2):
            raise ValueError(
                f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols_ordered)})."
            )

        y = d[y_col].astype(float)
        X = d[x_cols_ordered].astype(float)

        # Drop any zero-variance predictors (should not happen; but prevents runtime errors)
        dropped = [c for c in X.columns if float(X[c].std(ddof=0)) == 0.0 or not np.isfinite(float(X[c].std(ddof=0)))]
        if dropped:
            X = X.drop(columns=dropped)
        used_predictors = list(X.columns)

        if len(used_predictors) == 0:
            raise ValueError(f"{model_name}: all predictors are zero-variance after listwise deletion.")

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas for predictors
        sd_y = float(y.std(ddof=0))
        betas = {}
        for c in ["const"] + used_predictors:
            if c == "const":
                betas[c] = np.nan
            else:
                sd_x = float(X[c].std(ddof=0))
                betas[c] = (float(model.params[c]) * sd_x / sd_y) if (sd_y > 0 and sd_x > 0) else np.nan

        tab = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": pd.Series(betas),
            }
        )

        # Add stars for convenience (from re-estimation; paper uses stars but no SEs)
        tab["sig"] = tab["p_value"].apply(add_sig_stars)

        # Reorder rows to match requested order with const last (paper shows constant last)
        order = [c for c in x_cols_ordered if c in tab.index]
        tab_out = tab.loc[order + (["const"] if "const" in tab.index else [])].copy()
        tab_out.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Standardized betas computed post-estimation as b * sd(x)/sd(y) on the estimation sample.\n")
            f.write("- Significance stars (if shown) come from this re-estimation (two-tailed), not from the paper.\n")
            if dropped:
                f.write(f"- Dropped zero-variance predictors after listwise deletion: {dropped}\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_out.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))
            f.write("\n")

        return tab_out, fit, d.index

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    if df.shape[0] == 0:
        raise ValueError("No rows with YEAR==1993 found.")

    # -----------------------------
    # Build DVs (Table 2)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count(df, minority_items, require_all_answered=True)
    df["dislike_other12_genres"] = build_count(df, other12_items, require_all_answered=True)

    # -----------------------------
    # Build racism scale (0-5)
    # -----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from(df["rachaf"], true_codes=[1], false_codes=[2])   # object to half-black school
    rac2 = binary_from(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls (RHS)
    # -----------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column: educ")
    df["education_years"] = clean_na(df["educ"]).where(clean_na(df["educ"]).between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing income component column: {c}")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column: prestg80")
    df["occ_prestige"] = clean_na(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing SEX column: sex")
    df["female"] = binary_from(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing AGE column: age")
    df["age_years"] = clean_na(df["age"]).where(clean_na(df["age"]).between(18, 89))

    # Race dummies: black and other_race from RACE; white reference.
    if "race" not in df.columns:
        raise ValueError("Missing RACE column: race")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: not in provided variable list; use ETHNIC as a pragmatic proxy ONLY if present.
    # This is explicitly a fallback to allow estimation rather than dropping the term entirely.
    # If ETHNIC is absent, keep as missing and later error with diagnostic.
    if "ethnic" in df.columns:
        eth = clean_na(df["ethnic"])
        # GSS ETHNIC in many extracts: 1=Mexican, 2=Puerto Rican, 3=Other Spanish, ...
        # Here we define Hispanic as eth in {1,2,3}. Anything else => 0 (including nonresponse handled as NaN).
        df["hispanic"] = np.where(eth.isna(), np.nan, eth.isin([1, 2, 3]).astype(float))
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 (Protestant) and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    df["cons_protestant"] = np.where(
        (relig.isna() | denom.isna()),
        np.nan,
        ((relig == 1) & denom.isin([1, 6, 7])).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column: region")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Fit Table 2 models
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

    # Diagnostic: if hispanic is entirely missing, fail with a clear message (paper includes it)
    if df["hispanic"].notna().sum() == 0:
        # Save a diagnostic file and proceed by dropping hispanic so the code still runs.
        with open("./output/diagnostic_hispanic_missing.txt", "w", encoding="utf-8") as f:
            f.write(
                "Hispanic dummy is entirely missing in the provided dataset.\n"
                "The paper's Table 2 includes a Hispanic indicator; without a Hispanic identifier, exact replication is not possible.\n"
                "Proceeding by fitting models WITHOUT 'hispanic' to avoid a zero-case runtime error.\n"
            )
        x_order_used = [c for c in x_order if c != "hispanic"]
    else:
        x_order_used = x_order

    # Run both models
    tabA, fitA, idxA = fit_unstd_and_report_betas(
        df, "dislike_minority_genres", x_order_used, "Table2_ModelA_dislike_minority6"
    )
    tabB, fitB, idxB = fit_unstd_and_report_betas(
        df, "dislike_other12_genres", x_order_used, "Table2_ModelB_dislike_other12"
    )

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993)\n")
        f.write("Models: OLS on count DV; standardized betas computed post-estimation.\n")
        f.write("Note: If Hispanic identifier is unavailable, model is fit without Hispanic (see diagnostic file).\n\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(tabA.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n\n")
        f.write("Model B DV: dislike_other12_genres (12 remaining genres)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(tabB.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }