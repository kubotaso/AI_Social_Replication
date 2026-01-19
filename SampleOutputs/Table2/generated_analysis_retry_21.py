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
        - Treat common GSS sentinels as missing
        """
        x = to_num(series).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(series):
        """
        Music taste items: 1-5, where 4/5 = dislike, 1/2/3 = not-dislike.
        Anything outside 1-5 or NA-coded is missing.
        """
        x = clean_na(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def bin_from_codes(series, true_codes, false_codes):
        x = clean_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Sum of binary dislike indicators across items.
        Paper summary implies DK treated as missing and missing cases excluded.
        Implement DV as missing unless ALL component items are observed.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def standardized_ols(df, dv, xcols, model_name):
        """
        Standardized OLS coefficients:
        - listwise delete on dv + xcols
        - z-score y and ALL predictors (including dummies)
        - run OLS with intercept
        Output:
        - "paper_style" table: term, beta_std, sig (stars from our p-values; not from paper)
        - "replication_full" table: beta_std, std_err, t, p_value (for transparency; not in paper)
        - fit stats: N, R2, Adj R2
        """
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
        if d.shape[0] < len(xcols) + 5:
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = zscore(d[dv])
        X = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)

        # Drop any predictors that became all-NaN or zero-variance after z-scoring
        keep = []
        dropped = []
        for c in X.columns:
            if X[c].notna().all() and np.isfinite(X[c].std(ddof=0)) and X[c].std(ddof=0) > 0:
                keep.append(c)
            else:
                dropped.append(c)
        X = X[keep]

        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors dropped (constants or missing). Dropped={dropped}")

        ok = y.notna() & np.isfinite(y)
        y = y.loc[ok]
        X = X.loc[ok]
        if X.shape[0] < X.shape[1] + 5:
            raise ValueError(f"{model_name}: not enough cases after standardization cleaning (n={X.shape[0]}, k={X.shape[1]}).")

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        def stars(p):
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        full = pd.DataFrame(
            {
                "term": model.params.index,
                "coef_beta_std": model.params.values,
                "std_err": model.bse.values,
                "t": model.tvalues.values,
                "p_value_replication": model.pvalues.values,
            }
        )
        full["sig_replication"] = full["p_value_replication"].apply(stars)

        # Paper-style table: standardized betas + stars only
        paper = full[["term", "coef_beta_std", "sig_replication"]].copy()
        paper = paper.rename(columns={"coef_beta_std": "beta_std"})

        # Put intercept last (paper often reports Constant at bottom)
        if "const" in paper["term"].values:
            paper_nonconst = paper.loc[paper["term"] != "const"].copy()
            paper_const = paper.loc[paper["term"] == "const"].copy()
            paper = pd.concat([paper_nonconst, paper_const], axis=0, ignore_index=True)

            full_nonconst = full.loc[full["term"] != "const"].copy()
            full_const = full.loc[full["term"] == "const"].copy()
            full = pd.concat([full_nonconst, full_const], axis=0, ignore_index=True)

        fit = {
            "model": model_name,
            "n": int(round(model.nobs)),
            "k_including_intercept": int(model.df_model + 1),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "dropped_predictors_post_standardization": ", ".join(dropped) if dropped else "",
        }

        return model, paper, full, pd.DataFrame([fit])

    # -----------------------------
    # Load and filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

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

    rac1 = bin_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to majority-black school
    rac2 = bin_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = bin_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = bin_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = bin_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_na(df["educ"]).where(clean_na(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # HH income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = bin_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = clean_na(df["age"]).where(clean_na(df["age"]).between(18, 89))

    # Race indicators (reference is White)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables per mapping instruction.
    # Keep as missing so user can see it cannot be estimated from this extract.
    df["hispanic"] = np.nan

    # Conservative Protestant (proxy per instruction): RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Model spec (Table 2 RHS)
    # NOTE: Hispanic cannot be included/estimated from this extract (all-missing),
    # but we keep it in the requested Table-2 order for reporting; it will be dropped.
    # -----------------------------
    x_order_table2 = [
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

    # For estimation, drop predictors that are entirely missing in this dataset (e.g., hispanic).
    # This avoids collapsing N to zero while preserving transparency in the output.
    xcols_estimable = []
    dropped_all_missing = []
    for c in x_order_table2:
        if c not in df.columns:
            raise ValueError(f"Constructed predictor missing unexpectedly: {c}")
        if df[c].notna().sum() == 0:
            dropped_all_missing.append(c)
        else:
            xcols_estimable.append(c)

    # -----------------------------
    # Fit models
    # -----------------------------
    results = {}

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def run_one(dv, model_name):
        model, paper, full, fit = standardized_ols(df, dv, xcols_estimable, model_name)

        # Pretty reorder in Table-2 order (excluding non-estimable ones, then const last)
        order = [c for c in x_order_table2 if c in paper["term"].values]
        if "const" in paper["term"].values:
            order = order + ["const"]
        paper = paper.set_index("term").loc[order].reset_index()

        order_full = [c for c in x_order_table2 if c in full["term"].values]
        if "const" in full["term"].values:
            order_full = order_full + ["const"]
        full = full.set_index("term").loc[order_full].reset_index()

        # Save outputs
        save_text(f"./output/{model_name}_statsmodels_summary.txt", model.summary().as_text())

        # Paper-style table: standardized betas + stars only
        paper_txt = paper.to_string(index=False, float_format=lambda x: f"{x: .3f}")
        header = (
            f"{model_name}\n"
            f"Standardized OLS coefficients (betas) with replication-derived stars.\n"
            f"Note: Table 2 in the paper does not report SEs; stars here come from our re-estimation.\n"
        )
        if dropped_all_missing:
            header += f"Dropped (all-missing in provided extract, cannot estimate): {', '.join(dropped_all_missing)}\n"
        if fit.loc[0, "dropped_predictors_post_standardization"]:
            header += f"Dropped after standardization (constant/invalid): {fit.loc[0, 'dropped_predictors_post_standardization']}\n"
        header += "\n"
        save_text(f"./output/{model_name}_paper_style_table.txt", header + paper_txt + "\n\n" + fit.to_string(index=False))

        # Full replication table (not in paper)
        full_txt = full.to_string(index=False, float_format=lambda x: f"{x: .6f}")
        save_text(f"./output/{model_name}_replication_full_table.txt", full_txt + "\n\n" + fit.to_string(index=False))

        return paper, full, fit

    paperA, fullA, fitA = run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    paperB, fullB, fitB = run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    # Overview
    overview_lines = []
    overview_lines.append("Table 2 replication attempt (1993 GSS extract)")
    overview_lines.append("DVs:")
    overview_lines.append(" - Model A: count of disliked genres among {rap, reggae, blues, jazz, gospel, latin}")
    overview_lines.append(" - Model B: count of disliked genres among other 12 genres in the battery")
    overview_lines.append("")
    overview_lines.append("Key implementation notes:")
    overview_lines.append(" - Year restricted to YEAR==1993.")
    overview_lines.append(" - 'Dislike' coded as 4/5 on 1-5 scale; 1/2/3 coded 0; DK/refused treated as missing.")
    overview_lines.append(" - DV counts require all component items observed (complete-case within DV).")
    overview_lines.append(" - Racism score = sum of 5 dichotomous items (0-5), complete-case on the 5 items.")
    overview_lines.append(" - Standardized coefficients computed by z-scoring y and all x over the estimation sample.")
    overview_lines.append("")
    if dropped_all_missing:
        overview_lines.append("Variables required by the paper but not available/estimable in provided extract:")
        overview_lines.append(f" - {', '.join(dropped_all_missing)}")
        overview_lines.append("")
    overview_lines.append("Fit stats:")
    overview_lines.append(fitA.to_string(index=False))
    overview_lines.append("")
    overview_lines.append(fitB.to_string(index=False))
    save_text("./output/Table2_overview.txt", "\n".join(overview_lines) + "\n")

    results["ModelA_paper_style"] = paperA
    results["ModelB_paper_style"] = paperB
    results["ModelA_replication_full"] = fullA
    results["ModelB_replication_full"] = fullB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    return results