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

    def clean_gss_na(x):
        """
        Conservative NA cleaning for this extract:
        - Coerce to numeric
        - Treat common GSS "NA-like" codes as missing.
        We do NOT drop legitimate 0s unless a variable logically cannot be 0 (handled elsewhere).
        """
        x = to_num(x).copy()
        na_codes = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(na_codes))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5; dislike defined as 4/5.
        DK/NA/refused etc -> missing.
        """
        x = clean_gss_na(item)
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

    def build_count_complete_case(df, items):
        """
        Count of dislikes across `items`, requiring all items observed
        (DK treated as missing; cases with missing excluded in DV construction).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardized_betas_from_unstd(model, y, X_no_const):
        """
        Compute standardized betas from an OLS model estimated on original scales:
        beta_j = b_j * sd(x_j) / sd(y)
        """
        y_sd = np.std(y, ddof=0)
        betas = {}
        if not np.isfinite(y_sd) or y_sd == 0:
            for c in X_no_const.columns:
                betas[c] = np.nan
            return pd.Series(betas)

        for c in X_no_const.columns:
            x_sd = np.std(X_no_const[c], ddof=0)
            b = model.params.get(c, np.nan)
            if not np.isfinite(x_sd) or x_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (x_sd / y_sd)
        return pd.Series(betas)

    def star_from_p(p):
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
        OLS on original scales (so constant is interpretable in DV units),
        then compute standardized betas (Table 2-style).
        Listwise deletion on DV + RHS vars.
        """
        needed = [dv_col] + x_cols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        if d.shape[0] < (len(x_cols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = d[dv_col].astype(float)
        X = d[x_cols].astype(float)

        # Drop any zero-variance predictors (should not happen if coded right; but keep robust)
        zero_var = [c for c in X.columns if np.nanstd(X[c], ddof=0) == 0]
        if zero_var:
            X = X.drop(columns=zero_var)

        # Add constant
        Xc = sm.add_constant(X, has_constant="add")

        # Fit
        model = sm.OLS(y, Xc).fit()

        # Standardized betas computed post-estimation
        betas = standardized_betas_from_unstd(model, y.values, X)

        # Assemble Table 2-like output
        rows = []
        for c in X.columns:
            rows.append(
                {
                    "term": c,
                    "beta_std": float(betas.get(c, np.nan)),
                    "p_value": float(model.pvalues.get(c, np.nan)),
                    "sig": star_from_p(model.pvalues.get(c, np.nan)),
                }
            )

        # Constant (unstandardized)
        rows.append(
            {
                "term": "const",
                "beta_std": np.nan,
                "p_value": float(model.pvalues.get("const", np.nan)),
                "sig": star_from_p(model.pvalues.get("const", np.nan)),
            }
        )

        tab = pd.DataFrame(rows).set_index("term")

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),  # incl intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Standardized coefficients (beta) computed post-estimation: beta = b * sd(x)/sd(y)\n")
            f.write("- Stars computed from two-tailed OLS p-values (*<.05, **<.01, ***<.001)\n")
            if zero_var:
                f.write(f"- Dropped zero-variance predictors: {', '.join(zero_var)}\n")

        # A compact table that resembles the paper: beta + stars (no SE)
        tab_out = tab.copy()
        tab_out["beta_std_star"] = tab_out["beta_std"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + tab_out["sig"]
        tab_out = tab_out[["beta_std", "sig", "beta_std_star", "p_value"]]

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv_col}\n\n")
            f.write(tab_out.to_string(float_format=lambda x: f"{x:.6f}"))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))

        return model, tab_out, fit, d.index

    # ----------------------------
    # Load / filter
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required minimal columns
    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # Construct DVs (exact genre sets)
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # ----------------------------
    # Racism score (0-5 additive; complete-case on 5 items)
    # ----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of will

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Controls
    # ----------------------------
    # Education years (0-20)
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    edu = clean_gss_na(df["educ"]).where(clean_gss_na(df["educ"]).between(0, 20))
    df["education_years"] = edu

    # Income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(clean_gss_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige PRESTG80
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    # Female: SEX (1=male, 2=female)
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age (18-89)
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_gss_na(df["age"]).where(clean_gss_na(df["age"]).between(18, 89))
    df["age_years"] = age

    # Race indicators from RACE (1=white, 2=black, 3=other)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_gss_na(df["race"]).where(clean_gss_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not available in provided variables -> exclude from models (cannot replicate directly)
    # Keep as column for transparency; do NOT use as proxy.
    df["hispanic"] = np.nan

    # Religion and denomination
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion field: {c}")
    relig = clean_gss_na(df["relig"])
    denom = clean_gss_na(df["denom"])

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna(), np.nan)
    df["no_religion"] = norelig

    # Southern: REGION==3 (GSS regions 1-4)
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_gss_na(df["region"]).where(clean_gss_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Fit Table 2 models
    # ----------------------------
    # Because Hispanic is unavailable in this extract, we omit it (cannot be estimated).
    # We keep the rest aligned with Table 2.
    x_cols = [
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

    # Validate predictors exist
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    modelA, tabA, fitA, idxA = fit_table2_model(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    modelB, tabB, fitB, idxB = fit_table2_model(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )

    # Write a combined overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (GSS 1993)\n")
        f.write("OLS estimated on original DV scale; standardized betas computed post-estimation.\n")
        f.write("IMPORTANT: Hispanic dummy is not available in the provided extract and is omitted.\n\n")

        f.write("Model A DV: Dislike count of Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin (0-6)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nStandardized betas (with stars):\n")
        f.write(tabA[["beta_std_star"]].to_string())
        f.write("\n\n")

        f.write("Model B DV: Dislike count of the 12 remaining genres (0-12)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nStandardized betas (with stars):\n")
        f.write(tabB[["beta_std_star"]].to_string())
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }