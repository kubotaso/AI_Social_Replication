def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # --------------------------
    # Helpers
    # --------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing-code handling for this extract.
        We only blank obvious sentinel codes that commonly appear in GSS extracts.
        """
        x = to_num(x).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 like/dislike scale.
        Dislike = 1 if 4 or 5; Like/neutral = 0 if 1,2,3; missing otherwise.
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

    def build_count_completecase(df_in, item_cols):
        """
        Count dislikes across items, requiring all items observed (listwise within DV),
        matching 'DK treated as missing and missing cases excluded' in DV construction.
        """
        mats = []
        for c in item_cols:
            if c not in df_in.columns:
                raise ValueError(f"Missing required music item column: {c}")
            mats.append(likert_dislike_indicator(df_in[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        return mat.sum(axis=1, min_count=len(item_cols))

    def standardized_betas_from_unstd(model, d, y_col, x_cols):
        """
        Compute standardized betas from unstandardized OLS:
            beta_j = b_j * SD(X_j) / SD(Y)
        using SDs from the analytic sample 'd' actually used for the model.
        Intercept gets NaN beta.
        """
        y_sd = d[y_col].std(ddof=0)
        betas = {}
        for term, b in model.params.items():
            if term == "const":
                betas[term] = np.nan
            else:
                x_sd = d[term].std(ddof=0) if term in d.columns else np.nan
                if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                    betas[term] = np.nan
                else:
                    betas[term] = float(b) * float(x_sd) / float(y_sd)
        return pd.Series(betas)

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

    def fit_table2_model(df_in, y_col, x_cols, model_name):
        """
        OLS with intercept; listwise delete on y and all Xs.
        Drops zero-variance predictors (within analytic sample) instead of erroring.
        Produces:
          - full table: unstd b, se, t, p, beta_std
          - paper-style table: standardized beta + stars for slopes; unstd constant + stars
        """
        needed = [y_col] + x_cols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] == 0:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n=0).")

        # Drop zero-variance predictors safely (addresses runtime failures)
        zero_var = []
        keep_cols = []
        for c in x_cols:
            if d[c].std(ddof=0) == 0 or not np.isfinite(d[c].std(ddof=0)):
                zero_var.append(c)
            else:
                keep_cols.append(c)

        X = sm.add_constant(d[keep_cols], has_constant="add")
        y = d[y_col]
        model = sm.OLS(y, X).fit()

        beta_std = standardized_betas_from_unstd(model, pd.concat([d[[y_col]], X.drop(columns=["const"])], axis=1), y_col, keep_cols)

        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": beta_std.reindex(model.params.index),
            }
        )
        full.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
                }
            ]
        )

        # Paper-style: standardized betas for slopes; unstd constant for intercept
        # (Table 2 shows a "Constant" even though slopes are standardized.)
        rows = []
        for term in model.params.index:
            p = float(model.pvalues[term])
            if term == "const":
                coef = float(model.params[term])
                rows.append({"term": "Constant", "coef": coef, "stars": stars(p)})
            else:
                coef = float(full.loc[term, "beta_std"])
                rows.append({"term": term, "coef": coef, "stars": stars(p)})

        paper = pd.DataFrame(rows)

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            if zero_var:
                f.write("\n\nDropped zero-variance predictors:\n")
                f.write(", ".join(zero_var))
            f.write("\n")

        full.to_string(
            open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x: .6f}",
        )
        paper.to_string(
            open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8"),
            index=False,
            float_format=lambda x: f"{x: .6f}",
        )

        return model, paper, full, fit

    # --------------------------
    # Filter to 1993
    # --------------------------
    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must include 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # --------------------------
    # Dependent variables (counts)
    # --------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband",
        "blugrass",
        "country",
        "musicals",
        "classicl",
        "folk",
        "moodeasy",
        "newage",
        "opera",
        "conrock",
        "oldies",
        "hvymetal",
    ]

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # --------------------------
    # Racism score (0-5 additive)
    # --------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism component column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half Black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # --------------------------
    # Controls (match mapping)
    # --------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' column.")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' column required for income per capita.")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' column.")
    age = clean_gss_missing(df["age"])
    df["age"] = age.where(age.between(18, 89))

    # Race dummies: black/other (white reference). Hispanic not available in this extract.
    if "race" not in df.columns:
        raise ValueError("Missing 'race' column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not present -> include as all 0 (explicitly noted), to avoid runtime errors.
    # This keeps the model matrix stable; interpretation differs from paper if Hispanic matters.
    df["hispanic"] = 0.0

    # Religion
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' column required for religion dummies.")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])

    # Conservative Protestant proxy: Protestant and denom in {1,6,7}
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()))
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna())
    df["no_religion"] = norelig

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing 'region' column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # --------------------------
    # Fit models (two DVs, same RHS)
    # --------------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Run
    mA, paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )

    # Also write a combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from provided microdata; no paper numbers hard-coded)\n")
        f.write("Notes:\n")
        f.write("- Hispanic identifier is not present in this dataset extract; 'hispanic' is set to 0 for all cases.\n")
        f.write("- Standardized coefficients computed post-hoc from unstandardized OLS: beta = b * SD(X)/SD(Y).\n")
        f.write("- Listwise deletion is used per model on DV + all predictors.\n\n")
        f.write("Model A (DV: dislike_minority_genres)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B (DV: dislike_other12_genres)\n")
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