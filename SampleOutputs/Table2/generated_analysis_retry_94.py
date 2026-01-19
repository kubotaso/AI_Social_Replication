def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative NA-code cleaner for this extract:
        - Coerce to numeric
        - Treat common GSS NA codes as missing (8/9, 98/99, 998/999, 9998/9999, etc.)
        NOTE: This is intentionally conservative; do not wipe legitimate high values like REALINC.
        """
        x = to_num(x).copy()
        na_codes = {8, 9, 98, 99, 998, 999, 9998, 9999, 99998, 99999}
        return x.mask(x.isin(list(na_codes)))

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 scale; dislike if 4 or 5.
        1-3 => 0, 4-5 => 1, else missing.
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

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_completecase(df, items):
        """
        Count of disliked genres across items.
        Paper notes DK treated as missing and missing cases excluded; implement strict complete-case
        across component items for each DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        # Require all items observed
        return mat.sum(axis=1, min_count=len(items))

    def stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardized_beta_table(df_model, dv, xcols, model_name):
        """
        Standardized OLS coefficients (beta):
        - Standardize DV and continuous predictors.
        - Keep 0/1 dummies unstandardized? The paper says standardized OLS coefficients; to be faithful
          mechanically, standardize all RHS columns (including dummies) and DV, then OLS.
        - Intercept is estimated on the original DV scale by a second OLS (unstandardized) so we can
          report a constant comparable to the paper's constant row.
        """
        needed = [dv] + xcols
        d = df_model[needed].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}).")

        # Standardized regression for betas
        y_z = zscore(d[dv])
        X_z = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)

        # If any predictor becomes all-NaN after z-scoring, that's a pipeline error for Table 2 variables
        bad = [c for c in X_z.columns if not X_z[c].notna().all()]
        if bad:
            raise ValueError(
                f"{model_name}: predictors became undefined after standardization (likely zero variance): {bad}"
            )

        Xz_const = sm.add_constant(X_z, has_constant="add")
        m_z = sm.OLS(y_z, Xz_const).fit()

        # Unstandardized regression for intercept/fit on DV scale (still same sample)
        X_u = sm.add_constant(d[xcols], has_constant="add")
        m_u = sm.OLS(d[dv], X_u).fit()

        # Build Table-2-like output: standardized betas + stars; constant from unstandardized model
        rows = []
        for term in xcols:
            beta = float(m_z.params.get(term, np.nan))
            p = float(m_z.pvalues.get(term, np.nan))
            rows.append({"term": term, "beta": beta, "stars": stars_from_p(p)})

        out = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m_u.nobs),
                    "r2": float(m_u.rsquared),
                    "adj_r2": float(m_u.rsquared_adj),
                    "constant": float(m_u.params.get("const", np.nan)),
                }
            ]
        )

        # Save human-readable text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write("Unstandardized OLS (for intercept and fit on DV scale)\n")
            f.write(m_u.summary().as_text())
            f.write("\n\nStandardized OLS (z-scored DV and predictors; coefficients are betas)\n")
            f.write(m_z.summary().as_text())
            f.write("\n\nTable-style betas (standardized) + stars computed from standardized model p-values:\n")
            f.write(out.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
            f.write("\n\nFit (from unstandardized model):\n")
            f.write(fit.to_string(index=False))

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(out.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
            f.write("\n")

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))
            f.write("\n")

        return out, fit, m_z, m_u

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Year filter
    if "year" not in df.columns:
        raise ValueError("Missing YEAR column.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # DVs
    # -------------------------
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

    # -------------------------
    # Racism score (0-5)
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
    # Controls (as available in this extract)
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column for income per capita.")
    realinc = to_num(df["realinc"])  # do NOT blanket-mask 99999 etc; REALINC is continuous
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (white omitted)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: not available in provided extract -> cannot construct faithfully
    # Keep as missing and exclude from model specification (otherwise listwise deletion would destroy N).
    df["hispanic"] = np.nan

    # Conservative Protestant proxy from RELIG and DENOM (best-effort given provided variables)
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column for religion coding.")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    df["cons_protestant"] = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    df.loc[relig.isna() | denom.isna(), "cons_protestant"] = np.nan

    # No religion
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # South
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -------------------------
    # Models (Table 2 RHS as implementable here)
    # Note: Hispanic dummy cannot be included because it is not present in the provided data.
    # -------------------------
    xcols = [
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

    # Basic sanity: prevent silent dropping of key table variables (e.g., no_religion)
    for c in xcols:
        if c not in df.columns:
            raise ValueError(f"Constructed predictor missing: {c}")

    # Run both models
    tabA, fitA, mA_z, mA_u = standardized_beta_table(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    tabB, fitB, mB_z, mB_u = standardized_beta_table(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("1993 GSS Table 2-style replication\n")
        f.write("Standardized betas are from OLS on z-scored DV and z-scored predictors.\n")
        f.write("Constant/R2/AdjR2 are from unstandardized OLS on DV scale.\n")
        f.write("Stars computed from p-values of standardized model: *<.05, **<.01, ***<.001 (two-tailed).\n")
        f.write("Note: Hispanic dummy cannot be included because no Hispanic identifier exists in provided extract.\n\n")
        f.write("Model A: DV = count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin (0-6)\n")
        f.write(tabA.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
        f.write("\n\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write("Model B: DV = count of dislikes among the other 12 genres (0-12)\n")
        f.write(tabB.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
        f.write("\n\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }