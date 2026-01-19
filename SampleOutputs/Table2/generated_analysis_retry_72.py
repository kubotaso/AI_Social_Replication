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
        Conservative GSS missing-code handling.
        - Treat common sentinel codes as missing: 8/9, 98/99, 998/999, 9998/9999, etc.
        - Also treat extremely large values (>= 9e6) as missing (rare).
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999, 99998, 99999, 999998, 999999}
        x = x.mask(x.isin(list(sentinels)))
        x = x.mask(x >= 9_000_000)
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5; 4/5 => dislike indicator = 1; 1/2/3 => 0.
        Anything else => missing.
        """
        x = clean_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_na_codes(x)
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

    def build_count_complete_case(df, items):
        """
        Count dislikes across items.
        "Don't know" treated as missing at item level; DV requires complete data across all items.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
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

    def fit_standardized_ols(df, dv, xcols, model_name):
        """
        Standardized betas via regression on z-scored DV and z-scored Xs.
        Intercept is NOT standardized; we additionally fit an unstandardized OLS to report constant.
        Both models use the same estimation sample (listwise across DV + Xs).
        """
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: too few complete cases (n={d.shape[0]}, k={len(xcols)}).")

        # Standardized regression for betas and p-values
        y_z = zscore(d[dv])
        X_z = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)

        # Drop any columns that became all-missing (shouldn't happen unless constant)
        bad = [c for c in X_z.columns if X_z[c].isna().any() or (X_z[c].std(ddof=0) == 0)]
        if bad:
            # If any predictor is constant on the estimation sample, drop it but record it
            X_z = X_z.drop(columns=bad)

        Xz_c = sm.add_constant(X_z, has_constant="add")
        m_std = sm.OLS(y_z, Xz_c).fit()

        # Unstandardized regression for intercept and R^2
        X_un = sm.add_constant(d[xcols], has_constant="add")
        m_un = sm.OLS(d[dv], X_un).fit()

        # Build table in paper-like order; note: intercept reported from unstandardized model
        rows = []
        for v in xcols:
            if v in m_std.params.index:
                beta = float(m_std.params[v])
                p = float(m_std.pvalues[v])
            else:
                beta = np.nan
                p = np.nan
            rows.append((v, beta, p))

        out = pd.DataFrame(rows, columns=["variable", "beta_std", "p_value"])
        out["stars"] = out["p_value"].apply(stars_from_p)
        out["beta_std_star"] = out["beta_std"].map(lambda x: ("" if pd.isna(x) else f"{x:.3f}")) + out["stars"]

        # Append intercept (unstandardized)
        intercept = float(m_un.params["const"])
        intercept_p = float(m_un.pvalues["const"])
        intercept_stars = stars_from_p(intercept_p)
        const_row = pd.DataFrame(
            {
                "variable": ["Constant"],
                "beta_std": [np.nan],
                "p_value": [intercept_p],
                "stars": [intercept_stars],
                "beta_std_star": [f"{intercept:.3f}{intercept_stars}"],
            }
        )
        out = pd.concat([out, const_row], ignore_index=True)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m_un.nobs),
                    "k_predictors": int(len(xcols)),
                    "r2": float(m_un.rsquared),
                    "adj_r2": float(m_un.rsquared_adj),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write("Unstandardized OLS (used for Constant and R^2):\n")
            f.write(m_un.summary().as_text())
            f.write("\n\nStandardized OLS (z-scored DV and predictors; coefficients are standardized betas):\n")
            f.write(m_std.summary().as_text())
            if bad:
                f.write("\n\nDropped predictors (constant/invalid after standardization on estimation sample):\n")
                f.write(", ".join(bad) + "\n")

        # Save table (paper-like)
        table_path = f"./output/{model_name}_table.txt"
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("Standardized coefficients (beta) with significance stars based on this replication's p-values.\n")
            f.write("Note: Bryson (1996) Table 2 does not report SEs; p-values are computed from the microdata here.\n\n")
            show = out[["variable", "beta_std", "beta_std_star"]].copy()
            f.write(show.to_string(index=False, justify="left", col_space=2, float_format=lambda x: f"{x:.3f}"))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            f.write("\n")

        return out, fit, m_std, m_un, d.index

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs (Table 2)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dv_dislike_minority6"] = build_count_complete_case(df, minority_items)
    df["dv_dislike_other12"] = build_count_complete_case(df, other12_items)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
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

    # -----------------------------
    # RHS controls
    # -----------------------------
    # Education: EDUC (0-20)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))
    df["education"] = educ

    # Household income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column needed for income per capita.")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige: PRESTG80
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occupational_prestige"] = clean_na_codes(df["prestg80"])

    # Female: SEX (1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age: AGE
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(18, 89))
    df["age"] = age

    # Race dummies from RACE (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: dataset does not include a direct Hispanic identifier.
    # Do not fabricate a proxy from ETHNIC (instruction). Keep as missing and exclude from model.
    df["hispanic"] = np.nan

    # Conservative Protestant: RELIG==1 (Protestant) and DENOM in {1,6,7} (proxy per instruction)
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column needed for religion variables.")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["conservative_protestant"] = consprot

    # No religion: RELIG==4 (None)
    norelig = (relig == 4).astype(float)
    df["no_religion"] = norelig.where(~relig.isna(), np.nan)

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["southern"] = (region == 3).astype(float)
    df.loc[region.isna(), "southern"] = np.nan

    # -----------------------------
    # Models (Table 2): same RHS, two DVs
    # IMPORTANT: Since `hispanic` is not available, we cannot include it without collapsing N to zero.
    # We therefore omit it from estimation, but keep the rest faithful.
    # -----------------------------
    xcols = [
        "racism_score",
        "education",
        "hh_income_per_capita",
        "occupational_prestige",
        "female",
        "age",
        "black",
        "other_race",
        "conservative_protestant",
        "no_religion",
        "southern",
    ]

    # Basic diagnostics file (to help catch unintended sample collapses)
    diag_path = "./output/diagnostics.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("Diagnostics (1993 only)\n")
        f.write(f"Rows after YEAR==1993: {df.shape[0]}\n\n")
        f.write("Missingness rates for DV and predictors (fraction missing):\n")
        for col in ["dv_dislike_minority6", "dv_dislike_other12"] + xcols + ["hispanic"]:
            miss = float(df[col].isna().mean()) if col in df.columns else np.nan
            f.write(f"  {col}: {miss:.3f}\n")

    tabA, fitA, mA_std, mA_un, idxA = fit_standardized_ols(
        df, "dv_dislike_minority6", xcols, "Table2_ModelA_Dislike_Minority6"
    )
    tabB, fitB, mB_std, mB_un, idxB = fit_standardized_ols(
        df, "dv_dislike_other12", xcols, "Table2_ModelB_Dislike_Other12"
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993)\n")
        f.write("Models: OLS with standardized coefficients (betas) computed via regression on z-scored DV and predictors.\n")
        f.write("Intercept (Constant) reported from an unstandardized OLS on the same estimation sample.\n")
        f.write("Note: Hispanic dummy is not included because no direct Hispanic identifier exists in the provided extract.\n\n")
        f.write("Model A DV: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n\nModel B DV: count of dislikes among the other 12 genres\n")
        f.write(fitB.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }