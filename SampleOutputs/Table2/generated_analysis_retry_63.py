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

    def clean_na_codes_generic(x):
        """
        Conservative GSS missing-code cleaning:
        - Coerce to numeric
        - Treat common GSS-style sentinels as missing
        - Additionally treat very large "all 9s" codes as missing (>= 90 for small-coded items)
        """
        x = to_num(x).copy()
        # Common sentinels in extracts
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        # For many categorical items, 0 or negative can be "inapplicable"
        x = x.mask(x <= 0)
        return x

    def likert_dislike_indicator(x):
        """
        Music liking items: valid responses 1-5.
        Dislike is 4 or 5; Like/neutral is 1-3.
        Missing otherwise (including DK/NA codes).
        """
        x = clean_na_codes_generic(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_recode(x, true_codes, false_codes):
        x = clean_na_codes_generic(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_complete=True):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_complete:
            # Mirror "DK treated as missing and missing cases excluded": require all components observed
            return mat.sum(axis=1, min_count=len(items))
        # (not used here)
        return mat.sum(axis=1, min_count=1)

    def standardize(series, ddof=0):
        s = to_num(series)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_table2_model(df, dv_col, x_cols, model_name):
        """
        Fit OLS on *unstandardized* variables, then compute standardized betas post-hoc:
            beta_j = b_j * sd(x_j) / sd(y)
        This keeps the intercept on the DV scale (as in the paper's "Constant" row).
        Stars come from the OLS p-values (computed here from the same fitted model).
        """
        needed = [dv_col] + x_cols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < len(x_cols) + 5:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # Check for zero variance predictors in this analytic sample; drop them rather than error
        dropped = []
        keep_x = []
        for c in x_cols:
            v = d[c]
            if v.nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                keep_x.append(c)

        if len(keep_x) == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance after listwise deletion.")

        y = to_num(d[dv_col])
        X = pd.DataFrame({c: to_num(d[c]) for c in keep_x}, index=d.index)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # Post-hoc standardized betas for slopes only
        y_sd = y.std(ddof=0)
        betas = {}
        for c in ["const"] + keep_x:
            if c == "const":
                betas[c] = np.nan
            else:
                x_sd = X[c].std(ddof=0)
                betas[c] = (model.params[c] * x_sd / y_sd) if (np.isfinite(y_sd) and y_sd != 0 and np.isfinite(x_sd) and x_sd != 0) else np.nan

        def stars(p):
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        # Table in paper-like order: standardized betas for predictors; intercept shown separately (unstandardized)
        out_rows = []
        for c in keep_x:
            out_rows.append(
                {
                    "variable": c,
                    "beta_std": float(betas.get(c, np.nan)),
                    "p_value": float(model.pvalues.get(c, np.nan)),
                    "stars": stars(float(model.pvalues.get(c, np.nan))) if np.isfinite(model.pvalues.get(c, np.nan)) else "",
                }
            )

        paper_like = pd.DataFrame(out_rows).set_index("variable")

        intercept_row = pd.DataFrame(
            {
                "b_unstd": [float(model.params["const"])],
                "p_value": [float(model.pvalues["const"])],
                "stars": [stars(float(model.pvalues["const"]))],
            },
            index=["const"],
        )

        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std_posthoc": pd.Series(betas),
            }
        )

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        return model, paper_like, intercept_row, full, fit, d.index, dropped

    # -----------------------------
    # Load data and basic filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain columns YEAR and ID (case-insensitive).")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Dependent variables
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    # Require complete responses for each DV (paper: DK treated as missing; cases excluded)
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_complete=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_complete=True)

    # -----------------------------
    # Racism score (0-5 additive)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism component column: {c}")

    rac1 = binary_recode(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_recode(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_recode(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_recode(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_recode(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_na_codes_generic(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing REALINC and/or HOMPOP columns.")
    realinc = clean_na_codes_generic(df["realinc"])
    hompop = clean_na_codes_generic(df["hompop"]).where(lambda s: s > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_na_codes_generic(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_recode(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_na_codes_generic(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: Black, Other race (White is reference)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_na_codes_generic(df["race"]).where(lambda s: s.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: no direct mapping available in provided columns -> cannot construct faithfully.
    # To keep the model runnable and explicit, create missing; the model fitter will drop if zero-variance/NA-only.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy using RELIG and DENOM (best effort per mapping instruction)
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing RELIG and/or DENOM columns.")
    relig = clean_na_codes_generic(df["relig"])
    denom = clean_na_codes_generic(df["denom"])
    consprot = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )
    df["cons_protestant"] = consprot

    # No religion dummy
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_na_codes_generic(df["region"]).where(lambda s: s.isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Fit both models (same RHS)
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

    # Save diagnostics about RHS variation before modeling
    diag_rows = []
    for c in x_order + ["dislike_minority_genres", "dislike_other12_genres"]:
        if c not in df.columns:
            continue
        s = df[c]
        diag_rows.append(
            {
                "variable": c,
                "n_nonmissing": int(pd.Series(s).notna().sum()),
                "n_unique_nonmissing": int(pd.Series(s).dropna().nunique()),
                "mean": float(pd.Series(s).mean(skipna=True)) if pd.Series(s).notna().any() else np.nan,
                "std": float(pd.Series(s).std(skipna=True, ddof=0)) if pd.Series(s).notna().any() else np.nan,
                "min": float(pd.Series(s).min(skipna=True)) if pd.Series(s).notna().any() else np.nan,
                "max": float(pd.Series(s).max(skipna=True)) if pd.Series(s).notna().any() else np.nan,
            }
        )
    diagnostics = pd.DataFrame(diag_rows).set_index("variable")
    diagnostics.to_csv("./output/Table2_diagnostics.csv")

    mA, paperA, interceptA, fullA, fitA, idxA, droppedA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, interceptB, fullB, fitB, idxB, droppedB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # -----------------------------
    # Save human-readable outputs
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # Model summaries
    write_text("./output/Table2_ModelA_summary.txt", mA.summary().as_text())
    write_text("./output/Table2_ModelB_summary.txt", mB.summary().as_text())

    # Paper-like (standardized betas + stars; intercept separate)
    def format_paper_like(paper_df, intercept_df, fit_df, dv_label, dropped):
        lines = []
        lines.append(dv_label)
        lines.append("")
        lines.append("Standardized coefficients (beta) for predictors; stars from this model's p-values.")
        if dropped:
            lines.append(f"Dropped due to zero variance in analytic sample: {', '.join(dropped)}")
        lines.append("")
        tmp = paper_df.copy()
        tmp["beta_std"] = tmp["beta_std"].map(lambda x: f"{x: .3f}" if pd.notna(x) else "")
        tmp["p_value"] = tmp["p_value"].map(lambda x: f"{x: .6f}" if pd.notna(x) else "")
        lines.append(tmp[["beta_std", "stars", "p_value"]].to_string())
        lines.append("")
        lines.append("Constant (unstandardized):")
        ic = intercept_df.copy()
        ic["b_unstd"] = ic["b_unstd"].map(lambda x: f"{x: .3f}" if pd.notna(x) else "")
        ic["p_value"] = ic["p_value"].map(lambda x: f"{x: .6f}" if pd.notna(x) else "")
        lines.append(ic.to_string())
        lines.append("")
        lines.append("Fit:")
        lines.append(fit_df.to_string(index=False))
        lines.append("")
        return "\n".join(lines)

    write_text(
        "./output/Table2_ModelA_paper_like.txt",
        format_paper_like(
            paperA, interceptA, fitA,
            "Model A DV: Count of disliked minority-associated genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)",
            droppedA
        ),
    )
    write_text(
        "./output/Table2_ModelB_paper_like.txt",
        format_paper_like(
            paperB, interceptB, fitB,
            "Model B DV: Count of disliked 'other 12' genres",
            droppedB
        ),
    )

    # Full regression tables (explicitly labeled as computed here)
    fullA.to_csv("./output/Table2_ModelA_full_table.csv")
    fullB.to_csv("./output/Table2_ModelB_full_table.csv")

    # Also save as readable fixed-width text
    write_text("./output/Table2_ModelA_full_table.txt", fullA.to_string(float_format=lambda x: f"{x: .6f}"))
    write_text("./output/Table2_ModelB_full_table.txt", fullB.to_string(float_format=lambda x: f"{x: .6f}"))

    # Overview
    overview = []
    overview.append("Table 2 replication (computed from provided 1993 GSS extract).")
    overview.append("Notes:")
    overview.append("- Standardized betas are computed post-hoc from unstandardized OLS: beta = b * sd(x) / sd(y).")
    overview.append("- Intercept is unstandardized (on DV scale).")
    overview.append("- Stars are computed from this fitted model's p-values (paper reports stars but not SEs).")
    overview.append("- If a predictor has zero variance in the model's analytic sample, it is dropped and noted.")
    overview.append("")
    overview.append("Model A fit:")
    overview.append(fitA.to_string(index=False))
    overview.append("")
    overview.append("Model B fit:")
    overview.append(fitB.to_string(index=False))
    overview.append("")
    write_text("./output/Table2_overview.txt", "\n".join(overview))

    return {
        "ModelA_paper_like": paperA,
        "ModelA_intercept": interceptA,
        "ModelA_full": fullA,
        "ModelA_fit": fitA,
        "ModelB_paper_like": paperB,
        "ModelB_intercept": interceptB,
        "ModelB_full": fullB,
        "ModelB_fit": fitB,
        "Diagnostics": diagnostics,
    }