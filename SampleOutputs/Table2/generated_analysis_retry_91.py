def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(s):
        """
        Conservative NA-code cleaning for this extract:
        - explicit common NA codes (8/9, 98/99, 998/999, 9998/9999)
        Does NOT blanket-drop zeros because some items can validly be 0/1 codes.
        """
        x = to_num(s).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items: valid 1-5. Dislike if 4 or 5.
        Missing if not in 1..5 (after NA-code cleanup).
        """
        x = clean_na_codes(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_na_codes(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_sample(s):
        """
        Z-score with sample SD (ddof=1). Return NaN series if SD is 0 or undefined.
        """
        x = to_num(s)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        return (x - mu) / sd

    def build_count_dv(df_in, items, dv_name):
        """
        DV is a count of disliked genres. To match typical index construction in this paper:
        - Treat DK/NA as missing at the item level
        - Require ALL items observed to compute the index (listwise within the DV items),
          then the model applies listwise deletion across DV + RHS.
        """
        mats = []
        for c in items:
            if c not in df_in.columns:
                raise ValueError(f"Missing required music item column: {c}")
            mats.append(likert_dislike_indicator(df_in[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        dv = mat.sum(axis=1, min_count=len(items))
        dv.name = dv_name
        return dv

    def compute_standardized_betas_from_unstd(unstd_model, y_raw, X_raw, term_order):
        """
        Standardized beta_j = b_j * sd(x_j) / sd(y), using the estimation sample.
        Works for continuous and 0/1 dummies (as in standardized coefficient tables).
        """
        y_sd = y_raw.std(ddof=1)
        betas = {}
        for t in term_order:
            if t == "const":
                continue
            if t in unstd_model.params.index and t in X_raw.columns:
                x_sd = X_raw[t].std(ddof=1)
                if np.isfinite(x_sd) and x_sd > 0 and np.isfinite(y_sd) and y_sd > 0:
                    betas[t] = float(unstd_model.params[t] * (x_sd / y_sd))
                else:
                    betas[t] = np.nan
            else:
                betas[t] = np.nan
        return betas

    def stars_from_p(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_model(df_in, dv_col, rhs_cols, model_name, variable_order):
        """
        Fit OLS on raw scale (for intercept comparable to paper) and compute standardized betas
        from unstandardized coefficients using SD ratios on the estimation sample.
        """
        needed = [dv_col] + rhs_cols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(rhs_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(rhs_cols)}).")

        y = d[dv_col].astype(float)
        X = d[rhs_cols].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (exclude intercept)
        term_order = ["const"] + variable_order
        betas = compute_standardized_betas_from_unstd(model, y, X, term_order)

        rows = []
        # Intercept row (unstandardized; paper-style constant)
        rows.append(
            {
                "variable": "constant",
                "b_unstd": float(model.params.get("const", np.nan)),
                "beta_std": np.nan,
                "std_err": float(model.bse.get("const", np.nan)),
                "t": float(model.tvalues.get("const", np.nan)),
                "p_value": float(model.pvalues.get("const", np.nan)),
                "stars": stars_from_p(float(model.pvalues.get("const", np.nan))),
            }
        )

        for v in variable_order:
            rows.append(
                {
                    "variable": v,
                    "b_unstd": float(model.params.get(v, np.nan)),
                    "beta_std": float(betas.get(v, np.nan)),
                    "std_err": float(model.bse.get(v, np.nan)),
                    "t": float(model.tvalues.get(v, np.nan)),
                    "p_value": float(model.pvalues.get(v, np.nan)),
                    "stars": stars_from_p(float(model.pvalues.get(v, np.nan))),
                }
            )

        table = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTES:\n")
            f.write("- Standardized betas are computed from the unstandardized OLS fit via beta_j = b_j * sd(x_j)/sd(y).\n")
            f.write("- Intercept is reported on the DV's original (count) scale.\n")
            f.write("- Stars are computed from this model's two-tailed p-values (*<.05, **<.01, ***<.001).\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(table.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        return model, table, fit

    # -----------------------------
    # Filter to 1993
    # -----------------------------
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs (counts)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal",
    ]

    df["dislike_minority_genres"] = build_count_dv(df, minority_items, "dislike_minority_genres")
    df["dislike_other12_genres"] = build_count_dv(df, other12_items, "dislike_other12_genres")

    # -----------------------------
    # Racism score (0-5; all five components required)
    # -----------------------------
    needed_rac = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in needed_rac:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing required column: educ")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing required column: prestg80")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing required column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing required column: age")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: use White (non-Hispanic) as reference; create Black, Hispanic, Other (mutually exclusive)
    # Hispanic proxy: NOT available in provided variables per instruction. To avoid distortions:
    # - Create explicit hispanic column but do NOT force it into listwise deletion if it's missing.
    # Here we implement Hispanic as missing (unknown) and EXCLUDE it from models if it is all-missing.
    if "race" not in df.columns:
        raise ValueError("Missing required column: race")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic not present in the provided extract; keep as all-missing.
    df["hispanic"] = np.nan

    # Religion dummies
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])

    # Conservative Protestant proxy (as specified in mapping instruction)
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # Southern
    if "region" not in df.columns:
        raise ValueError("Missing required column: region")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["southern"] = (region == 3).astype(float)
    df.loc[region.isna(), "southern"] = np.nan

    # -----------------------------
    # RHS list in paper order
    # (Include Hispanic only if it has any non-missing variation in this dataset.)
    # -----------------------------
    rhs_base = [
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
        "southern",
    ]

    # If Hispanic is all-missing (as in provided extract), drop it to avoid n=0.
    if df["hispanic"].notna().sum() == 0:
        rhs_cols = [c for c in rhs_base if c != "hispanic"]
        variable_order = [c for c in rhs_base if c != "hispanic"]  # for table order
        hispanic_note = "Hispanic variable is not available in the provided dataset extract; excluded from estimation."
    else:
        rhs_cols = rhs_base
        variable_order = rhs_base
        hispanic_note = "Hispanic variable included."

    # -----------------------------
    # Fit models
    # -----------------------------
    modelA, tableA, fitA = fit_model(
        df_in=df,
        dv_col="dislike_minority_genres",
        rhs_cols=rhs_cols,
        model_name="Table2_ModelA_dislike_minority6",
        variable_order=variable_order,
    )
    modelB, tableB, fitB = fit_model(
        df_in=df,
        dv_col="dislike_other12_genres",
        rhs_cols=rhs_cols,
        model_name="Table2_ModelB_dislike_other12",
        variable_order=variable_order,
    )

    # -----------------------------
    # Overview + diagnostics
    # -----------------------------
    diag_cols = ["dislike_minority_genres", "dislike_other12_genres"] + rhs_cols
    diag = []
    for c in diag_cols:
        diag.append(
            {
                "variable": c,
                "nonmissing_n": int(df[c].notna().sum()),
                "mean": float(df[c].mean(skipna=True)) if df[c].notna().any() else np.nan,
                "std": float(df[c].std(skipna=True, ddof=1)) if df[c].notna().sum() > 1 else np.nan,
                "min": float(df[c].min(skipna=True)) if df[c].notna().any() else np.nan,
                "max": float(df[c].max(skipna=True)) if df[c].notna().any() else np.nan,
            }
        )
    diag_df = pd.DataFrame(diag)

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from microdata; no numbers copied from paper)\n")
        f.write("GSS 1993 only.\n\n")
        f.write("Dependent variables:\n")
        f.write("- Model A: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin (0-6; complete across these 6 items)\n")
        f.write("- Model B: count of dislikes among the other 12 genres listed in the extract (0-12; complete across these 12 items)\n\n")
        f.write("Racism score: sum of 5 dichotomies (0-5; complete across all 5 items)\n\n")
        f.write("Important note:\n")
        f.write(f"- {hispanic_note}\n\n")
        f.write("Fit:\n")
        f.write(fitA.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n\n")
        f.write(fitB.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n\nDiagnostics (non-missing, mean, sd, min, max):\n")
        f.write(diag_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    # Save diagnostics table
    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write(diag_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    return {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "Diagnostics": diag_df,
    }