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
        Conservative NA-code cleaning for this extract.
        - Music items: later constrained to 1..5.
        - Dichotomies: later constrained to 1..2.
        - Region/race/sex/relig/denom: later constrained to known sets.
        - Income/prestige/etc.: keep as numeric; drop obvious sentinel codes.
        """
        x = to_num(x).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(item):
        """
        1-5 like/dislike: dislike if 4 or 5; not-dislike if 1/2/3; else missing.
        """
        v = clean_na_codes(item)
        v = v.where(v.between(1, 5))
        out = pd.Series(np.nan, index=v.index, dtype="float64")
        out.loc[v.isin([1, 2, 3])] = 0.0
        out.loc[v.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        v = clean_na_codes(series)
        out = pd.Series(np.nan, index=v.index, dtype="float64")
        out.loc[v.isin(false_codes)] = 0.0
        out.loc[v.isin(true_codes)] = 1.0
        return out

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_dislike_count(df, items, require_all_answered=True):
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        if require_all_answered:
            return mat.sum(axis=1, min_count=len(items))
        else:
            # not used; kept for clarity
            return mat.sum(axis=1, min_count=1)

    def standardized_betas_from_unstd(y, X, params_unstd):
        """
        beta_j = b_j * sd(x_j) / sd(y) for non-constant terms.
        """
        sd_y = y.std(ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            sd_x = X[col].std(ddof=0)
            b = params_unstd.get(col, np.nan)
            if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                betas[col] = np.nan
            else:
                betas[col] = float(b) * float(sd_x) / float(sd_y)
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

    def fit_table2_model(df, dv_col, rhs_cols_ordered, model_name):
        # listwise deletion on exactly DV + RHS
        needed = [dv_col] + rhs_cols_ordered
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # require enough cases
        if d.shape[0] < (len(rhs_cols_ordered) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(rhs_cols_ordered)}).")

        y = to_num(d[dv_col])
        X = pd.DataFrame({c: to_num(d[c]) for c in rhs_cols_ordered}, index=d.index)

        # add constant
        Xc = sm.add_constant(X, has_constant="add")

        # drop any perfectly constant predictors (to avoid NaN/aliasing)
        drop_cols = []
        for c in Xc.columns:
            if c == "const":
                continue
            if Xc[c].nunique(dropna=True) <= 1:
                drop_cols.append(c)
        if drop_cols:
            Xc = Xc.drop(columns=drop_cols)

        # Fit OLS
        model = sm.OLS(y, Xc).fit()

        # standardized betas computed from unstandardized model
        betas = standardized_betas_from_unstd(y, Xc, model.params.to_dict())

        # build paper-ordered table (with explicit variable names; include intercept separately)
        rows = []
        for v in rhs_cols_ordered:
            if v not in Xc.columns:
                rows.append(
                    {
                        "variable": v,
                        "beta": np.nan,
                        "stars": "",
                        "p_value_model": np.nan,
                        "note": "omitted (constant/collinear or all-missing after listwise deletion)",
                    }
                )
            else:
                p = float(model.pvalues.get(v, np.nan))
                rows.append(
                    {
                        "variable": v,
                        "beta": float(betas.get(v, np.nan)),
                        "stars": stars_from_p(p),
                        "p_value_model": p,
                        "note": "",
                    }
                )

        # intercept (unstandardized, as typically reported)
        intercept = float(model.params.get("const", np.nan))
        intercept_p = float(model.pvalues.get("const", np.nan))
        intercept_row = pd.DataFrame(
            [
                {
                    "variable": "constant",
                    "beta": np.nan,
                    "stars": stars_from_p(intercept_p),
                    "p_value_model": intercept_p,
                    "note": f"unstandardized intercept on DV scale; b={intercept:.6f}",
                }
            ]
        )

        table = pd.DataFrame(rows)
        table = pd.concat([table, intercept_row], ignore_index=True)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(model.nobs),
                    "k_including_const": int(len(model.params)),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_predictors": ", ".join(drop_cols) if drop_cols else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE:\n")
            f.write("- Coefficients in the table file are standardized betas (computed from unstandardized OLS via sd ratios).\n")
            f.write("- Stars are based on model p-values (*<.05, **<.01, ***<.001; two-tailed).\n")
            f.write("- Intercept is reported as unstandardized in the note field.\n")
            if drop_cols:
                f.write(f"- Dropped constant/collinear predictors: {', '.join(drop_cols)}\n")

        # paper-style table: variable, beta, stars
        table_out = table.copy()
        def fmt_beta(x):
            return "" if pd.isna(x) else f"{x:.3f}"
        table_out["beta_fmt"] = table_out["beta"].map(fmt_beta)
        table_out["beta_with_stars"] = table_out["beta_fmt"] + table_out["stars"].astype(str)

        # Write table
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv_col}\n\n")
            f.write("Standardized OLS coefficients (beta) with significance stars\n")
            f.write("(Stars from re-estimated model p-values; table in paper reports stars but not SEs)\n\n")
            for _, r in table_out.iterrows():
                var_label = r["variable"]
                beta_star = r["beta_with_stars"]
                note = r["note"]
                if var_label == "constant":
                    f.write(f"{var_label:22s} {beta_star:>10s}   {note}\n")
                else:
                    if note:
                        f.write(f"{var_label:22s} {beta_star:>10s}   {note}\n")
                    else:
                        f.write(f"{var_label:22s} {beta_star:>10s}\n")

            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))

        return model, table, fit

    # -----------------------------
    # Load and preprocess
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # filter 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # Ensure required columns exist (from mapping)
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    base_controls = ["educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region"]

    required = set(minority_items + other12_items + racism_fields + base_controls)
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # DVs: require all components answered (DK treated as missing; listwise on DV components)
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_answered=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_answered=True)

    # Racism score (0-5), require all 5 items answered
    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # Education (years)
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (white reference)
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: attempt to construct from 'ethnic' if present and appears usable.
    # NOTE: If 'ethnic' is not a Hispanic identifier in the user's extract, this will be imperfect,
    # but we include it to avoid the earlier omission/runtime issues.
    if "ethnic" in df.columns:
        eth = clean_na_codes(df["ethnic"])
        # Heuristic: treat codes 1..5 as "Hispanic/Latino origin groups" if present; else leave missing.
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[eth.between(1, 5)] = 1.0
        # if clearly non-hispanic codes exist (>=6), set to 0
        hisp.loc[eth >= 6] = 0.0
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["southern"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Fit models (Table 2 order)
    # -----------------------------
    rhs_order = [
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

    # Write diagnostics
    diag_vars = ["dislike_minority_genres", "dislike_other12_genres"] + rhs_order
    diag = []
    for c in diag_vars:
        v = to_num(df[c]) if c in df.columns else pd.Series(np.nan, index=df.index)
        diag.append(
            {
                "variable": c,
                "nonmissing_n": int(v.notna().sum()),
                "mean": float(v.mean(skipna=True)) if v.notna().any() else np.nan,
                "std": float(v.std(skipna=True, ddof=0)) if v.notna().any() else np.nan,
                "min": float(v.min(skipna=True)) if v.notna().any() else np.nan,
                "max": float(v.max(skipna=True)) if v.notna().any() else np.nan,
            }
        )
    diag_df = pd.DataFrame(diag)
    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("Diagnostics (1993 only): non-missing and basic moments\n\n")
        f.write(diag_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Fit each model
    modelA, tableA, fitA = fit_table2_model(
        df=df,
        dv_col="dislike_minority_genres",
        rhs_cols_ordered=rhs_order,
        model_name="Table2_ModelA_Dislike_Minority_Associated6",
    )
    modelB, tableB, fitB = fit_table2_model(
        df=df,
        dv_col="dislike_other12_genres",
        rhs_cols_ordered=rhs_order,
        model_name="Table2_ModelB_Dislike_Other12_Remaining",
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (GSS 1993): OLS with standardized coefficients (beta)\n")
        f.write("Betas computed from unstandardized OLS: beta_j = b_j * sd(x_j) / sd(y)\n")
        f.write("Stars computed from model p-values: *<.05, **<.01, ***<.001 (two-tailed)\n\n")
        f.write("Model A DV: count of disliked among {Rap, Reggae, Blues, Jazz, Gospel, Latin}\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: count of disliked among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nSee individual model table files for coefficients in paper order.\n")

    return {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "Diagnostics": diag_df,
    }