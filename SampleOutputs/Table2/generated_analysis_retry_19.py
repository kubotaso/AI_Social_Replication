def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_na(series):
        """
        Conservative missing-code handling for this extract:
        - set common sentinel codes to NaN
        - do NOT drop legitimate small integers (e.g., RELIG categories)
        """
        x = to_num(series).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(item_series):
        """
        Music items: 1-5; dislike if 4/5; like/neutral if 1/2/3.
        Anything else -> missing.
        """
        x = clean_gss_na(item_series)
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

    def build_dislike_count_allow_partial(df, items, min_answered):
        """
        Count of disliked genres across 'items', allowing partial missingness.
        - Each item contributes 0/1 if answered, else missing.
        - DV is computed as the sum if respondent answered at least min_answered items.
        This is the key lever to avoid collapsing N too much.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        answered = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, skipna=True)
        count = count.where(answered >= min_answered)
        return count

    def zscore_sample(s):
        """
        Sample z-score (ddof=1) computed on estimation sample.
        """
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def standardized_betas_from_ols(y, X_with_const):
        """
        Compute standardized coefficients (beta weights) from an OLS fit on unstandardized variables:
            beta_j = b_j * sd(x_j) / sd(y)
        Intercept has no standardized beta (NaN).
        """
        sd_y = y.std(ddof=1)
        betas = {}
        for col in X_with_const.columns:
            if col == "const":
                betas[col] = np.nan
            else:
                sd_x = X_with_const[col].std(ddof=1)
                b = float(X_with_const.attrs["params"][col])
                if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                    betas[col] = np.nan
                else:
                    betas[col] = b * (sd_x / sd_y)
        return pd.Series(betas)

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

    def fit_model(df_in, dv_col, xcols, model_name):
        """
        OLS with intercept. Standardized betas computed post-hoc.
        Listwise deletion over dv and xcols ONLY.
        """
        d = df_in[[dv_col] + xcols].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        y = to_num(d[dv_col])
        X = d[xcols].apply(to_num)

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Attach params so standardized_betas_from_ols can access them without refitting
        Xc.attrs["params"] = model.params.to_dict()

        beta = standardized_betas_from_ols(y, Xc)

        # Build Table-2-style output: standardized betas for predictors, unstandardized constant
        terms_order = ["const"] + xcols
        out = pd.DataFrame(index=terms_order)
        out.index.name = "term"
        out["std_beta"] = beta.reindex(terms_order)
        out["b_unstd"] = model.params.reindex(terms_order)
        out["p_value_replication"] = model.pvalues.reindex(terms_order)
        out["sig_replication"] = out["p_value_replication"].apply(stars_from_p)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTES:\n")
            f.write("- Standardized betas computed as b * sd(x)/sd(y) on the estimation sample.\n")
            f.write("- p-values/stars are replication-derived; Table 2 in the paper reports betas + stars only.\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(out.to_string(float_format=lambda x: f"{x: .6f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

        return model, out, fit, d.index

    # -------------------------
    # Load and filter year
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: 'year' and/or 'id'.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Construct DVs (counts)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Allow partial item missingness to avoid halving N:
    # require answered >= 4 of 6 for minority DV, and >= 8 of 12 for other DV.
    df["dislike_minority_genres"] = build_dislike_count_allow_partial(df, minority_items, min_answered=4)
    df["dislike_other12_genres"] = build_dislike_count_allow_partial(df, other12_items, min_answered=8)

    # -------------------------
    # Racism score (0-5), require all 5 items answered (as specified)
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
    # Controls (RHS)
    # -------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_na(df["educ"]).where(clean_gss_na(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(clean_gss_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_gss_na(df["age"]).where(clean_gss_na(df["age"]).between(18, 89))
    df["age_years"] = age

    # Race dummies
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_na(df["race"]).where(clean_gss_na(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic dummy: not available in provided variables; include as all-missing so it doesn't silently
    # contaminate "other race". We still create it and keep it out of regressions.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_gss_na(df["relig"])
    denom = clean_gss_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_na(df["region"]).where(clean_gss_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -------------------------
    # Sanity checks for key dummies before modeling
    # -------------------------
    def freq01(series):
        s = series.dropna()
        return pd.Series(
            {
                "n_nonmissing": int(s.shape[0]),
                "n_0": int((s == 0).sum()),
                "n_1": int((s == 1).sum()),
                "share_1": float((s == 1).mean()) if s.shape[0] else np.nan,
            }
        )

    checks = pd.DataFrame(
        {
            "female": freq01(df["female"]),
            "black": freq01(df["black"]),
            "other_race": freq01(df["other_race"]),
            "cons_protestant": freq01(df["cons_protestant"]),
            "no_religion": freq01(df["no_religion"]),
            "south": freq01(df["south"]),
        }
    ).T

    checks_path = "./output/pre_model_checks.txt"
    with open(checks_path, "w", encoding="utf-8") as f:
        f.write("Pre-model 0/1 dummy frequency checks (after YEAR==1993 filter; before listwise deletion):\n")
        f.write(checks.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n\nNOTE: 'hispanic' is unavailable in the provided extract and is therefore not modeled.\n")

    # -------------------------
    # Model RHS (as faithful as possible with available fields)
    # IMPORTANT: include 'south' and 'no_religion' and do NOT drop them unless truly constant after listwise.
    # Hispanic cannot be included with this extract; kept out explicitly.
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

    # Ensure predictors exist
    for c in xcols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # -------------------------
    # Fit two models
    # -------------------------
    modelA, tableA, fitA, idxA = fit_model(df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6")
    modelB, tableB, fitB, idxB = fit_model(df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12")

    # Produce a "Table 2 style" output: standardized betas + stars; constant unstandardized
    def table2_style(table, title):
        # order like paper (no hispanic available)
        display_order = [
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
            "const",
        ]
        t = table.reindex(display_order).copy()

        # Pretty labels
        label_map = {
            "racism_score": "Racism score",
            "education_years": "Education (years)",
            "hh_income_per_capita": "Household income per capita",
            "occ_prestige": "Occupational prestige",
            "female": "Female",
            "age_years": "Age",
            "black": "Black",
            "other_race": "Other race",
            "cons_protestant": "Conservative Protestant",
            "no_religion": "No religion",
            "south": "Southern",
            "const": "Constant",
        }
        t.insert(0, "label", [label_map.get(i, i) for i in t.index])

        # Build reported coefficient column:
        # - predictors: standardized beta with stars
        # - constant: unstandardized b with stars
        coef_str = []
        for term, row in t.iterrows():
            if term == "const":
                val = row["b_unstd"]
            else:
                val = row["std_beta"]
            if pd.isna(val):
                coef_str.append("")
            else:
                coef_str.append(f"{val: .3f}{row['sig_replication']}")
        t["coef_reported"] = coef_str

        out = t[["label", "coef_reported"]].copy()
        out.index.name = "term"
        return out

    paperA = table2_style(tableA, "Model A")
    paperB = table2_style(tableB, "Model B")

    with open("./output/Table2_style_ModelA.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 style (replication): standardized betas for predictors; unstandardized constant\n")
        f.write("NOTE: Stars are based on replication p-values (Table 2 in paper does not report SEs).\n")
        f.write("NOTE: Hispanic dummy not available in provided extract; not included.\n\n")
        f.write(paperA.to_string(index=False))

    with open("./output/Table2_style_ModelB.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 style (replication): standardized betas for predictors; unstandardized constant\n")
        f.write("NOTE: Stars are based on replication p-values (Table 2 in paper does not report SEs).\n")
        f.write("NOTE: Hispanic dummy not available in provided extract; not included.\n\n")
        f.write(paperB.to_string(index=False))

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Replication overview (computed from provided microdata)\n")
        f.write("Key implementation notes:\n")
        f.write("- YEAR==1993 filter.\n")
        f.write("- DV counts are sums of per-genre dislike indicators (4/5=dislike; 1-3=not dislike).\n")
        f.write("- DV counts allow partial missingness to reduce sample collapse:\n")
        f.write("    * minority6: require >=4 of 6 answered\n")
        f.write("    * other12: require >=8 of 12 answered\n")
        f.write("- Racism score requires all 5 items non-missing.\n")
        f.write("- Standardized betas computed post-hoc: beta = b * sd(x)/sd(y) on estimation sample.\n")
        f.write("- Hispanic dummy is not present in this extract and is omitted (cannot reproduce paper exactly).\n\n")
        f.write("Fit statistics:\n")
        f.write(fitA.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n")
        f.write(fitB.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n\nPre-model dummy checks written to: ./output/pre_model_checks.txt\n")

    return {
        "ModelA_table_full": tableA,
        "ModelB_table_full": tableB,
        "ModelA_table2_style": paperA,
        "ModelB_table2_style": paperB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "pre_model_checks": checks,
    }