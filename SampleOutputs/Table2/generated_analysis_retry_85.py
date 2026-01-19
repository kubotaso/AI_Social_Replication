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

    def clean_gss_missing(x):
        """
        Conservative missing-code cleaning for this extract.
        We only blank out common GSS sentinel codes; we do NOT drop valid values.
        """
        x = to_num(x).copy()
        sentinels = {
            8, 9, 98, 99, 997, 998, 999,
            9997, 9998, 9999,
            99997, 99998, 99999
        }
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 like/dislike scale.
        Dislike = 4 or 5. Like/neutral = 1,2,3. Other/missing => NA.
        """
        x = clean_gss_missing(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_gss_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete_case(df, items):
        """
        Count dislikes across items. DK/NA at item-level treated as missing.
        To match 'DK treated as missing and cases excluded', require complete data
        on ALL items for the count (listwise for DV components).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardized_betas_from_ols(y, X, add_intercept=True):
        """
        Fit OLS on raw y, raw X, then compute standardized betas:
            beta_j = b_j * sd(X_j) / sd(y)
        Intercept is kept as raw intercept (not standardized).
        """
        X_fit = X.copy()
        if add_intercept:
            X_fit = sm.add_constant(X_fit, has_constant="add")
        model = sm.OLS(y, X_fit).fit()

        y_sd = y.std(ddof=0)
        betas = pd.Series(index=model.params.index, dtype="float64")
        betas.loc[:] = np.nan
        if "const" in betas.index:
            betas.loc["const"] = model.params.loc["const"]

        # standardized betas for non-constant regressors
        for col in X.columns:
            x_sd = X[col].std(ddof=0)
            if np.isfinite(x_sd) and x_sd > 0 and np.isfinite(y_sd) and y_sd > 0:
                betas.loc[col] = model.params.loc[col] * (x_sd / y_sd)

        return model, betas

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    # -----------------------------
    # Load + filter year
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs: dislike counts (complete-case on the DV components)
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
    # Racism score (0-5): complete-case on all 5 components
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_gss_missing(df["educ"])
    df["education"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_pc"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_gss_missing(df["age"])
    df["age"] = age.where(age.between(18, 89))

    # Race dummies: black, other_race (white reference)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: present in paper, but not in provided variables. Use ETHNIC proxy ONLY if it exists.
    # This keeps the model runnable and avoids all-missing causing n=0.
    # NOTE: This is a proxy; if a proper Hispanic flag exists in the full dataset, replace here.
    if "hispanic" in df.columns:
        hisp = clean_gss_missing(df["hispanic"])
        # try common codings: 1=yes 2=no or 1/0
        if hisp.dropna().isin([1, 2]).any():
            df["hispanic_dummy"] = binary_from_codes(hisp, true_codes=[1], false_codes=[2])
        else:
            df["hispanic_dummy"] = hisp.where(hisp.isin([0, 1]))
    elif "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        # Very common ANCESTRY-ish codes: treat typical "Hispanic/Latino" ranges as 200-299 if present.
        # If codes are not in that range, fallback to missing (avoid fabricating).
        if eth.dropna().between(200, 299).any():
            df["hispanic_dummy"] = np.where(eth.isna(), np.nan, eth.between(200, 299).astype(float))
        else:
            df["hispanic_dummy"] = np.nan
    else:
        df["hispanic_dummy"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["southern"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Fit models: listwise deletion on exactly DV + RHS
    # (Keep intercept; compute standardized betas post hoc; intercept remains raw.)
    # -----------------------------
    rhs_order = [
        "racism_score",
        "education",
        "hh_income_pc",
        "occ_prestige",
        "female",
        "age",
        "black",
        "hispanic_dummy",
        "other_race",
        "cons_protestant",
        "no_religion",
        "southern",
    ]

    pretty = {
        "const": "Constant",
        "racism_score": "Racism score",
        "education": "Education",
        "hh_income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic_dummy": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    def fit_one(dv_col, model_name):
        needed = [dv_col] + rhs_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(rhs_order) + 5):
            # Write diagnostics then raise a clearer error
            miss = df[needed].isna().mean().sort_values(ascending=False)
            diag = "Not enough complete cases.\n\nMissingness rates (full 1993 sample):\n"
            diag += miss.to_string()
            write_text(f"./output/{model_name}_diagnostics.txt", diag)
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(rhs_order)}).")

        y = to_num(d[dv_col])
        X = d[rhs_order].apply(to_num)

        # Check rank / constant columns
        zero_var = [c for c in X.columns if X[c].std(ddof=0) == 0 or not np.isfinite(X[c].std(ddof=0))]
        # Keep them (to mirror intended spec) but record; statsmodels will drop via singularity handling.
        X_fit = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_fit).fit()

        # Standardized betas computed from the SAME estimation sample
        model2, betas = standardized_betas_from_ols(y, X, add_intercept=True)
        # model2 should match model in coefficients; use model for SE/pvals
        betas = betas.reindex(model.params.index)

        out = pd.DataFrame(
            {
                "term": betas.index,
                "label": [pretty.get(t, t) for t in betas.index],
                "beta_std": betas.values,
                "p_value": model.pvalues.reindex(betas.index).values,
            }
        )
        out["sig"] = out["p_value"].apply(stars)
        out = out.set_index("label")[["beta_std", "sig", "p_value"]]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_or_zero_var_predictors_note": ", ".join(zero_var) if zero_var else "",
                }
            ]
        )

        # Save human-readable outputs
        write_text(f"./output/{model_name}_summary.txt", model.summary().as_text())
        write_text(f"./output/{model_name}_table.txt", out.to_string(float_format=lambda v: f"{v: .6f}"))

        return model, out, fit

    modelA, tableA, fitA = fit_one("dv_dislike_minority6", "Table2_ModelA_Dislike_Minority_Associated6")
    modelB, tableB, fitB = fit_one("dv_dislike_other12", "Table2_ModelB_Dislike_Other12_Remaining")

    # Combined overview
    overview = []
    overview.append("Table 2 replication attempt (GSS 1993) from provided microdata extract.\n")
    overview.append("OLS on raw DV counts; standardized coefficients computed as b*sd(x)/sd(y) on estimation sample.\n")
    overview.append("Stars are based on model p-values (two-tailed): * p<.05, ** p<.01, *** p<.001.\n")
    overview.append("NOTE: If a proper Hispanic identifier is absent, 'Hispanic' may be a proxy or missing.\n\n")
    overview.append("=== Model A: DV = count dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin ===\n")
    overview.append(fitA.to_string(index=False))
    overview.append("\n\n")
    overview.append(tableA.to_string(float_format=lambda v: f"{v: .6f}"))
    overview.append("\n\n=== Model B: DV = count dislikes among the 12 remaining genres ===\n")
    overview.append(fitB.to_string(index=False))
    overview.append("\n\n")
    overview.append(tableB.to_string(float_format=lambda v: f"{v: .6f}"))
    write_text("./output/Table2_overview.txt", "".join(overview))

    return {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }