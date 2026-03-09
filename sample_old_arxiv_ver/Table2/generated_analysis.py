def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)

    # ---- Filter year ----
    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in data.")
    df = df.loc[df["year"] == 1993].copy()

    # ---- Helpers ----
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def dislike_indicator(x):
        # 1 if 4/5, 0 if 1/2/3, missing otherwise
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich_item(x, ones, zeros):
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def zscore(s):
        s = to_num(s)
        m = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd == 0 or np.isnan(sd):
            return s * np.nan
        return (s - m) / sd

    def standardized_ols(endog, exog_df):
        # Standardize y and X, fit OLS with intercept; coefficients on standardized X are beta weights.
        y = zscore(endog)
        Xz = exog_df.apply(zscore, axis=0)
        data = pd.concat([y.rename("y"), Xz], axis=1).dropna()
        y2 = data["y"]
        X2 = sm.add_constant(data.drop(columns=["y"]), has_constant="add")
        model = sm.OLS(y2, X2).fit()

        # Build a clean table
        tab = pd.DataFrame(
            {
                "beta": model.params,
                "se": model.bse,
                "t": model.tvalues,
                "p": model.pvalues,
            }
        )
        # Beta for intercept not meaningful; keep but label clearly
        tab.index = tab.index.map(lambda s: "const" if s == "const" else s)

        return model, tab, data.shape[0]

    # ---- Dependent variables (counts of dislikes; require complete items) ----
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
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

    missing_music_cols = [c for c in (minority_genres + remaining_genres) if c not in df.columns]
    if missing_music_cols:
        raise ValueError(f"Missing expected music columns: {missing_music_cols}")

    for c in minority_genres + remaining_genres:
        df[c] = to_num(df[c])

    # Dislike indicators
    for c in minority_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    for c in remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    # Counts; set missing if any component missing (listwise within DV construction)
    dv1_components = [f"d_{c}" for c in minority_genres]
    dv2_components = [f"d_{c}" for c in remaining_genres]

    df["dv1_minority_dislike"] = df[dv1_components].sum(axis=1, min_count=len(dv1_components))
    df.loc[df[dv1_components].isna().any(axis=1), "dv1_minority_dislike"] = np.nan

    df["dv2_remaining_dislike"] = df[dv2_components].sum(axis=1, min_count=len(dv2_components))
    df.loc[df[dv2_components].isna().any(axis=1), "dv2_remaining_dislike"] = np.nan

    # ---- Racism score (0-5) ----
    needed_racism = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    missing_racism = [c for c in needed_racism if c not in df.columns]
    if missing_racism:
        raise ValueError(f"Missing expected racism-item columns: {missing_racism}")

    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])

    racism_components = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = df[racism_components].sum(axis=1, min_count=len(racism_components))
    df.loc[df[racism_components].isna().any(axis=1), "racism_score"] = np.nan

    # ---- Controls ----
    for c in ["educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    # Income per capita
    df["income_pc"] = np.where((df["hompop"] > 0) & df["realinc"].notna() & df["hompop"].notna(),
                               df["realinc"] / df["hompop"], np.nan)

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Race dummies (reference: white)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic not available; omit (cannot replicate this term).
    # Conservative Protestant proxy: RELIG==1 and DENOM==1 (baptist)
    df["cons_prot"] = np.where(df["relig"].isin([1, 2, 3, 4, 5, 6, 7]) | df["relig"].isna(),
                               np.where((df["relig"] == 1) & (df["denom"] == 1), 1.0,
                                        np.where(df["relig"].notna() & df["denom"].notna(), 0.0, np.nan)),
                               np.nan)

    # No religion
    df["no_religion"] = np.where(df["relig"].isin([1, 2, 3, 4, 5, 6, 7]), (df["relig"] == 4).astype(float), np.nan)

    # Southern
    df["southern"] = np.where(df["region"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), (df["region"] == 3).astype(float), np.nan)

    # ---- Model matrices ----
    predictors = [
        "racism_score",
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    missing_pred = [c for c in predictors if c not in df.columns]
    if missing_pred:
        raise ValueError(f"Missing expected predictors after construction: {missing_pred}")

    X = df[predictors].copy()

    # ---- Fit models (standardized betas) ----
    results = {}

    for dv_name, dv_col in [
        ("model_dv1_minority_linked", "dv1_minority_dislike"),
        ("model_dv2_remaining", "dv2_remaining_dislike"),
    ]:
        y = df[dv_col]
        model, tab, n_used = standardized_ols(y, X)

        # Add fit stats
        fit = pd.DataFrame(
            {
                "N_used": [n_used],
                "R2": [model.rsquared],
                "Adj_R2": [model.rsquared_adj],
                "DF_model": [model.df_model],
                "DF_resid": [model.df_resid],
            },
            index=[dv_name],
        )

        results[dv_name] = {"table": tab, "fit": fit, "statsmodels_summary": model.summary().as_text()}

        # Save per-model text outputs
        with open(f"./output/{dv_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- OLS on standardized y and standardized X; coefficients are standardized beta weights.\n")
            f.write("- Listwise deletion applied to DV and all included predictors.\n")
            f.write("- Hispanic indicator not included (not available in provided fields).\n")
            f.write("- Conservative Protestant proxied as RELIG==Protestant and DENOM==Baptist.\n")

        tab_out = tab.copy()
        tab_out.index.name = "term"
        tab_out.to_csv(f"./output/{dv_name}_regression_table.csv")

        with open(f"./output/{dv_name}_regression_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_out.to_string(float_format=lambda v: f"{v:0.4f}"))
            f.write("\n\n")
            f.write(fit.to_string(float_format=lambda v: f"{v:0.4f}"))

    # ---- Combined human-readable summary ----
    combined_fit = pd.concat([results[k]["fit"] for k in results], axis=0)

    combined_betas = []
    for k in results:
        t = results[k]["table"][["beta", "se", "t", "p"]].copy()
        t["model"] = k
        combined_betas.append(t.reset_index().rename(columns={"index": "term"}))
    combined_betas = pd.concat(combined_betas, ignore_index=True)

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Replicated models (as closely as possible with provided fields)\n")
        f.write("===========================================================\n\n")
        f.write("Fit statistics\n--------------\n")
        f.write(combined_fit.to_string(float_format=lambda v: f"{v:0.4f}"))
        f.write("\n\nStandardized coefficients (beta weights)\n---------------------------------------\n")
        f.write(combined_betas.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        f.write("\n")

    results_tables = {
        "fit": combined_fit,
        "betas_long": combined_betas,
        "dv_descriptives": df[["dv1_minority_dislike", "dv2_remaining_dislike"]].describe(),
    }
    results_tables["dv_descriptives"].to_csv("./output/dv_descriptives.csv")

    return results_tables