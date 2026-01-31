def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)

    # -----------------------------
    # Basic checks + year filter
    # -----------------------------
    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def dislike_indicator(x):
        """
        1 if 4/5 (dislike/dislike very much), 0 if 1/2/3, else missing.
        """
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

    def strict_row_sum(dfin, cols):
        """
        Row-wise sum, but returns missing if any component missing.
        """
        s = dfin[cols].sum(axis=1)
        s[dfin[cols].isna().any(axis=1)] = np.nan
        return s

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

    def standardized_betas(unstd_model, y, X_no_const):
        """
        beta_std_j = b_j * sd(X_j) / sd(Y), computed on the analytic sample.
        Intercept excluded.
        """
        y = to_num(y)
        y_sd = y.std(skipna=True, ddof=0)
        out = {}
        for term in unstd_model.params.index:
            if term == "const":
                continue
            x = to_num(X_no_const[term])
            x_sd = x.std(skipna=True, ddof=0)
            b = unstd_model.params[term]
            if pd.isna(y_sd) or y_sd == 0 or pd.isna(x_sd) or x_sd == 0:
                out[term] = np.nan
            else:
                out[term] = float(b) * float(x_sd) / float(y_sd)
        return pd.Series(out, name="beta_std")

    def drop_empty_or_constant_cols(X):
        """
        Remove columns that are all-missing, constant, or nearly constant (avoid singular design matrix).
        Keeps numeric columns only.
        """
        X2 = X.copy()
        keep = []
        for c in X2.columns:
            s = X2[c]
            if s.notna().sum() == 0:
                continue
            # constant among non-missing
            if s.dropna().nunique() <= 1:
                continue
            keep.append(c)
        return X2[keep].copy()

    # -----------------------------
    # Required columns
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_cols = [
        "id", "hompop", "educ", "realinc", "prestg80", "sex", "age",
        "race", "relig", "denom", "region"
    ]
    needed = base_cols + minority_genres + remaining_genres + racism_items
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # numeric coercion
    for c in needed:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables (strict count outcomes)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_dislike_minority_linked_6"] = strict_row_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_dislike_remaining_12"] = strict_row_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5, strict)
    # -----------------------------
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])     # object to school > half black
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])     # oppose busing
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])   # not due to discrimination
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])   # not due to lack of education opportunity
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])   # due to lack of motivation/will power

    df["racism_score"] = strict_row_sum(df, ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"])

    # -----------------------------
    # Controls
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Race dummies (reference: white/non-hisp white in paper; we only have RACE)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: not available in provided variables -> omit from model to avoid artificial missingness
    # (Including an all-missing column forces huge N loss and/or singularities.)
    # If you later add a clean Hispanic flag column, include it here and in predictors_order.
    # df["hispanic"] = ...

    # Conservative Protestant proxy: RELIG==1 (protestant) and DENOM==1 (baptist)
    df["cons_prot"] = np.nan
    denom_known = df["relig"].notna() & df["denom"].notna()
    df.loc[denom_known, "cons_prot"] = ((df.loc[denom_known, "relig"] == 1) & (df.loc[denom_known, "denom"] == 1)).astype(float)

    # No religion
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictor order to match Table 2 as closely as possible with available fields
    predictors_order = [
        "racism_score",
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        # "hispanic",  # not available in this extract
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    pretty_names = {
        "racism_score": "Racism score",
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy)",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    # -----------------------------
    # Model fitting + Table-2-style output (ONLY standardized betas + stars + constant + fit stats)
    # -----------------------------
    def fit_table2_style(dv_col, model_label):
        cols_needed = [dv_col] + predictors_order
        d = df[cols_needed].copy().dropna()

        # Design matrix, drop empty/constant predictors (prevents the "0.000000, se=0, p=NaN" row)
        y = d[dv_col].astype(float)
        X = d[predictors_order].astype(float)
        X = drop_empty_or_constant_cols(X)

        # keep order where possible
        kept = [c for c in predictors_order if c in X.columns]
        X = X[kept]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        beta_std = standardized_betas(model, y, X)

        # Stars from the OLS two-tailed p-values for each term (excluding intercept)
        pvals = model.pvalues.drop(labels=["const"], errors="ignore")
        stars = pvals.apply(star_from_p)

        # Build Table 2 style: standardized betas + stars, in predictor order
        out = pd.DataFrame({"beta": beta_std, "star": stars})
        out = out.reindex(kept)

        out_named = out.copy()
        out_named.index = [pretty_names.get(i, i) for i in out_named.index]

        fit_stats = pd.DataFrame(
            {
                "N": [int(model.nobs)],
                "R2": [float(model.rsquared)],
                "Adj_R2": [float(model.rsquared_adj)],
                "Constant (unstd)": [float(model.params.get("const", np.nan))],
            },
            index=[model_label],
        )

        # Save text outputs: table only (paper-like) + fit stats
        with open(f"./output/{model_label}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("Table-2-style output: Standardized OLS coefficients (beta weights)\n")
            f.write("==================================================================\n\n")
            f.write(f"DV: {dv_col}\n")
            f.write("Coding: dislike = 1 if genre rating in {4,5}; 0 if in {1,2,3}; missing otherwise.\n")
            f.write("DV construction: strict count (missing if any component missing).\n")
            f.write("Estimation: OLS; reported coefficients are standardized betas; stars from two-tailed OLS p-values.\n\n")
            f.write(out_named.to_string(float_format=lambda v: f"{v:0.3f}"))
            f.write("\n\nFit statistics\n--------------\n")
            f.write(fit_stats.to_string(float_format=lambda v: f"{v:0.3f}"))
            f.write("\n")

        # Also save a compact CSV
        out_csv = out_named.copy()
        out_csv.to_csv(f"./output/{model_label}_table2_style.csv", index=True)
        fit_stats.to_csv(f"./output/{model_label}_fit.csv", index=True)

        # Save a model summary for debugging (not presented as "paper table")
        with open(f"./output/{model_label}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: The paper prints standardized betas only; this summary is saved for diagnostics.\n")

        return out_named, fit_stats, model

    t2a, fit1, m1 = fit_table2_style("dv1_dislike_minority_linked_6", "model_2A_dislike_minority_linked_6")
    t2b, fit2, m2 = fit_table2_style("dv2_dislike_remaining_12", "model_2B_dislike_remaining_12")

    combined_fit = pd.concat([fit1, fit2], axis=0)

    combined_betas = pd.concat(
        [
            t2a.rename(columns={"beta": "beta_model_2A", "star": "star_model_2A"}),
            t2b.rename(columns={"beta": "beta_model_2B", "star": "star_model_2B"}),
        ],
        axis=1,
    )

    # DV descriptives (constructed counts before listwise deletion)
    dv_desc = df[["dv1_dislike_minority_linked_6", "dv2_dislike_remaining_12"]].describe()

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (1993 GSS; available fields)\n")
        f.write("======================================================================\n\n")
        f.write("IMPORTANT LIMITATION\n")
        f.write("--------------------\n")
        f.write("This extract does not include a clean Hispanic indicator; the Table 2 Hispanic dummy cannot be reproduced.\n")
        f.write("Accordingly, the models omit Hispanic to avoid severe sample loss and singularities.\n\n")
        f.write("Fit statistics\n--------------\n")
        f.write(combined_fit.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nStandardized coefficients (Table-2-style)\n----------------------------------------\n")
        f.write(combined_betas.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nDV descriptives (constructed counts; before listwise deletion)\n-------------------------------------------------------------\n")
        f.write(dv_desc.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n")

    combined_fit.to_csv("./output/combined_fit.csv")
    combined_betas.to_csv("./output/combined_table2_style.csv")
    dv_desc.to_csv("./output/dv_descriptives.csv")

    return {
        "table2_style": combined_betas,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
    }