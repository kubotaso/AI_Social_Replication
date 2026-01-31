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

    def zscore(s):
        s = to_num(s)
        m = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if sd is None or np.isnan(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - m) / sd

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

    def count_sum_strict(df_in, cols):
        """
        Row-wise sum, but returns missing if any component missing.
        """
        s = df_in[cols].sum(axis=1)
        s[df_in[cols].isna().any(axis=1)] = np.nan
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

    def standardized_betas_from_unstandardized(unstd_model, y, X):
        """
        Compute standardized betas from an unstandardized OLS fit:
            beta_std_j = b_j * sd(X_j) / sd(Y)
        (Intercept is excluded / set to NaN)
        """
        y_sd = to_num(y).std(skipna=True, ddof=0)
        betas = {}
        for term in unstd_model.params.index:
            if term == "const":
                betas[term] = np.nan
                continue
            x_sd = to_num(X[term]).std(skipna=True, ddof=0)
            b = unstd_model.params[term]
            if y_sd == 0 or np.isnan(y_sd) or x_sd == 0 or np.isnan(x_sd):
                betas[term] = np.nan
            else:
                betas[term] = b * (x_sd / y_sd)
        return pd.Series(betas, name="beta_std")

    # -----------------------------
    # Required columns
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    core_cols = (
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]
        + minority_genres + remaining_genres + racism_items
    )
    missing = [c for c in core_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Coerce to numeric where applicable
    for c in core_cols:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables (counts)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    dv1_cols = [f"d_{c}" for c in minority_genres]
    dv2_cols = [f"d_{c}" for c in remaining_genres]

    df["dv1_dislike_minority_linked_6"] = count_sum_strict(df, dv1_cols)
    df["dv2_dislike_remaining_12"] = count_sum_strict(df, dv2_cols)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
    # Directions per mapping instruction
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])

    rcols = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = count_sum_strict(df, rcols)

    # -----------------------------
    # Controls
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: best-effort from ETHNIC if available.
    # In many GSS extracts, ETHNIC codes include a Hispanic/Spanish category.
    # We set hispanic=1 for a common set of "Hispanic/Spanish" codes if present; otherwise missing.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        # Common GSS ETHNIC patterns: 20=Mexican, 21=Puerto Rican, 22=Cuban, 23=Central/South American, 24=Other Spanish.
        hisp_codes = {20, 21, 22, 23, 24}
        # Some extracts instead use 16=Hispanic/Latino; include it defensively.
        hisp_codes |= {16}
        # If ETHNIC is in a completely different scheme, this yields mostly zeros (still defined).
        df.loc[df["ethnic"].notna(), "hispanic"] = df.loc[df["ethnic"].notna(), "ethnic"].isin(list(hisp_codes)).astype(float)

    # Conservative Protestant proxy: RELIG==1 (protestant) and DENOM==1 (baptist)
    df["cons_prot"] = np.nan
    denom_known = df["relig"].notna() & df["denom"].notna()
    df.loc[denom_known, "cons_prot"] = ((df.loc[denom_known, "relig"] == 1) & (df.loc[denom_known, "denom"] == 1)).astype(float)

    # No religion
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
        "racism_score",
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    # -----------------------------
    # Model fitting
    # -----------------------------
    def fit_one_model(dv_col, model_name):
        cols_needed = [dv_col] + predictors
        d = df[cols_needed].copy()
        d = d.dropna()  # listwise per model

        y = d[dv_col]
        X = d[predictors]
        X = sm.add_constant(X, has_constant="add")

        unstd = sm.OLS(y, X).fit()

        # Standardized betas computed from unstandardized coefficients (matches "standardized OLS coefficients")
        beta_std = standardized_betas_from_unstandardized(unstd, y, X.drop(columns=["const"]))

        # Build tables aligned strictly by term name (avoid any mis-merge bugs)
        params = unstd.params.rename("b_unstd")
        pvals = unstd.pvalues.rename("p")
        tvals = unstd.tvalues.rename("t")
        ses = unstd.bse.rename("se")

        table_full = pd.concat([params, ses, tvals, pvals], axis=1)
        table_full["beta_std"] = beta_std.reindex(table_full.index)
        table_full["star"] = table_full["p"].apply(star_from_p)

        # "Table 2 style" output: standardized betas only (no SE/t/p), exclude intercept
        table_t2 = table_full.loc[table_full.index != "const", ["beta_std", "star"]].copy()
        table_t2 = table_t2.rename(index={
            "racism_score": "Racism score",
            "educ": "Education (years)",
            "income_pc": "Household income per capita",
            "prestg80": "Occupational prestige",
            "female": "Female",
            "age": "Age",
            "black": "Black",
            "hispanic": "Hispanic",
            "other_race": "Other race",
            "cons_prot": "Conservative Protestant",
            "no_religion": "No religion",
            "southern": "Southern",
        })

        fit_stats = pd.DataFrame(
            {
                "N": [int(unstd.nobs)],
                "R2": [float(unstd.rsquared)],
                "Adj_R2": [float(unstd.rsquared_adj)],
                "DF_model": [float(unstd.df_model)],
                "DF_resid": [float(unstd.df_resid)],
                "Intercept_unstd": [float(unstd.params.get("const", np.nan))],
            },
            index=[model_name],
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(unstd.summary().as_text())
            f.write("\n\nNotes\n-----\n")
            f.write(f"- Year filtered: 1993 only.\n")
            f.write(f"- DV: {dv_col} (count of disliked genres; dislikes are ratings 4/5).\n")
            f.write("- Missing data: listwise deletion per model on DV + all predictors.\n")
            f.write("- Table-2-style output reports standardized betas (computed from unstandardized OLS).\n")
            f.write("- Hispanic is constructed from ETHNIC using a best-effort code set; verify against your extract coding.\n")
            f.write("- Conservative Protestant is proxied as RELIG==1 and DENOM==1 (Baptist).\n")

        # Human-readable regression tables
        table_full_out = table_full.copy()
        table_full_out.index.name = "term"
        table_full_out.to_csv(f"./output/{model_name}_regression_full.csv")

        with open(f"./output/{model_name}_regression_full.txt", "w", encoding="utf-8") as f:
            f.write("Unstandardized OLS + derived standardized betas\n")
            f.write("=============================================\n\n")
            f.write(table_full_out.to_string(float_format=lambda v: f"{v:0.6f}"))
            f.write("\n\nFit statistics\n--------------\n")
            f.write(fit_stats.to_string(float_format=lambda v: f"{v:0.6f}"))
            f.write("\n")

        table_t2_out = table_t2.copy()
        table_t2_out.index.name = "predictor"
        table_t2_out.to_csv(f"./output/{model_name}_table2_style.csv")
        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("Table-2-style: Standardized OLS coefficients (beta weights)\n")
            f.write("===========================================================\n\n")
            f.write(table_t2_out.to_string(float_format=lambda v: f"{v:0.6f}"))
            f.write("\n\n")
            f.write(fit_stats.to_string(float_format=lambda v: f"{v:0.6f}"))
            f.write("\n")

        return {
            "table2_style": table_t2_out,
            "full": table_full_out,
            "fit": fit_stats,
            "model": unstd,
        }

    res1 = fit_one_model("dv1_dislike_minority_linked_6", "model_2A_dv1_minority_linked")
    res2 = fit_one_model("dv2_dislike_remaining_12", "model_2B_dv2_remaining")

    # -----------------------------
    # Combined summary
    # -----------------------------
    fit_all = pd.concat([res1["fit"], res2["fit"]], axis=0)

    # Combine standardized betas side-by-side
    t2_1 = res1["table2_style"][["beta_std", "star"]].rename(columns={"beta_std": "beta_std_model_2A", "star": "star_2A"})
    t2_2 = res2["table2_style"][["beta_std", "star"]].rename(columns={"beta_std": "beta_std_model_2B", "star": "star_2B"})
    combined_t2 = t2_1.join(t2_2, how="outer")

    # DV descriptives (non-missing only; before listwise)
    dv_desc = df[["dv1_dislike_minority_linked_6", "dv2_dislike_remaining_12"]].describe()

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (with available fields)\n")
        f.write("=================================================================\n\n")
        f.write("Fit statistics\n--------------\n")
        f.write(fit_all.to_string(float_format=lambda v: f"{v:0.6f}"))
        f.write("\n\nStandardized coefficients (Table-2-style)\n----------------------------------------\n")
        f.write(combined_t2.to_string(float_format=lambda v: f"{v:0.6f}"))
        f.write("\n\nDV descriptives (raw constructed counts, before listwise deletion)\n-----------------------------------------------------------------\n")
        f.write(dv_desc.to_string(float_format=lambda v: f"{v:0.6f}"))
        f.write("\n")

    # Save combined tables
    fit_all.to_csv("./output/combined_fit.csv")
    combined_t2.to_csv("./output/combined_table2_style.csv")
    dv_desc.to_csv("./output/dv_descriptives.csv")

    return {
        "fit": fit_all,
        "table2_style": combined_t2,
        "dv_descriptives": dv_desc,
        "model_2A_full": res1["full"],
        "model_2B_full": res2["full"],
    }