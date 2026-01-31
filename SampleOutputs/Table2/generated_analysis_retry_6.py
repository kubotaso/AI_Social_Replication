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
        1 if 4/5 (dislike/dislike very much),
        0 if 1/2/3,
        missing otherwise.
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
        # Missing if ANY component missing (strict/listwise index construction)
        return dfin[cols].sum(axis=1, skipna=False)

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

    def standardized_betas_from_fit(fit, y, X_no_const):
        """
        Standardized beta weights computed from unstandardized OLS:
            beta_std_j = b_j * sd(X_j) / sd(Y)
        computed on the analytic sample (after listwise deletion).
        """
        y = to_num(y)
        y_sd = float(y.std(ddof=0))
        betas = {}
        for term in X_no_const.columns:
            b = float(fit.params.get(term, np.nan))
            x_sd = float(to_num(X_no_const[term]).std(ddof=0))
            if np.isnan(b) or y_sd == 0 or np.isnan(y_sd) or x_sd == 0 or np.isnan(x_sd):
                betas[term] = np.nan
            else:
                betas[term] = b * (x_sd / y_sd)
        return pd.Series(betas, name="beta")

    def safe_binary(series, true_value):
        s = to_num(series)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.notna()] = (s.loc[s.notna()] == true_value).astype(float)
        return out

    # -----------------------------
    # Variable lists (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Coerce numeric (id can be left as-is)
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_dislike_minority_linked_6"] = strict_row_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_dislike_remaining_12"] = strict_row_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])

    df["racism_score"] = strict_row_sum(df, ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"])

    # -----------------------------
    # Controls / dummies
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = safe_binary(df["sex"], 2)

    # Race dummies (white reference)
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic: best-effort from ETHNIC (since dedicated Hispanic flag not present)
    # Use common "Spanish/Hispanic" code ranges when possible; else leave missing.
    df["hispanic"] = np.nan
    eth = df["ethnic"]
    if eth.notna().any():
        # Broad, defensive approach:
        # - Many GSS ETHNIC schemes use codes in ~20-29 for Spanish/Hispanic origins.
        # - Some extracts use 16 as "Hispanic".
        hisp_mask = (
            eth.isin([16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
            | ((eth >= 20) & (eth < 30))
        )
        df.loc[eth.notna(), "hispanic"] = hisp_mask.loc[eth.notna()].astype(float)

    # Conservative Protestant proxy: RELIG==1 (protestant) & DENOM==1 (baptist)
    df["cons_prot"] = np.nan
    rel_denom_known = df["relig"].notna() & df["denom"].notna()
    df.loc[rel_denom_known, "cons_prot"] = (
        (df.loc[rel_denom_known, "relig"] == 1) & (df.loc[rel_denom_known, "denom"] == 1)
    ).astype(float)

    # No religion
    df["no_religion"] = safe_binary(df["relig"], 4)

    # Southern
    df["southern"] = safe_binary(df["region"], 3)

    # Predictor order (Table 2 order)
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

    pretty = {
        "racism_score": "Racism score",
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic (proxy from ETHNIC)",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy: Protestant & Baptist)",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    def fit_table2_style(dv_col, model_tag, dv_label):
        cols = [dv_col] + predictors
        d = df[cols].copy()

        # listwise deletion for THIS model only
        d = d.dropna()

        # Build X and drop any constant/empty columns AFTER listwise deletion to avoid singularities
        X = d[predictors].astype(float)
        y = d[dv_col].astype(float)

        kept = []
        dropped = []
        for c in predictors:
            if c not in X.columns:
                dropped.append(c)
                continue
            s = X[c]
            if s.notna().sum() == 0 or s.dropna().nunique() <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        X = X[kept].copy()

        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        beta = standardized_betas_from_fit(fit, y, X)
        pvals = fit.pvalues.reindex(["const"] + kept)
        stars = pvals.apply(star_from_p)

        # Table-2-style: standardized betas + stars, plus constant separately
        table = pd.DataFrame(
            {
                "beta": beta.reindex(kept),
                "star": stars.reindex(kept).fillna(""),
            },
            index=kept,
        )
        table.index = [pretty.get(i, i) for i in table.index]

        fit_stats = pd.DataFrame(
            {
                "N": [int(fit.nobs)],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant (unstd)": [float(fit.params.get("const", np.nan))],
                "Constant star": [stars.get("const", "")],
            },
            index=[model_tag],
        )

        # Save: human-readable table file matching paper-style reporting
        lines = []
        lines.append(f"{model_tag} (Table-2-style output)")
        lines.append("=" * len(lines[-1]))
        lines.append("")
        lines.append(f"DV: {dv_label}")
        lines.append("Estimation: OLS; coefficients reported are standardized betas (beta weights).")
        lines.append("Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("Index construction: strict count (missing if any component item missing).")
        lines.append("Missing data: listwise deletion on DV + all predictors used in this model.")
        lines.append("Stars: two-tailed p-values (* p<.05, ** p<.01, *** p<.001).")
        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("------------------------")
        lines.append(table.to_string(float_format=lambda v: f"{v:0.3f}"))
        lines.append("")
        lines.append("Fit statistics")
        lines.append("--------------")
        lines.append(fit_stats.to_string(float_format=lambda v: f"{v:0.3f}"))
        lines.append("")
        lines.append("Design columns used (in order)")
        lines.append("-----------------------------")
        lines.append(", ".join(kept) if kept else "(none)")
        if dropped:
            lines.append("")
            lines.append("Dropped predictors (constant/all-missing after listwise deletion)")
            lines.append("--------------------------------------------------------------")
            lines.append(", ".join(dropped))

        with open(f"./output/{model_tag}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Save diagnostics: full OLS summary (not the paper table)
        with open(f"./output/{model_tag}_ols_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n\nNOTE: Paper-style table is in *_table2_style.txt\n")

        # Save CSV versions
        table.to_csv(f"./output/{model_tag}_table2_style.csv", index=True)
        fit_stats.to_csv(f"./output/{model_tag}_fit.csv", index=True)

        return table, fit_stats, kept, dropped, fit

    t1, fs1, kept1, dropped1, m1 = fit_table2_style(
        "dv1_dislike_minority_linked_6",
        "Model_2A_Dislike_MinorityLinked_6",
        "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
    )
    t2, fs2, kept2, dropped2, m2 = fit_table2_style(
        "dv2_dislike_remaining_12",
        "Model_2B_Dislike_Remaining_12",
        "Dislike of the 12 Remaining Genres (count of 12)",
    )

    combined_fit = pd.concat([fs1, fs2], axis=0)

    combined_betas = pd.concat(
        [
            t1.rename(columns={"beta": "beta_model_2A", "star": "star_model_2A"}),
            t2.rename(columns={"beta": "beta_model_2B", "star": "star_model_2B"}),
        ],
        axis=1,
    )

    dv_desc = df[["dv1_dislike_minority_linked_6", "dv2_dislike_remaining_12"]].describe()

    # Missingness diagnostics: show shares missing for each key component to identify N collapse
    key_for_model1 = ["dv1_dislike_minority_linked_6"] + predictors
    key_for_model2 = ["dv2_dislike_remaining_12"] + predictors
    miss1 = df[key_for_model1].isna().mean().sort_values(ascending=False)
    miss2 = df[key_for_model2].isna().mean().sort_values(ascending=False)

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (1993 GSS; available fields)\n")
        f.write("======================================================================\n\n")
        f.write("Combined fit statistics\n")
        f.write("----------------------\n")
        f.write(combined_fit.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nCombined standardized coefficients (Table-2-style)\n")
        f.write("-------------------------------------------------\n")
        f.write(combined_betas.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nDV descriptives (constructed counts; before listwise deletion)\n")
        f.write("-------------------------------------------------------------\n")
        f.write(dv_desc.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nMissingness shares (Model 2A variables)\n")
        f.write("-------------------------------------\n")
        f.write(miss1.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nMissingness shares (Model 2B variables)\n")
        f.write("-------------------------------------\n")
        f.write(miss2.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nNOTE\n----\n")
        f.write("- Hispanic is approximated from ETHNIC because a dedicated Hispanic flag is not present in this extract.\n")
        f.write("- Conservative Protestant is proxied as RELIG==1 (Protestant) & DENOM==1 (Baptist) per mapping note.\n")
        f.write("- If N is far below the published table, missingness in music items and/or racism items is the usual cause.\n")

    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    combined_betas.to_csv("./output/combined_table2_style.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=True)
    miss1.to_csv("./output/missingness_model_2A.csv", index=True)
    miss2.to_csv("./output/missingness_model_2B.csv", index=True)

    return {
        "table2_style": combined_betas,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_model_2A": miss1.to_frame("share_missing"),
        "missingness_model_2B": miss2.to_frame("share_missing"),
        "model_2A_predictors_used": pd.DataFrame({"predictor": kept1}),
        "model_2B_predictors_used": pd.DataFrame({"predictor": kept2}),
    }