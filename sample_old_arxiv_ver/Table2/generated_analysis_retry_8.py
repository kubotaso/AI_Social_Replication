def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + year filter
    # -----------------------------
    df = pd.read_csv(data_source)

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
        missing otherwise (including DK/NA/refused if coded non-1..5).
        """
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich_item(x, ones, zeros):
        """
        Dichotomize item to 0/1; anything else -> missing.
        """
        x = to_num(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def strict_row_sum(dfin, cols):
        """
        Strict sum: missing if ANY component is missing (skipna=False).
        """
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

    def safe_binary(series, true_value):
        s = to_num(series)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.notna()] = (s.loc[s.notna()] == true_value).astype(float)
        return out

    def standardized_betas_from_fit(fit, y, X_no_const):
        """
        Post-estimation standardized slopes:
            beta_j = b_j * sd(X_j) / sd(Y)
        computed on the analytic sample (after listwise deletion).
        """
        y_sd = float(to_num(y).std(ddof=0))
        betas = {}
        for term in X_no_const.columns:
            b = float(fit.params.get(term, np.nan))
            x_sd = float(to_num(X_no_const[term]).std(ddof=0))
            if np.isnan(b) or np.isnan(y_sd) or y_sd == 0 or np.isnan(x_sd) or x_sd == 0:
                betas[term] = np.nan
            else:
                betas[term] = b * (x_sd / y_sd)
        return pd.Series(betas, name="beta")

    def drop_nonvarying_predictors(X, predictors_in_order):
        kept, dropped = [], []
        for c in predictors_in_order:
            s = X[c]
            # drop if all-missing or no variation (incl. constant dummy)
            if s.notna().sum() == 0 or s.dropna().nunique() <= 1:
                dropped.append(c)
            else:
                kept.append(c)
        return kept, dropped

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
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Coerce numeric (id can remain as-is)
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables: strict dislike counts
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_dislike_minority_linked_6"] = strict_row_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_dislike_remaining_12"] = strict_row_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5): strict sum of 5 dichotomies (direction per mapping)
    # -----------------------------
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1

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

    # Race dummies (white reference; set missing if race not in {1,2,3})
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic proxy from ETHNIC if present (best-effort; documented in outputs)
    df["hispanic"] = np.nan
    eth = df["ethnic"]
    if eth.notna().any():
        # Defensive proxy: either explicit 16, or 20-29 block (common "Spanish/Hispanic origin" codes in some extracts)
        hisp_mask = eth.isin([16]) | ((eth >= 20) & (eth < 30))
        df.loc[eth.notna(), "hispanic"] = hisp_mask.loc[eth.notna()].astype(float)

    # Conservative Protestant proxy: RELIG==1 (protestant) & DENOM==1 (baptist)
    df["cons_prot"] = np.nan
    rel_denom_known = df["relig"].notna() & df["denom"].notna()
    df.loc[rel_denom_known, "cons_prot"] = (
        (df.loc[rel_denom_known, "relig"] == 1) & (df.loc[rel_denom_known, "denom"] == 1)
    ).astype(float)

    # No religion (FIX: include explicitly)
    df["no_religion"] = safe_binary(df["relig"], 4)

    # Southern
    df["southern"] = safe_binary(df["region"], 3)

    # Predictor order aligned to Table 2
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
        "educ": "Education",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic (proxy from ETHNIC)",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy: Prot & Baptist)",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    # -----------------------------
    # Model runner: outputs "paper style" only (betas + stars + N/R2/AdjR2/Constant)
    # -----------------------------
    def fit_table2_style(dv_col, model_tag, dv_label):
        cols = [dv_col] + predictors
        d = df[cols].copy()

        # listwise deletion on DV + all predictors requested (baseline)
        d_listwise = d.dropna().copy()

        # If listwise is very small, we still proceed (replication limitations will appear in diagnostics)
        y = d_listwise[dv_col].astype(float)
        X = d_listwise[predictors].astype(float)

        # Drop non-varying predictors AFTER listwise deletion to prevent singularities/NaN rows
        kept, dropped = drop_nonvarying_predictors(X, predictors)
        X = X[kept].copy()
        Xc = sm.add_constant(X, has_constant="add")

        fit = sm.OLS(y, Xc).fit()

        beta = standardized_betas_from_fit(fit, y, X)
        pvals = fit.pvalues.reindex(["const"] + kept)
        stars = pvals.apply(star_from_p)

        table = pd.DataFrame(
            {"beta": beta.reindex(kept), "star": stars.reindex(kept).fillna("")},
            index=kept,
        )
        table.index = [pretty.get(i, i) for i in table.index]

        fit_stats = pd.DataFrame(
            {
                "N": [int(fit.nobs)],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_star": [stars.get("const", "")],
            },
            index=[model_tag],
        )

        # Save paper-style table text
        out_lines = []
        out_lines.append(model_tag)
        out_lines.append("=" * len(model_tag))
        out_lines.append("")
        out_lines.append(f"DV: {dv_label}")
        out_lines.append("Estimation: OLS. Reported coefficients are standardized betas (beta weights).")
        out_lines.append("Standardization: beta_j = b_j * SD(X_j)/SD(Y), computed on this model's analytic sample.")
        out_lines.append("Dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        out_lines.append("Index construction: strict count (missing if any component item missing).")
        out_lines.append("Missing data: listwise deletion on DV + all included predictors.")
        out_lines.append("Stars: two-tailed p-values (* p<.05, ** p<.01, *** p<.001).")
        out_lines.append("")
        out_lines.append("Standardized coefficients (Table 2 style)")
        out_lines.append("---------------------------------------")
        out_lines.append(table.to_string(float_format=lambda v: f"{v:0.3f}"))
        out_lines.append("")
        out_lines.append("Fit statistics")
        out_lines.append("--------------")
        out_lines.append(fit_stats.to_string(float_format=lambda v: f"{v:0.3f}"))
        out_lines.append("")
        out_lines.append("Design matrix columns (kept, in order)")
        out_lines.append("-------------------------------------")
        out_lines.append(", ".join(kept) if kept else "(none)")
        if dropped:
            out_lines.append("")
            out_lines.append("Dropped predictors (no variation after listwise deletion)")
            out_lines.append("--------------------------------------------------------")
            out_lines.append(", ".join(dropped))

        with open(f"./output/{model_tag}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))

        # Save diagnostics (not part of paper table)
        with open(f"./output/{model_tag}_ols_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Save CSVs
        table.to_csv(f"./output/{model_tag}_table2_style.csv", index=True)
        fit_stats.to_csv(f"./output/{model_tag}_fit.csv", index=True)

        return table, fit_stats, kept, dropped, fit

    t1, fs1, kept1, dropped1, m1 = fit_table2_style(
        "dv1_dislike_minority_linked_6",
        "Model_1_Dislike_MinorityLinked_6",
        "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
    )
    t2, fs2, kept2, dropped2, m2 = fit_table2_style(
        "dv2_dislike_remaining_12",
        "Model_2_Dislike_Remaining_12",
        "Dislike of the 12 Remaining Genres (count of 12)",
    )

    combined_fit = pd.concat([fs1, fs2], axis=0)

    combined_betas = pd.concat(
        [
            t1.rename(columns={"beta": "beta_model_1", "star": "star_model_1"}),
            t2.rename(columns={"beta": "beta_model_2", "star": "star_model_2"}),
        ],
        axis=1,
    )

    # Descriptives for constructed DVs (before listwise)
    dv_desc = df[["dv1_dislike_minority_linked_6", "dv2_dislike_remaining_12"]].describe()

    # Missingness diagnostics to explain N collapse if it happens
    key1 = ["dv1_dislike_minority_linked_6"] + predictors
    key2 = ["dv2_dislike_remaining_12"] + predictors
    miss1 = df[key1].isna().mean().sort_values(ascending=False).to_frame("share_missing")
    miss2 = df[key2].isna().mean().sort_values(ascending=False).to_frame("share_missing")

    music1_items = [f"d_{c}" for c in minority_genres]
    music2_items = [f"d_{c}" for c in remaining_genres]
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    item_missing = pd.DataFrame(
        {
            "missing_share_dv1_items": df[music1_items].isna().mean(),
            "missing_share_dv2_items": df[music2_items].isna().mean(),
            "missing_share_racism_items": df[racism_comp].isna().mean(),
        }
    )

    # Combined summary
    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt (1993 GSS; available fields)\n")
        f.write("======================================================================\n\n")
        f.write("Combined fit statistics\n")
        f.write("----------------------\n")
        f.write(combined_fit.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nCombined standardized coefficients (Table 2 style)\n")
        f.write("-------------------------------------------------\n")
        f.write(combined_betas.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nDV descriptives (constructed counts; before listwise deletion)\n")
        f.write("-------------------------------------------------------------\n")
        f.write(dv_desc.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nMissingness shares (Model 1 variables)\n")
        f.write("-------------------------------------\n")
        f.write(miss1.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nMissingness shares (Model 2 variables)\n")
        f.write("-------------------------------------\n")
        f.write(miss2.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nPer-item missingness (music + racism components)\n")
        f.write("----------------------------------------------\n")
        f.write(item_missing.to_string(float_format=lambda v: f"{v:0.3f}"))
        f.write("\n\nNotes\n-----\n")
        f.write("- Paper-style outputs are the *_table2_style.txt files (standardized betas + stars + N/R2/AdjR2/constant).\n")
        f.write("- SEs/t/p are not shown in the paper-style table; they are available in *_ols_diagnostics.txt.\n")
        f.write("- Hispanic is proxied from ETHNIC because this extract lacks a dedicated Hispanic indicator.\n")
        f.write("- Conservative Protestant is proxied as RELIG==1 (Protestant) & DENOM==1 (Baptist).\n")
        f.write("- If N is far below the published table, see missingness diagnostics to identify drivers.\n")

    # Save CSVs
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    combined_betas.to_csv("./output/combined_table2_style.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=True)
    miss1.to_csv("./output/missingness_model_1.csv", index=True)
    miss2.to_csv("./output/missingness_model_2.csv", index=True)
    item_missing.to_csv("./output/item_missingness.csv", index=True)

    return {
        "table2_style": combined_betas,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_model_1": miss1,
        "missingness_model_2": miss2,
        "item_missingness": item_missing,
        "model_1_predictors_used": pd.DataFrame({"predictor": kept1}),
        "model_2_predictors_used": pd.DataFrame({"predictor": kept2}),
    }