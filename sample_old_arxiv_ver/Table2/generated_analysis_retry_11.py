def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + normalize column names
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def dislike_indicator(series):
        """
        1 if 4/5 (dislike/dislike very much),
        0 if 1/2/3,
        missing otherwise.
        """
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich_item(series, ones, zeros):
        """
        Map to {0,1}; other values -> missing.
        """
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def strict_sum(dfin, cols):
        # Missing if ANY component missing (skipna=False).
        return dfin[cols].sum(axis=1, skipna=False)

    def safe_dummy(series, true_value):
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        m = x.notna()
        out.loc[m] = (x.loc[m] == true_value).astype(float)
        return out

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

    def standardized_betas(fit, y, X_no_const):
        """
        Post-estimation standardized slopes:
            beta_j = b_j * SD(X_j) / SD(Y)
        computed on analytic sample.
        """
        y = to_num(y)
        y_sd = float(y.std(ddof=0))
        betas = {}
        for col in X_no_const.columns:
            b = float(fit.params.get(col, np.nan))
            x_sd = float(to_num(X_no_const[col]).std(ddof=0))
            if np.isnan(b) or np.isnan(y_sd) or y_sd == 0 or np.isnan(x_sd) or x_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = b * (x_sd / y_sd)
        return pd.Series(betas, name="Std_Beta")

    def fmt_float(x, nd=3):
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    def df_to_pretty_string(dfin, nd=3):
        d = dfin.copy()
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].map(lambda v: fmt_float(v, nd))
        return d.to_string()

    def drop_nonvarying_predictors(d, predictors):
        """
        After listwise deletion, drop predictors that have <=1 unique non-missing value.
        This prevents runtime errors and matches common practice when a dummy collapses.
        """
        kept, dropped = [], []
        for p in predictors:
            s = d[p]
            nun = s.nunique(dropna=True)
            if nun <= 1:
                dropped.append(p)
            else:
                kept.append(p)
        return kept, dropped

    # -----------------------------
    # Variable mapping (per instructions)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_required = ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
    required = base_required + minority_genres + remaining_genres + racism_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # numeric coercion for required numeric fields
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables (strict dislike counts)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_dislike_minority_linked_6"] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_dislike_remaining_12"] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0–5, strict sum of 5 dichotomies)
    # Direction per mapping:
    #   RACHAF: 1=yes object -> 1; 2=no -> 0
    #   BUSING: 2=oppose -> 1; 1=favor -> 0
    #   RACDIF1: 2=no (not mainly due to discrimination) -> 1; 1=yes -> 0
    #   RACDIF3: 2=no -> 1; 1=yes -> 0
    #   RACDIF4: 1=yes (will power) -> 1; 2=no -> 0
    # -----------------------------
    df["r_rachaf"] = dich_item(df["rachaf"], ones=[1], zeros=[2])
    df["r_busing"] = dich_item(df["busing"], ones=[2], zeros=[1])
    df["r_racdif1"] = dich_item(df["racdif1"], ones=[2], zeros=[1])
    df["r_racdif3"] = dich_item(df["racdif3"], ones=[2], zeros=[1])
    df["r_racdif4"] = dich_item(df["racdif4"], ones=[1], zeros=[2])

    df["racism_score"] = strict_sum(df, ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"])

    # -----------------------------
    # Controls / indicators
    # -----------------------------
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["female"] = safe_dummy(df["sex"], 2)

    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic not available in this extract: set to 0 where race is known (prevents full-case dropping)
    df["hispanic"] = np.where(race_known, 0.0, np.nan)

    # Conservative Protestant proxy: RELIG==1 (protestant) & DENOM==1 (baptist)
    df["cons_prot"] = np.nan
    rel_denom_known = df["relig"].notna() & df["denom"].notna()
    df.loc[rel_denom_known, "cons_prot"] = (
        (df.loc[rel_denom_known, "relig"] == 1) & (df.loc[rel_denom_known, "denom"] == 1)
    ).astype(float)

    df["no_religion"] = safe_dummy(df["relig"], 4)
    df["southern"] = safe_dummy(df["region"], 3)

    predictors_full = [
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

    table_labels = {
        "racism_score": "Racism score",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic (not in extract; set to 0 when race observed)",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy: Protestant & Baptist)",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    # -----------------------------
    # Fit + output
    # -----------------------------
    def fit_model(dv_col, model_name, dv_label):
        cols = [dv_col] + predictors_full
        d = df[cols].copy()
        d = d.dropna().copy()  # listwise deletion

        if d.shape[0] < 20:
            raise ValueError(
                f"Too few complete cases after listwise deletion for {model_name}: N={d.shape[0]}. "
                f"Check missingness and coding."
            )

        kept, dropped = drop_nonvarying_predictors(d, predictors_full)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        fit = sm.OLS(y, Xc).fit()

        betas = standardized_betas(fit, y, X).reindex(kept)
        pvals = fit.pvalues.reindex(["const"] + kept)

        out = pd.DataFrame(
            {
                "Std_Beta": betas,
                "Sig": [star_from_p(pvals.get(p, np.nan)) for p in kept],
            },
            index=[table_labels.get(p, p) for p in kept],
        )

        fit_stats = pd.DataFrame(
            {
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Sig": [star_from_p(pvals.get("const", np.nan))],
            },
            index=[model_name],
        )

        dropped_df = pd.DataFrame(
            {"Dropped_predictor": dropped, "Reason": ["No variation after listwise deletion"] * len(dropped)}
        )

        # Text outputs
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {dv_label}")
        lines.append("Model: OLS; reported coefficients are standardized betas (beta weights).")
        lines.append("Standardization: beta_j = b_j * SD(X_j)/SD(Y), computed on this model's analytic sample.")
        lines.append("Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing.")
        lines.append("DV construction: strict count; DV missing if any component genre rating missing.")
        lines.append("Racism score: strict sum of 5 dichotomies; missing if any component missing.")
        lines.append("Missing data: listwise deletion on DV + all (attempted) predictors.")
        lines.append("Stars: two-tailed p-values on the unstandardized OLS coefficients (* p<.05, ** p<.01, *** p<.001).")
        lines.append("")
        if dropped:
            lines.append("NOTE: Some predictors were dropped because they had no variation after listwise deletion:")
            for p in dropped:
                lines.append(f"- {p}")
            lines.append("")
        lines.append("Standardized coefficients")
        lines.append("-------------------------")
        tmp = out.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt_float(v, 3))
        lines.append(tmp.to_string())
        lines.append("")
        lines.append("Fit statistics")
        lines.append("--------------")
        fs = fit_stats.copy()
        fs["N"] = fs["N"].map(lambda v: fmt_float(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt_float(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt_float(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt_float(v, 3))
        lines.append(fs.to_string())
        lines.append("")

        with open(f"./output/{model_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        with open(f"./output/{model_name}_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        out.to_csv(f"./output/{model_name}_table.csv", index=True)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)
        if dropped:
            dropped_df.to_csv(f"./output/{model_name}_dropped_predictors.csv", index=False)

        return out, fit_stats, dropped_df

    m1_table, m1_fit, m1_dropped = fit_model(
        "dv1_dislike_minority_linked_6",
        "Table2_ModelA_Dislike_MinorityLinked_6",
        "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
    )
    m2_table, m2_fit, m2_dropped = fit_model(
        "dv2_dislike_remaining_12",
        "Table2_ModelB_Dislike_Remaining_12",
        "Dislike of the 12 Remaining Genres (count of 12)",
    )

    combined = pd.concat(
        [
            m1_table.rename(columns={"Std_Beta": "ModelA_Std_Beta", "Sig": "ModelA_Sig"}),
            m2_table.rename(columns={"Std_Beta": "ModelB_Std_Beta", "Sig": "ModelB_Sig"}),
        ],
        axis=1,
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # Diagnostics: DV descriptives and missingness shares (pre-listwise)
    dv_desc = df[["dv1_dislike_minority_linked_6", "dv2_dislike_remaining_12"]].describe()

    key_cols_1 = ["dv1_dislike_minority_linked_6"] + predictors_full
    key_cols_2 = ["dv2_dislike_remaining_12"] + predictors_full
    miss_1 = df[key_cols_1].isna().mean().sort_values(ascending=False).to_frame("share_missing")
    miss_2 = df[key_cols_2].isna().mean().sort_values(ascending=False).to_frame("share_missing")

    item_missing_music = pd.DataFrame(
        {
            "missing_share_minority_genres": df[[f"d_{c}" for c in minority_genres]].isna().mean(),
            "missing_share_remaining_genres": df[[f"d_{c}" for c in remaining_genres]].isna().mean(),
        }
    )
    item_missing_racism = df[["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]].isna().mean().to_frame(
        "missing_share"
    )

    # Combined summary text
    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt using provided 1993 GSS extract\n")
        f.write("=======================================================================\n\n")

        f.write("Combined standardized coefficients (betas) and significance stars\n")
        f.write("----------------------------------------------------------------\n")
        f.write(df_to_pretty_string(combined, nd=3))
        f.write("\n\n")

        f.write("Fit statistics\n")
        f.write("--------------\n")
        f.write(df_to_pretty_string(combined_fit, nd=3))
        f.write("\n\n")

        f.write("DV descriptives (constructed counts; before listwise deletion)\n")
        f.write("-------------------------------------------------------------\n")
        f.write(df_to_pretty_string(dv_desc, nd=3))
        f.write("\n\n")

        f.write("Missingness shares (Model A variables; before listwise)\n")
        f.write("------------------------------------------------------\n")
        f.write(df_to_pretty_string(miss_1, nd=3))
        f.write("\n\n")

        f.write("Missingness shares (Model B variables; before listwise)\n")
        f.write("------------------------------------------------------\n")
        f.write(df_to_pretty_string(miss_2, nd=3))
        f.write("\n\n")

        f.write("Per-item missingness (music dislike indicators)\n")
        f.write("---------------------------------------------\n")
        f.write(df_to_pretty_string(item_missing_music, nd=3))
        f.write("\n\n")

        f.write("Per-item missingness (racism-score components)\n")
        f.write("--------------------------------------------\n")
        f.write(df_to_pretty_string(item_missing_racism, nd=3))
        f.write("\n\n")

        f.write("Dropped predictors (if any) due to no variation after listwise deletion\n")
        f.write("-----------------------------------------------------------------------\n")
        all_dropped = pd.concat(
            [
                m1_dropped.assign(Model="ModelA"),
                m2_dropped.assign(Model="ModelB"),
            ],
            axis=0,
            ignore_index=True,
        )
        if all_dropped.shape[0] == 0:
            f.write("(none)\n")
        else:
            f.write(all_dropped.to_string(index=False))
            f.write("\n")

        f.write("\nNotes\n-----\n")
        f.write("- Hispanic indicator is not present in this extract; it is set to 0 when RACE is observed.\n")
        f.write("- Some dummies can become constant after listwise deletion; they are dropped to avoid runtime failure.\n")
        f.write("- If resulting N is much smaller than published, the missingness tables above indicate which items drive it.\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=True)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=True)
    miss_1.to_csv("./output/missingness_modelA.csv", index=True)
    miss_2.to_csv("./output/missingness_modelB.csv", index=True)
    item_missing_music.to_csv("./output/item_missingness_music.csv", index=True)
    item_missing_racism.to_csv("./output/item_missingness_racism.csv", index=True)

    return {
        "table2_betas": combined,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_modelA": miss_1,
        "missingness_modelB": miss_2,
        "item_missingness_music": item_missing_music,
        "item_missingness_racism": item_missing_racism,
    }