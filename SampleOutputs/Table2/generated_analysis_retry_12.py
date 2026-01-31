def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + normalize columns
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

    def recode_1to5_like_dislike(series):
        """
        Music battery: expected 1..5.
        We follow the mapping instruction:
          dislike = 1 if 4 or 5
          dislike = 0 if 1,2,3
          else missing
        """
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        """Map to {0,1}; anything else missing."""
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(zeros)] = 0.0
        out.loc[x.isin(ones)] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

    def dummy_eq(series, value):
        """Binary indicator; missing if input missing."""
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        m = x.notna()
        out.loc[m] = (x.loc[m] == value).astype(float)
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

    def zscore(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def format_float(x, nd=3):
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

    def labelled_missingness(dfin, cols, labels):
        miss = dfin[cols].isna().mean()
        out = pd.DataFrame(
            {"variable": cols, "label": [labels.get(c, c) for c in cols], "share_missing": miss.values}
        ).sort_values("share_missing", ascending=False)
        return out

    # -----------------------------
    # Required fields
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Coerce numerics (leave id as-is)
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables (strict dislike counts)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = recode_1to5_like_dislike(df[c])

    df["dv1_minority6_dislikes"] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_remaining12_dislikes"] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5) per mapping (strict sum)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1 yes object
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2 oppose
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2 no discrimination
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2 no education chance
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1 yes willpower

    df["racism_score"] = strict_sum(df, ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"])

    # -----------------------------
    # Controls / indicators
    # -----------------------------
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["female"] = dummy_eq(df["sex"], 2)

    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic indicator not available in this extract; follow mapping note:
    # keep it present but set to 0 where race is known (prevents listwise deletion due solely to missing field).
    df["hispanic"] = np.where(race_known, 0.0, np.nan)

    # Conservative Protestant proxy: Protestant & Baptist
    rel_denom_known = df["relig"].notna() & df["denom"].notna()
    df["cons_prot"] = np.nan
    df.loc[rel_denom_known, "cons_prot"] = (
        (df.loc[rel_denom_known, "relig"] == 1) & (df.loc[rel_denom_known, "denom"] == 1)
    ).astype(float)

    df["no_religion"] = dummy_eq(df["relig"], 4)
    df["southern"] = dummy_eq(df["region"], 3)

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

    labels = {
        "racism_score": "Racism score (0–5)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (not in extract; set 0 if race observed)",
        "other_race": "Other race (1=other)",
        "cons_prot": "Conservative Protestant (proxy: Protestant & Baptist)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Modeling: OLS on unstandardized DV, report standardized betas
    # Standardized betas computed by fitting on z-scored Y and z-scored X (no intercept),
    # which yields beta weights directly and avoids NaN beta conversion artifacts.
    # Intercept/fit stats from unstandardized model (as in paper tables).
    # -----------------------------
    def fit_table2_models(dv_col, model_name, dv_label):
        cols = [dv_col] + predictors
        d = df[cols].copy()

        # Listwise deletion for this model only
        d = d.dropna().copy()

        # Drop any predictors that have no variance after listwise deletion (prevents singular matrices)
        kept = []
        dropped = []
        for p in predictors:
            nun = d[p].nunique(dropna=True)
            if nun <= 1:
                dropped.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)

        # Unstandardized model (for intercept, R2, p-values)
        X_unstd = d[kept].astype(float)
        Xc = sm.add_constant(X_unstd, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas via regression on standardized variables without intercept
        y_z = zscore(y)
        X_z = pd.DataFrame({p: zscore(d[p]) for p in kept})
        # If any zscore became missing due to zero SD (shouldn't given variance check), drop listwise again
        dz = pd.concat([y_z.rename("y_z"), X_z], axis=1).dropna()
        y_z2 = dz["y_z"].astype(float)
        X_z2 = dz[kept].astype(float)

        fit_beta = sm.OLS(y_z2, X_z2).fit()
        betas = fit_beta.params.reindex(kept)

        # Stars based on p-values from unstandardized model (common for table stars)
        pvals = fit_unstd.pvalues.reindex(["const"] + kept)
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        # Output table (keep exact predictor order)
        out_rows = []
        for p in predictors:
            if p in kept:
                out_rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": stars.get(p, ""),
                    }
                )
            else:
                out_rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Sig": "",
                    }
                )
        table = pd.DataFrame(out_rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [dv_label],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [const_star],
                "Dropped_predictors_no_variation": [", ".join(dropped) if dropped else ""],
            },
            index=[model_name],
        )

        # Human-readable text output
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {dv_label}")
        lines.append("Model: OLS.")
        lines.append("Reported coefficients: standardized OLS coefficients (beta weights).")
        lines.append("How betas are computed here: regress z(Y) on z(X) with no intercept on the analytic sample.")
        lines.append("Significance stars: from two-tailed p-values of the unstandardized OLS regression.")
        lines.append("Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing.")
        lines.append("DV construction: strict sum (missing if any component genre rating missing).")
        lines.append("Racism score: strict sum of 5 dichotomies (missing if any component missing).")
        lines.append("Missing data: listwise deletion on DV and all included predictors for this model.")
        if dropped:
            lines.append("")
            lines.append("Dropped predictors due to no variation after listwise deletion:")
            for p in dropped:
                lines.append(f"- {p}")
        lines.append("")
        lines.append("Standardized coefficients (Table 2 style)")
        lines.append("---------------------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: format_float(v, 3))
        lines.append(tmp.to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Sig"]].copy()
        fs["N"] = fs["N"].map(lambda v: format_float(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: format_float(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: format_float(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: format_float(v, 3))
        lines.append(fs.to_string())

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

        with open(f"./output/{model_name}_ols_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(fit_unstd.summary().as_text())
            f.write("\n\n")
            f.write("Standardized-beta regression (zY on zX, no intercept) diagnostics:\n")
            f.write(fit_beta.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        return table, fit_stats

    m1_table, m1_fit = fit_table2_models(
        "dv1_minority6_dislikes",
        "Table2_ModelA_MinorityLinked6",
        "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
    )
    m2_table, m2_fit = fit_table2_models(
        "dv2_remaining12_dislikes",
        "Table2_ModelB_Remaining12",
        "Dislike of the 12 Remaining Genres (count of 12)",
    )

    # -----------------------------
    # Combined outputs
    # -----------------------------
    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "ModelA_Std_Beta": m1_table["Std_Beta"],
            "ModelA_Sig": m1_table["Sig"],
            "ModelB_Std_Beta": m2_table["Std_Beta"],
            "ModelB_Sig": m2_table["Sig"],
        }
    )

    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # Descriptives
    dv_desc = df[["dv1_minority6_dislikes", "dv2_remaining12_dislikes"]].describe()

    # Labelled missingness tables for each model (pre-listwise)
    key_cols_A = ["dv1_minority6_dislikes"] + predictors
    key_cols_B = ["dv2_remaining12_dislikes"] + predictors
    miss_A = labelled_missingness(df, key_cols_A, labels)
    miss_B = labelled_missingness(df, key_cols_B, labels)

    # Per-item missingness for constructed components (clear, non-NaN)
    music_item_missing = pd.DataFrame(
        {
            "item": [f"d_{c}" for c in (minority_genres + remaining_genres)],
            "missing_share": df[[f"d_{c}" for c in (minority_genres + remaining_genres)]].isna().mean().values,
            "group": (["minority_linked_6"] * len(minority_genres)) + (["remaining_12"] * len(remaining_genres)),
        }
    )
    racism_item_missing = pd.DataFrame(
        {
            "item": ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4", "racism_score"],
            "missing_share": df[["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4", "racism_score"]]
            .isna()
            .mean()
            .values,
        }
    )

    # Human-readable combined summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars")
    lines.append("-----------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: format_float(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: format_float(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS)")
    lines.append("---------------------------------")
    fs = combined_fit[["DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig"]].copy()
    fs["N"] = fs["N"].map(lambda v: format_float(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: format_float(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: format_float(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: format_float(v, 3))
    lines.append(fs.to_string())
    lines.append("")
    lines.append("DV descriptives (constructed counts; before listwise deletion)")
    lines.append("-------------------------------------------------------------")
    dv_desc_fmt = dv_desc.copy()
    for c in dv_desc_fmt.columns:
        dv_desc_fmt[c] = dv_desc_fmt[c].map(lambda v: format_float(v, 3))
    lines.append(dv_desc_fmt.to_string())
    lines.append("")
    lines.append("Missingness shares (labelled; Model A variables; before listwise)")
    lines.append("----------------------------------------------------------------")
    missA = miss_A.copy()
    missA["share_missing"] = missA["share_missing"].map(lambda v: format_float(v, 3))
    lines.append(missA.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares (labelled; Model B variables; before listwise)")
    lines.append("----------------------------------------------------------------")
    missB = miss_B.copy()
    missB["share_missing"] = missB["share_missing"].map(lambda v: format_float(v, 3))
    lines.append(missB.to_string(index=False))
    lines.append("")
    lines.append("Per-item missingness: music dislike indicators")
    lines.append("--------------------------------------------")
    mi = music_item_missing.copy()
    mi["missing_share"] = mi["missing_share"].map(lambda v: format_float(v, 3))
    lines.append(mi.to_string(index=False))
    lines.append("")
    lines.append("Per-item missingness: racism components and scale")
    lines.append("------------------------------------------------")
    ri = racism_item_missing.copy()
    ri["missing_share"] = ri["missing_share"].map(lambda v: format_float(v, 3))
    lines.append(ri.to_string(index=False))
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("- Hispanic indicator is not present in this extract; it is set to 0 when RACE is observed to avoid losing cases.")
    lines.append("- Conservative Protestant is proxied as RELIG==1 (Protestant) & DENOM==1 (Baptist) using available fields.")
    lines.append("- If N is far from the published table, the labelled missingness sections show which variables drive case loss.")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    # Save combined tables
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=True)
    miss_A.to_csv("./output/missingness_modelA_labelled.csv", index=False)
    miss_B.to_csv("./output/missingness_modelB_labelled.csv", index=False)
    music_item_missing.to_csv("./output/item_missingness_music.csv", index=False)
    racism_item_missing.to_csv("./output/item_missingness_racism.csv", index=False)

    return {
        "table2_betas": combined,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_modelA": miss_A,
        "missingness_modelB": miss_B,
        "item_missingness_music": music_item_missing,
        "item_missingness_racism": racism_item_missing,
    }