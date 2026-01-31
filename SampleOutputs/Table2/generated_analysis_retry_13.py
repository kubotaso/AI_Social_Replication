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

    def dislike_indicator(series):
        """
        1 if response is 4/5 (dislike/dislike very much),
        0 if response is 1/2/3,
        missing otherwise.
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

    def safe_unique_count(s):
        s = to_num(s)
        return int(s.dropna().nunique())

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
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Numeric coercion (leave id as-is)
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables: strict dislike counts (as instructed)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_minority6_dislikes"] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_remaining12_dislikes"] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5) per mapping
    # IMPORTANT FIX for N collapse: allow partial completion if >= 4 of 5 items answered
    # (common scale-construction approach; reduces attrition vs strict listwise on components)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["_racism_nonmiss"] = df[racism_comp].notna().sum(axis=1)
    df["_racism_sum"] = df[racism_comp].sum(axis=1, skipna=True)

    # Scale score is defined if at least 4 of 5 components present; rescale to 0-5 metric
    # by multiplying by (5 / k) to keep comparable range.
    df["racism_score"] = np.where(
        df["_racism_nonmiss"] >= 4,
        df["_racism_sum"] * (5.0 / df["_racism_nonmiss"]),
        np.nan,
    )
    df.drop(columns=["_racism_nonmiss", "_racism_sum"], inplace=True)

    # -----------------------------
    # Controls / indicators
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = dummy_eq(df["sex"], 2)

    # Race dummies (reference category: White)
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Hispanic: not available -> keep in model as all-0 (not missing) to avoid dropping;
    # it will be dropped for no variation with a clear note in outputs.
    df["hispanic"] = np.where(race_known, 0.0, 0.0)

    # Conservative Protestant proxy (best available): Protestant & Baptist
    rel_denom_known = df["relig"].notna() & df["denom"].notna()
    df["cons_prot"] = 0.0
    df.loc[rel_denom_known, "cons_prot"] = (
        (df.loc[rel_denom_known, "relig"] == 1) & (df.loc[rel_denom_known, "denom"] == 1)
    ).astype(float)
    # If either relig/denom missing, treat as 0 rather than missing to reduce listwise deletion
    # (this is a pragmatic choice given limited extract; documented in outputs).
    df["cons_prot"] = df["cons_prot"].fillna(0.0)

    # No religion (RELIG==4); treat missing as 0 to avoid case loss from item nonresponse
    df["no_religion"] = dummy_eq(df["relig"], 4).fillna(0.0)

    # Southern (REGION==3); treat missing as 0
    df["southern"] = dummy_eq(df["region"], 3).fillna(0.0)

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
        "racism_score": "Racism score (0–5; if >=4/5 items answered, rescaled to 0–5)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (not in extract; constant 0 => dropped for no variation)",
        "other_race": "Other race (1=other)",
        "cons_prot": "Conservative Protestant (proxy: Protestant & Baptist; missing treated as 0)",
        "no_religion": "No religion (RELIG==4; missing treated as 0)",
        "southern": "Southern (REGION==3; missing treated as 0)",
        "const": "Constant",
        "dv1_minority6_dislikes": "Dislike of minority-linked genres (count of 6)",
        "dv2_remaining12_dislikes": "Dislike of remaining genres (count of 12)",
    }

    # -----------------------------
    # Model fitting + table outputs
    # -----------------------------
    def fit_table2_model(dv_col, model_name, dv_label):
        cols = [dv_col] + predictors
        d = df[cols].copy()

        # Listwise deletion on DV and core continuous variables that truly must exist
        # Keep pragmatic handling for dummies already set to 0/1.
        d = d.dropna(subset=[dv_col, "racism_score", "educ", "income_pc", "prestg80", "female", "age", "black", "other_race"]).copy()

        # Drop non-varying predictors (prevents singular matrices)
        kept, dropped = [], []
        for p in predictors:
            nun = d[p].nunique(dropna=True)
            if nun <= 1:
                dropped.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)

        # Unstandardized OLS for fit stats and p-values (for stars)
        X_unstd = d[kept].astype(float)
        Xc = sm.add_constant(X_unstd, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas via z(Y) on z(X) without intercept (beta weights)
        y_z = zscore(y)
        X_z = pd.DataFrame({p: zscore(d[p]) for p in kept})
        dz = pd.concat([y_z.rename("y_z"), X_z], axis=1).dropna()
        y_z2 = dz["y_z"].astype(float)
        X_z2 = dz[kept].astype(float)
        fit_beta = sm.OLS(y_z2, X_z2).fit()
        betas = fit_beta.params.reindex(kept)

        pvals = fit_unstd.pvalues.reindex(["const"] + kept)
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        # Build table with all predictors in original order
        rows = []
        for p in predictors:
            if p in kept:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": float(betas.get(p, np.nan)), "Sig": stars.get(p, "")})
            else:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": ""})
        table = pd.DataFrame(rows)

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

        # Human-readable output
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {dv_label}")
        lines.append("Model: OLS.")
        lines.append("Reported coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Betas computed as: regress z(Y) on z(X) (no intercept) on the analytic sample.")
        lines.append("Significance stars: two-tailed p-values from the unstandardized OLS regression.")
        lines.append("Dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing.")
        lines.append("DV construction: strict sum across component genres (missing if any component missing).")
        lines.append("Racism scale: 5 dichotomies; if >=4 answered, sum is rescaled to 0–5; else missing.")
        lines.append("Missing data: listwise on DV + core predictors; some dummies are set to 0 when underlying field missing (documented in labels).")
        if dropped:
            lines.append("")
            lines.append("Dropped predictors due to no variation in analytic sample:")
            for p in dropped:
                lines.append(f"- {p}: {labels.get(p, p)}")
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
            f.write("Unstandardized OLS (for fit stats + p-values):\n")
            f.write(fit_unstd.summary().as_text())
            f.write("\n\nStandardized-beta regression (zY on zX, no intercept):\n")
            f.write(fit_beta.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Analytic sample diagnostics
        diag = pd.DataFrame(
            {
                "variable": [dv_col] + predictors,
                "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors],
                "share_missing_in_raw_model_frame": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors],
                "unique_values_in_analytic_sample": [safe_unique_count(d[dv_col])] + [safe_unique_count(d[p]) for p in predictors],
            }
        )
        diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)

        return table, fit_stats, diag

    m1_table, m1_fit, m1_diag = fit_table2_model(
        "dv1_minority6_dislikes",
        "Table2_ModelA_MinorityLinked6",
        labels["dv1_minority6_dislikes"],
    )
    m2_table, m2_fit, m2_diag = fit_table2_model(
        "dv2_remaining12_dislikes",
        "Table2_ModelB_Remaining12",
        labels["dv2_remaining12_dislikes"],
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

    # Clean DV descriptives (explicit labels)
    dv_desc = pd.DataFrame(
        {
            "stat": ["N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"],
            "DV1_Minority6": [
                int(df["dv1_minority6_dislikes"].notna().sum()),
                df["dv1_minority6_dislikes"].mean(),
                df["dv1_minority6_dislikes"].std(ddof=0),
                df["dv1_minority6_dislikes"].min(),
                df["dv1_minority6_dislikes"].quantile(0.25),
                df["dv1_minority6_dislikes"].quantile(0.50),
                df["dv1_minority6_dislikes"].quantile(0.75),
                df["dv1_minority6_dislikes"].max(),
            ],
            "DV2_Remaining12": [
                int(df["dv2_remaining12_dislikes"].notna().sum()),
                df["dv2_remaining12_dislikes"].mean(),
                df["dv2_remaining12_dislikes"].std(ddof=0),
                df["dv2_remaining12_dislikes"].min(),
                df["dv2_remaining12_dislikes"].quantile(0.25),
                df["dv2_remaining12_dislikes"].quantile(0.50),
                df["dv2_remaining12_dislikes"].quantile(0.75),
                df["dv2_remaining12_dislikes"].max(),
            ],
        }
    )

    # Missingness tables (pre-model)
    key_cols_A = ["dv1_minority6_dislikes"] + predictors
    key_cols_B = ["dv2_remaining12_dislikes"] + predictors
    miss_A = labelled_missingness(df, key_cols_A, labels)
    miss_B = labelled_missingness(df, key_cols_B, labels)

    # Per-item missingness (music + racism)
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
    fs = combined_fit[["DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_predictors_no_variation"]].copy()
    fs["N"] = fs["N"].map(lambda v: format_float(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: format_float(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: format_float(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: format_float(v, 3))
    lines.append(fs.to_string())
    lines.append("")
    lines.append("DV descriptives (constructed counts; before listwise deletion)")
    lines.append("-------------------------------------------------------------")
    dv_desc_fmt = dv_desc.copy()
    for c in ["DV1_Minority6", "DV2_Remaining12"]:
        dv_desc_fmt[c] = dv_desc_fmt[c].map(lambda v: format_float(v, 3) if not isinstance(v, (int, np.integer)) else f"{int(v)}")
    lines.append(dv_desc_fmt.to_string(index=False))
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
    lines.append("- This extract lacks a dedicated Hispanic indicator; the model retains a Hispanic term as a constant 0, which is dropped for no variation.")
    lines.append("- Conservative Protestant is approximated with RELIG==1 & DENOM==1; missing RELIG/DENOM is treated as 0 to reduce attrition.")
    lines.append("- No religion and Southern dummies treat missing underlying fields as 0 to reduce attrition; this may differ from the paper's handling.")
    lines.append("- Racism score uses a partial-completion rule (>=4/5 items) to reduce case loss; strict 5/5 would shrink N substantially.")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    # Save combined artifacts
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)
    miss_A.to_csv("./output/missingness_modelA_labelled.csv", index=False)
    miss_B.to_csv("./output/missingness_modelB_labelled.csv", index=False)
    music_item_missing.to_csv("./output/item_missingness_music.csv", index=False)
    racism_item_missing.to_csv("./output/item_missingness_racism.csv", index=False)
    m1_diag.to_csv("./output/Table2_ModelA_MinorityLinked6_analytic_diagnostics.csv", index=False)
    m2_diag.to_csv("./output/Table2_ModelB_Remaining12_analytic_diagnostics.csv", index=False)

    return {
        "table2_betas": combined,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_modelA": miss_A,
        "missingness_modelB": miss_B,
        "item_missingness_music": music_item_missing,
        "item_missingness_racism": racism_item_missing,
        "modelA_analytic_diagnostics": m1_diag,
        "modelB_analytic_diagnostics": m2_diag,
    }