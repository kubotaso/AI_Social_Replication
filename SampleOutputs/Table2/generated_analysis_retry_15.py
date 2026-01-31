def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load data
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
        """1 if 4/5, 0 if 1/2/3, else missing."""
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

    def zscore(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

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

    base_required = (
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "region"]
        + minority_genres
        + remaining_genres
        + racism_items
    )
    missing_cols = [c for c in base_required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Optional fields for improved approximation (if present)
    # Hispanic: try direct flag, else try a conservative proxy from ETHNIC if available.
    hisp_direct_candidates = [c for c in ["hispanic", "hispanicx", "hispan", "hisp"] if c in df.columns]
    ethnic_available = "ethnic" in df.columns
    # Conservative Protestant: prefer denom16 if present else denom
    denom_var = "denom16" if "denom16" in df.columns else ("denom" if "denom" in df.columns else None)

    # Coerce numerics (except id)
    for c in set(base_required + hisp_direct_candidates + (["ethnic"] if ethnic_available else []) + ([denom_var] if denom_var else [])):
        if c != "id" and c in df.columns:
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables: strict dislike counts (missing if any item missing)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_minority6_dislikes"] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_remaining12_dislikes"] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5) strict sum of 5 dichotomies (missing if any missing)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls / indicators (retain missing as missing; listwise deletion later)
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female (SEX==2)
    df["female"] = dummy_eq(df["sex"], 2)

    # Race dummies (reference: White); treat unknown as missing
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic indicator: best-effort, but NEVER force missing to 0.
    used_hisp_source = None
    df["hispanic"] = np.nan
    for c in hisp_direct_candidates:
        x = to_num(df[c])
        # accept 0/1, or map 1=yes 2=no
        if x.dropna().isin([0, 1]).all() and x.notna().sum() > 0:
            df["hispanic"] = x
            used_hisp_source = c
            break
        cand = pd.Series(np.nan, index=df.index, dtype="float64")
        cand.loc[x.isin([2])] = 0.0
        cand.loc[x.isin([1])] = 1.0
        if cand.notna().sum() > 0 and cand.nunique(dropna=True) > 1:
            df["hispanic"] = cand
            used_hisp_source = c
            break

    if used_hisp_source is None and ethnic_available:
        # Conservative proxy: mark as Hispanic only when ETHNIC matches common "Hispanic/Spanish" codes.
        # Without a codebook, we keep this conservative and leave other codes as 0 when clearly non-Hispanic.
        x = to_num(df["ethnic"])
        hisp_set = {8, 9, 10, 11, 12, 13, 14, 15}
        cand = pd.Series(np.nan, index=df.index, dtype="float64")
        cand.loc[x.isin(list(hisp_set))] = 1.0
        cand.loc[x.notna() & (~x.isin(list(hisp_set)))] = 0.0
        if cand.nunique(dropna=True) > 1:
            df["hispanic"] = cand
            used_hisp_source = "ethnic(proxy_codes_8_15)"

    # Conservative Protestant proxy (limited by available variables; keep missing as missing)
    used_consprot_source = None
    df["cons_prot"] = np.nan
    if denom_var is not None and "relig" in df.columns:
        rel = to_num(df["relig"])
        denom = to_num(df[denom_var])
        # per mapping: Protestant (RELIG==1) & Baptist (DENOM==1)
        cand = pd.Series(np.nan, index=df.index, dtype="float64")
        ok = rel.notna() & denom.notna()
        cand.loc[ok] = ((rel.loc[ok] == 1) & (denom.loc[ok] == 1)).astype(float)
        df["cons_prot"] = cand
        used_consprot_source = f"relig+{denom_var}"

    # No religion (RELIG==4)
    df["no_religion"] = dummy_eq(df["relig"], 4)

    # Southern (REGION==3)
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
        "dv1_minority6_dislikes": "Dislike of minority-linked genres (count of 6)",
        "dv2_remaining12_dislikes": "Dislike of remaining genres (count of 12)",
        "racism_score": "Racism score (0–5; strict sum of 5 dichotomies)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": f"Hispanic (constructed; source={used_hisp_source or 'UNAVAILABLE'})",
        "other_race": "Other race (1=other)",
        "cons_prot": f"Conservative Protestant (proxy; source={used_consprot_source or 'UNAVAILABLE'})",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting with robust empty-sample protection
    # -----------------------------
    def fit_table2_model(dv_col, model_name):
        cols = [dv_col] + predictors
        d = df[cols].copy()

        # Strict listwise deletion on full model frame (as close as possible to paper)
        d = d.dropna(subset=cols).copy()

        # Protect against empty analytic sample (previous runtime error)
        if len(d) == 0:
            # Write minimal diagnostic
            msg = [
                model_name,
                "=" * len(model_name),
                "",
                f"ERROR: Analytic sample is empty after listwise deletion for DV={dv_col}.",
                "This usually means one or more required predictors are entirely missing in the extract.",
                "",
                "Missingness in 1993 (share missing):",
            ]
            miss = labelled_missingness(df, cols, labels)
            miss_fmt = miss.copy()
            miss_fmt["share_missing"] = miss_fmt["share_missing"].map(lambda v: format_float(v, 3))
            msg.append(miss_fmt.to_string(index=False))
            with open(f"./output/{model_name}_ERROR.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(msg) + "\n")

            empty_table = pd.DataFrame(
                {"Independent Variable": [labels.get(p, p) for p in predictors], "Std_Beta": np.nan, "Sig": ""}
            )
            empty_fit = pd.DataFrame(
                {"DV": [labels.get(dv_col, dv_col)], "N": [0], "R2": [np.nan], "Adj_R2": [np.nan], "Constant": [np.nan]},
                index=[model_name],
            )
            return empty_table, empty_fit, d

        # Drop non-varying predictors (avoid singular matrices)
        kept, dropped = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped.append(p)
            else:
                kept.append(p)

        # If all predictors dropped, we can only fit intercept; still avoid crash.
        y = d[dv_col].astype(float)

        # Unstandardized OLS for p-values/stars and intercept
        if len(kept) > 0:
            X_unstd = d[kept].astype(float)
            Xc = sm.add_constant(X_unstd, has_constant="add")
        else:
            Xc = pd.DataFrame({"const": np.ones(len(d), dtype=float)}, index=d.index)

        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas via z(Y) on z(X) without intercept
        betas = pd.Series(index=kept, dtype="float64")
        if len(kept) > 0:
            y_z = zscore(y)
            X_z = pd.DataFrame({p: zscore(d[p]) for p in kept}, index=d.index)
            dz = pd.concat([y_z.rename("y_z"), X_z], axis=1).dropna()
            if len(dz) > 0 and all(dz[p].std(ddof=0) > 0 for p in kept) and dz["y_z"].std(ddof=0) > 0:
                fit_beta = sm.OLS(dz["y_z"].astype(float), dz[kept].astype(float)).fit()
                betas = fit_beta.params.reindex(kept)
            else:
                fit_beta = None
        else:
            fit_beta = None

        # Stars from unstandardized p-values (replication stars; not paper's if model differs)
        pvals = fit_unstd.pvalues if hasattr(fit_unstd, "pvalues") else pd.Series(dtype=float)
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        # Build table in the paper's predictor order
        rows = []
        for p in predictors:
            if p in kept:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": float(betas.get(p, np.nan)), "Sig": stars.get(p, "")})
            else:
                # keep row but blank beta; note dropped predictors later
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": ""})
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared) if hasattr(fit_unstd, "rsquared") else np.nan],
                "Adj_R2": [float(fit_unstd.rsquared_adj) if hasattr(fit_unstd, "rsquared_adj") else np.nan],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [const_star],
                "Dropped_predictors_no_variation": [", ".join(dropped) if dropped else ""],
            },
            index=[model_name],
        )

        # Write human-readable file
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS. Reported coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: regress z(Y) on z(X) with no intercept, on the analytic (listwise) sample.")
        lines.append("Stars: two-tailed p-values from the unstandardized OLS regression on the same sample.")
        lines.append("")
        lines.append("Construction notes:")
        lines.append("- Dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV construction: strict sum across component genres (missing if any component missing).")
        lines.append("- Racism scale: strict sum of 5 dichotomies (missing if any component missing).")
        lines.append("- Missing data: strict listwise deletion on DV and all predictors in the model frame.")
        lines.append(f"- Hispanic variable source: {used_hisp_source or 'UNAVAILABLE (will reduce N / may differ from paper)'}")
        lines.append(f"- Conservative Protestant source: {used_consprot_source or 'UNAVAILABLE (will reduce N / may differ from paper)'}")
        if dropped:
            lines.append("")
            lines.append("Dropped predictors due to no variation in analytic sample:")
            for p in dropped:
                lines.append(f"- {p}: {labels.get(p, p)}")
        lines.append("")
        lines.append("Standardized coefficients (Table-2 style)")
        lines.append("---------------------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: format_float(v, 3))
        lines.append(tmp.to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_predictors_no_variation"]].copy()
        fs["N"] = fs["N"].map(lambda v: format_float(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: format_float(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: format_float(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: format_float(v, 3))
        lines.append(fs.to_string())

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Diagnostics: store the model summaries too
        with open(f"./output/{model_name}_ols_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write("Unstandardized OLS (fit stats + p-values):\n")
            f.write(fit_unstd.summary().as_text())
            f.write("\n\n")
            if fit_beta is not None:
                f.write("Standardized-beta regression (zY on zX, no intercept):\n")
                f.write(fit_beta.summary().as_text())
                f.write("\n")
            else:
                f.write("Standardized-beta regression not estimated (insufficient variation after standardization).\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Analytic sample frequency checks
        freq_cols = ["black", "hispanic", "other_race", "female", "cons_prot", "no_religion", "southern"]
        freq_frames = []
        for c in freq_cols:
            vc = d[c].value_counts(dropna=False).rename_axis("value").reset_index(name="n")
            vc.insert(0, "variable", c)
            freq_frames.append(vc)
        freq_df = pd.concat(freq_frames, ignore_index=True)
        freq_df.to_csv(f"./output/{model_name}_analytic_sample_frequencies.csv", index=False)

        # Analytic sample diagnostics
        diag = pd.DataFrame(
            {
                "variable": [dv_col] + predictors,
                "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors],
                "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors],
                "unique_values_in_analytic_sample": [safe_unique_count(d[dv_col])] + [safe_unique_count(d[p]) for p in predictors],
            }
        )
        diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)

        return table, fit_stats, d

    m1_table, m1_fit, m1_frame = fit_table2_model("dv1_minority6_dislikes", "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame = fit_table2_model("dv2_remaining12_dislikes", "Table2_ModelB_Remaining12")

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

    # DVs descriptives (before listwise)
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

    miss_A = labelled_missingness(df, ["dv1_minority6_dislikes"] + predictors, labels)
    miss_B = labelled_missingness(df, ["dv2_remaining12_dislikes"] + predictors, labels)

    # Write combined summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Implementation (key points)")
    lines.append("---------------------------")
    lines.append("- Year filter: year==1993")
    lines.append("- DVs: strict dislike counts; missing if any genre item missing within DV set.")
    lines.append("- Racism score: strict sum of 5 dichotomies; missing if any component missing.")
    lines.append("- Missing data: listwise deletion within each model on DV + all predictors.")
    lines.append(f"- Hispanic construction source: {used_hisp_source or 'UNAVAILABLE'}")
    lines.append(f"- Conservative Protestant construction source: {used_consprot_source or 'UNAVAILABLE'}")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and stars")
    lines.append("----------------------------------------------------------")
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
    lines.append("DV descriptives (before listwise deletion)")
    lines.append("----------------------------------------")
    dv_desc_fmt = dv_desc.copy()
    for c in ["DV1_Minority6", "DV2_Remaining12"]:
        dv_desc_fmt[c] = dv_desc_fmt[c].map(lambda v: str(int(v)) if isinstance(v, (int, np.integer)) else format_float(v, 3))
    lines.append(dv_desc_fmt.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares (Model A variables; before listwise)")
    lines.append("------------------------------------------------------")
    missA = miss_A.copy()
    missA["share_missing"] = missA["share_missing"].map(lambda v: format_float(v, 3))
    lines.append(missA.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares (Model B variables; before listwise)")
    lines.append("------------------------------------------------------")
    missB = miss_B.copy()
    missB["share_missing"] = missB["share_missing"].map(lambda v: format_float(v, 3))
    lines.append(missB.to_string(index=False))

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)
    miss_A.to_csv("./output/missingness_modelA_labelled.csv", index=False)
    miss_B.to_csv("./output/missingness_modelB_labelled.csv", index=False)

    return {
        "table2_betas": combined,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_modelA": miss_A,
        "missingness_modelB": miss_B,
    }