def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

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

    def safe_unique_count(s):
        s = to_num(s)
        return int(s.dropna().nunique())

    def labelled_missingness(dfin, cols, labels):
        miss = dfin[cols].isna().mean()
        out = pd.DataFrame(
            {"variable": cols, "label": [labels.get(c, c) for c in cols], "share_missing": miss.values}
        ).sort_values("share_missing", ascending=False)
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

    base_required = [
        "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "region"
    ] + minority_genres + remaining_genres + racism_items

    # Conservative Protestant: allow multiple possible sources (broad denom, or detailed denom16)
    denom_candidates = [c for c in ["denom16", "denom"] if c in df.columns]

    # Hispanic: allow multiple possible sources; fall back to 'ethnic' as a best-effort mapping
    hisp_candidates = [c for c in ["hispanic", "hispanicx", "hispan", "hisp"] if c in df.columns]
    if "ethnic" in df.columns:
        hisp_candidates.append("ethnic")

    required = list(dict.fromkeys(base_required + denom_candidates + hisp_candidates))
    missing_cols = [c for c in base_required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # numeric coercion (leave id as-is)
    for c in required:
        if c != "id" and c in df.columns:
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables: strict dislike counts
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    df["dv1_minority6_dislikes"] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df["dv2_remaining12_dislikes"] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5): STRICT complete-case per mapping
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = dummy_eq(df["sex"], 2)

    # Age
    df["age"] = to_num(df["age"])

    # Race dummies (reference: White)
    df["black"] = np.where(df["race"].notna(), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].notna(), (df["race"] == 3).astype(float), np.nan)

    # Hispanic indicator (best effort):
    # - if a direct hispanic variable exists and is 0/1 -> use it
    # - else if 'ethnic' exists: treat certain origin codes as Hispanic when identifiable
    #   (cannot perfectly replicate without the true GSS hispanic flag; we do NOT force to 0)
    df["hispanic"] = np.nan
    used_hisp_source = None
    for c in [cc for cc in ["hispanic", "hispanicx", "hispan", "hisp"] if cc in df.columns]:
        # If it's already 0/1, accept; else try common encodings (1=yes, 2=no)
        x = to_num(df[c])
        if x.dropna().isin([0, 1]).all():
            df["hispanic"] = x
            used_hisp_source = c
            break
        # common yes/no
        cand = pd.Series(np.nan, index=df.index, dtype="float64")
        cand.loc[x.isin([2])] = 0.0
        cand.loc[x.isin([1])] = 1.0
        if cand.notna().sum() > 0 and cand.nunique(dropna=True) > 1:
            df["hispanic"] = cand
            used_hisp_source = c
            break

    if used_hisp_source is None and "ethnic" in df.columns:
        x = to_num(df["ethnic"])
        # Heuristic: in many GSS extracts, ETHNIC codes include Spanish/Hispanic origin codes.
        # We only classify when we can identify a plausible Hispanic set; otherwise leave missing.
        # Commonly Spanish-origin categories often appear as small integers near 1..10; however,
        # without a codebook we avoid overreach: only mark as Hispanic when code is in a
        # conservative set frequently used for Hispanic/Spanish origin (8, 9, 10, 11, 12, 13, 14, 15).
        hisp_set = {8, 9, 10, 11, 12, 13, 14, 15}
        cand = pd.Series(np.nan, index=df.index, dtype="float64")
        cand.loc[x.notna() & (~x.isin(hisp_set))] = 0.0
        cand.loc[x.isin(list(hisp_set))] = 1.0
        # Keep only if there is variation; otherwise keep missing to avoid a fake constant
        if cand.nunique(dropna=True) > 1:
            df["hispanic"] = cand
            used_hisp_source = "ethnic(heuristic)"

    # Conservative Protestant indicator:
    # Use a cautious rule that does not impute missing as 0.
    # Prefer denom16 if available else denom.
    denom_var = denom_candidates[0] if denom_candidates else None
    df["cons_prot"] = np.nan
    if denom_var is not None:
        rel = to_num(df["relig"])
        denom = to_num(df[denom_var])

        # Protestant base: RELIG==1 (per extract labels)
        # Conservative Protestant proxy:
        # - Baptist (DENOM==1) OR other conservative-coded groups when available.
        # With broad denom recode, only Baptist is clearly conservative; so use Baptist.
        cand = pd.Series(np.nan, index=df.index, dtype="float64")
        base_ok = rel.notna() & denom.notna()
        cand.loc[base_ok] = ((rel.loc[base_ok] == 1) & (denom.loc[base_ok] == 1)).astype(float)
        df["cons_prot"] = cand

    # No religion (RELIG==4), keep missing as missing
    df["no_religion"] = dummy_eq(df["relig"], 4)

    # Southern (REGION==3), keep missing as missing
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
        "racism_score": "Racism score (0–5; strict complete-case sum of 5 items)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": f"Hispanic (constructed; source={used_hisp_source or 'UNAVAILABLE'})",
        "other_race": "Other race (1=other)",
        "cons_prot": f"Conservative Protestant (proxy; RELIG==1 & {denom_var}==1; missing retained)",
        "no_religion": "No religion (RELIG==4; missing retained)",
        "southern": "Southern (REGION==3; missing retained)",
        "const": "Constant",
        "dv1_minority6_dislikes": "Dislike of minority-linked genres (count of 6)",
        "dv2_remaining12_dislikes": "Dislike of remaining genres (count of 12)",
    }

    # -----------------------------
    # Model fitting
    # -----------------------------
    def fit_table2_model(dv_col, model_name, dv_label):
        cols = [dv_col] + predictors
        d = df[cols].copy()

        # STRICT listwise deletion on all variables in model frame (faithful replication target)
        d = d.dropna(subset=cols).copy()

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
        X_z = pd.DataFrame({p: zscore(d[p]) for p in kept}, index=d.index)
        dz = pd.concat([y_z.rename("y_z"), X_z], axis=1).dropna()
        y_z2 = dz["y_z"].astype(float)
        X_z2 = dz[kept].astype(float)
        fit_beta = sm.OLS(y_z2, X_z2).fit()
        betas = fit_beta.params.reindex(kept)

        pvals = fit_unstd.pvalues.reindex(["const"] + kept)
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": stars.get(p, ""),
                    }
                )
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

        # Save outputs
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {dv_label}")
        lines.append("Model: OLS with standardized coefficients (beta weights).")
        lines.append("Betas computed as: regress z(Y) on z(X) (no intercept) on the analytic sample.")
        lines.append("Significance stars: two-tailed p-values from unstandardized OLS regression.")
        lines.append("Dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing.")
        lines.append("DV construction: strict sum across component genres (missing if any component missing).")
        lines.append("Racism scale: strict sum across 5 dichotomous items (missing if any component missing).")
        lines.append("Missing data: strict listwise deletion on DV and all predictors in the model frame.")
        lines.append("")
        lines.append("Standardized coefficients (Table 2 style)")
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

        with open(f"./output/{model_name}_ols_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write("Unstandardized OLS (fit stats + p-values):\n")
            f.write(fit_unstd.summary().as_text())
            f.write("\n\nStandardized-beta regression (zY on zX, no intercept):\n")
            f.write(fit_beta.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        diag = pd.DataFrame(
            {
                "variable": [dv_col] + predictors,
                "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors],
                "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors],
                "unique_values_in_analytic_sample": [safe_unique_count(d[dv_col])] + [safe_unique_count(d[p]) for p in predictors],
            }
        )
        diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)

        return table, fit_stats, diag, d

    m1_table, m1_fit, m1_diag, m1_frame = fit_table2_model(
        "dv1_minority6_dislikes",
        "Table2_ModelA_MinorityLinked6",
        labels["dv1_minority6_dislikes"],
    )
    m2_table, m2_fit, m2_diag, m2_frame = fit_table2_model(
        "dv2_remaining12_dislikes",
        "Table2_ModelB_Remaining12",
        labels["dv2_remaining12_dislikes"],
    )

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

    # Descriptives on constructed DVs (before listwise)
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

    # Quick frequencies for key dummies (on each analytic sample)
    def freq_table(frame, col):
        if col not in frame.columns:
            return pd.DataFrame({"value": [], "n": []})
        s = frame[col]
        return s.value_counts(dropna=False).rename_axis("value").reset_index(name="n").sort_values("value")

    freq_lines = []
    freq_lines.append("Analytic-sample frequency checks")
    freq_lines.append("================================")
    freq_lines.append("")
    freq_lines.append("Model A (DV1) analytic sample:")
    freq_lines.append(f"N={len(m1_frame)}")
    freq_lines.append("")
    for col in ["black", "hispanic", "other_race", "female", "cons_prot", "no_religion", "southern"]:
        ft = freq_table(m1_frame, col)
        freq_lines.append(f"{col}:")
        freq_lines.append(ft.to_string(index=False))
        freq_lines.append("")
    freq_lines.append("Model B (DV2) analytic sample:")
    freq_lines.append(f"N={len(m2_frame)}")
    freq_lines.append("")
    for col in ["black", "hispanic", "other_race", "female", "cons_prot", "no_religion", "southern"]:
        ft = freq_table(m2_frame, col)
        freq_lines.append(f"{col}:")
        freq_lines.append(ft.to_string(index=False))
        freq_lines.append("")

    with open("./output/analytic_sample_frequencies.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(freq_lines) + "\n")

    # Combined human-readable summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Key implementation choices (faithful to provided mapping)")
    lines.append("--------------------------------------------------------")
    lines.append("- Year filter: YEAR==1993")
    lines.append("- DV1: strict count of dislikes across RAP, REGGAE, BLUES, JAZZ, GOSPEL, LATIN (missing if any missing)")
    lines.append("- DV2: strict count of dislikes across the other 12 genres (missing if any missing)")
    lines.append("- Racism score: strict sum of 5 dichotomous items (missing if any missing)")
    lines.append("- Missing data: strict listwise deletion on DV and all predictors")
    lines.append(f"- Hispanic source used (if any): {used_hisp_source or 'NONE FOUND (will cause listwise loss / possible dropping)'}")
    lines.append(f"- Conservative Protestant denom variable used (if any): {denom_var or 'NONE FOUND (cons_prot will be all-missing and dropped)'}")
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
        dv_desc_fmt[c] = dv_desc_fmt[c].map(
            lambda v: (str(int(v)) if isinstance(v, (int, np.integer)) else format_float(v, 3))
        )
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
    m1_diag.to_csv("./output/Table2_ModelA_MinorityLinked6_analytic_diagnostics.csv", index=False)
    m2_diag.to_csv("./output/Table2_ModelB_Remaining12_analytic_diagnostics.csv", index=False)

    return {
        "table2_betas": combined,
        "fit": combined_fit,
        "dv_descriptives": dv_desc,
        "missingness_modelA": miss_A,
        "missingness_modelB": miss_B,
        "modelA_analytic_diagnostics": m1_diag,
        "modelB_analytic_diagnostics": m2_diag,
    }