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
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

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
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

    def partial_count_sum(dfin, cols, min_valid):
        """
        Count-style sum allowing partial completion:
        - Compute sum across cols
        - Requires at least min_valid non-missing components
        - Missing otherwise
        """
        nn = dfin[cols].notna().sum(axis=1)
        s = dfin[cols].sum(axis=1, skipna=True)
        out = s.where(nn >= min_valid, np.nan)
        return out

    def zscore(s):
        s = to_num(s)
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - s.mean()) / sd

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

    def fmt(x, nd=3):
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

    def fit_standardized_beta_table(d, ycol, xcols):
        """
        - Unstandardized OLS w/ intercept for fit stats and p-values
        - Standardized betas via OLS of z(y) on z(X) w/out intercept
        Returns: (beta_series, fit_unstd, fit_beta, table_df, fit_stats_df)
        """
        y = d[ycol].astype(float)
        X = d[xcols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        yz = zscore(y)
        Xz = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)
        dz = pd.concat([yz.rename("yz"), Xz], axis=1).dropna()

        fit_beta = None
        betas = pd.Series({c: np.nan for c in xcols}, dtype="float64")
        if len(dz) >= max(10, len(xcols) + 2):
            fit_beta = sm.OLS(dz["yz"].astype(float), dz[xcols].astype(float)).fit()
            betas = fit_beta.params.reindex(xcols)

        pvals = fit_unstd.pvalues
        rows = []
        for c in xcols:
            rows.append(
                {
                    "Variable": c,
                    "Std_Beta": float(betas.get(c, np.nan)),
                    "Sig": star_from_p(pvals.get(c, np.nan)),
                }
            )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [star_from_p(pvals.get("const", np.nan))],
            }
        )
        return betas, fit_unstd, fit_beta, table, fit_stats

    # -----------------------------
    # Required columns
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

    # Numeric coercion (leave id as-is)
    for c in df.columns:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # DVs: dislike counts
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    dv1_col = "dv1_minority6_dislikes"
    dv2_col = "dv2_remaining12_dislikes"

    # Bryson describes count of disliked genres; DK treated as missing.
    # To avoid excessive N loss (per feedback), allow partial completion but require strong coverage:
    # - DV1: at least 5 of 6 genres answered
    # - DV2: at least 10 of 12 genres answered
    # This keeps the model faithful while preventing severe sample collapse from a single missing item.
    dv1_items = [f"d_{c}" for c in minority_genres]
    dv2_items = [f"d_{c}" for c in remaining_genres]
    df[dv1_col] = partial_count_sum(df, dv1_items, min_valid=5)
    df[dv2_col] = partial_count_sum(df, dv2_items, min_valid=10)

    # -----------------------------
    # Racism score (0–5): sum of 5 dichotomies; strict (no rescale) with >=4 answered
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    k = df[racism_comp].notna().sum(axis=1)
    s = df[racism_comp].sum(axis=1, skipna=True)
    # Keep on a 0-5 scale; for partial completion, rescale to 0-5 metric.
    df["racism_score"] = np.where(k >= 4, s * (5.0 / k), np.nan)

    # -----------------------------
    # Controls / indicators (do NOT impute missing to 0; use listwise deletion)
    # -----------------------------
    # Education (years)
    df["education"] = df["educ"]

    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Occupational prestige
    df["occ_prestige"] = df["prestg80"]

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Age
    df["age_years"] = df["age"]

    # Race / ethnicity dummies (mutually exclusive style expected by table)
    # Reference: non-Hispanic white (implicitly)
    # Hispanic best-effort: ETHNIC==1 in this extract indicates Hispanic in typical recodes.
    df["hispanic"] = np.where(df["ethnic"].notna(), (df["ethnic"] == 1).astype(float), np.nan)

    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # Conservative Protestant: with only RELIG and broad DENOM available, the closest we can do:
    # - Restrict to Protestants (RELIG==1)
    # - Mark as conservative Protestant if DENOM in {1,2} where 1=baptist, 2=methodist? (often mainline),
    #   BUT methodist is not conservative. With this extract we cannot reproduce Bryson's exact scheme.
    # Therefore: keep the simplest and transparent proxy: Protestant & Baptist (DENOM==1), but do not fill missing.
    df["cons_prot"] = np.nan
    rel_denom_obs = df["relig"].notna() & df["denom"].notna()
    df.loc[rel_denom_obs, "cons_prot"] = (
        ((df.loc[rel_denom_obs, "relig"] == 1) & (df.loc[rel_denom_obs, "denom"] == 1)).astype(float)
    )

    # No religion
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
        "racism_score",
        "education",
        "income_pc",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    # Presentation labels to match Table 2 naming
    present_labels = {
        "racism_score": "Racism score",
        "education": "Education",
        "income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant",
        dv1_col: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count)",
        dv2_col: "Dislike of the 12 Remaining Genres (count)",
    }

    # -----------------------------
    # Fit models with listwise deletion
    # -----------------------------
    def fit_model(dv_col, model_name):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Listwise deletion on all model variables (faithful)
        d = d0.dropna(subset=model_cols).copy()

        # Drop no-variation predictors (avoid singular matrices)
        kept = []
        dropped_no_var = []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        if len(d) < max(30, len(kept) + 5):
            # Still write a diagnostic file, then return NA tables
            lines = []
            lines.append(model_name)
            lines.append("=" * len(model_name))
            lines.append("")
            lines.append(f"ERROR: Analytic sample too small after listwise deletion (N={len(d)}).")
            lines.append(f"DV: {present_labels.get(dv_col, dv_col)}")
            lines.append("")
            miss = d0.isna().mean().reset_index()
            miss.columns = ["variable", "share_missing"]
            miss["label"] = miss["variable"].map(lambda v: present_labels.get(v, v))
            miss = miss.sort_values("share_missing", ascending=False)
            miss["share_missing"] = miss["share_missing"].map(lambda v: fmt(v, 3))
            lines.append("Missingness (1993; before listwise):")
            lines.append(miss.to_string(index=False))
            with open(f"./output/{model_name}_ERROR.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            table = pd.DataFrame(
                {
                    "Independent Variable": [present_labels.get(p, p) for p in predictors],
                    "Std_Beta": [np.nan] * len(predictors),
                    "Sig": [""] * len(predictors),
                }
            )
            fit_stats = pd.DataFrame(
                {
                    "DV": [present_labels.get(dv_col, dv_col)],
                    "N": [len(d)],
                    "R2": [np.nan],
                    "Adj_R2": [np.nan],
                    "Constant": [np.nan],
                    "Constant_Sig": [""],
                    "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
                },
                index=[model_name],
            )
            return table, fit_stats, d, None, None

        betas, fit_unstd, fit_beta, table_raw, fit_stats_raw = fit_standardized_beta_table(d, dv_col, kept)

        # Align table rows to requested predictor order (even if dropped)
        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": present_labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit_unstd.pvalues.get(p, np.nan)),
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": present_labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Sig": "",
                    }
                )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [present_labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [star_from_p(fit_unstd.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        # Write human-readable file
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {present_labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS.")
        lines.append("Coefficients shown: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: regress z(Y) on z(X) with no intercept on the analytic (listwise) sample.")
        lines.append("Stars: two-tailed p-values from unstandardized OLS with intercept (replication-based).")
        lines.append("")
        lines.append("Construction:")
        lines.append("- Year filter: YEAR==1993")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV1: count across 6 items, computed if >=5 items answered")
        lines.append("- DV2: count across 12 items, computed if >=10 items answered")
        lines.append("- Racism score: 5 dichotomies; computed if >=4 answered; rescaled to 0–5")
        lines.append("- Missing data in regression: listwise deletion on DV + all predictors used")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped predictors due to no variation in analytic sample:")
            for p in dropped_no_var:
                lines.append(f"- {p}: {present_labels.get(p, p)}")
        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp.to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())
        lines.append("")
        lines.append("Unstandardized OLS summary")
        lines.append("-------------------------")
        lines.append(fit_unstd.summary().as_text())
        if fit_beta is not None:
            lines.append("")
            lines.append("Standardized-beta regression summary (zY on zX, no intercept)")
            lines.append("-----------------------------------------------------------")
            lines.append(fit_beta.summary().as_text())

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        return table, fit_stats, d, fit_unstd, fit_beta

    m1_table, m1_fit, m1_frame, m1_fit_unstd, m1_fit_beta = fit_model(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame, m2_fit_unstd, m2_fit_beta = fit_model(dv2_col, "Table2_ModelB_Remaining12")

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

    # Descriptives (constructed DVs before listwise)
    dv_desc = pd.DataFrame(
        {
            "stat": ["N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"],
            "DV1": [
                int(df[dv1_col].notna().sum()),
                df[dv1_col].mean(),
                df[dv1_col].std(ddof=0),
                df[dv1_col].min(),
                df[dv1_col].quantile(0.25),
                df[dv1_col].quantile(0.50),
                df[dv1_col].quantile(0.75),
                df[dv1_col].max(),
            ],
            "DV2": [
                int(df[dv2_col].notna().sum()),
                df[dv2_col].mean(),
                df[dv2_col].std(ddof=0),
                df[dv2_col].min(),
                df[dv2_col].quantile(0.25),
                df[dv2_col].quantile(0.50),
                df[dv2_col].quantile(0.75),
                df[dv2_col].max(),
            ],
        }
    )

    # Missingness overview in 1993 (before listwise)
    miss = df[[dv1_col, dv2_col] + predictors].isna().mean().reset_index()
    miss.columns = ["variable", "share_missing_1993"]
    miss["label"] = miss["variable"].map(lambda v: present_labels.get(v, v))
    miss = miss.sort_values("share_missing_1993", ascending=False)

    # Frequency checks for key dummies (to help debug discrepancies)
    freq_vars = ["race", "ethnic", "hispanic", "relig", "denom", "cons_prot", "no_religion", "region", "southern", "black", "other_race"]
    freq_lines = []
    for v in freq_vars:
        if v in df.columns:
            vc = df[v].value_counts(dropna=False).sort_index()
            freq_lines.append(f"{v} value counts (1993):")
            freq_lines.append(vc.to_string())
            freq_lines.append("")

    # Human-readable combined summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract)")
    lines.append("===========================================================")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) with replication-based stars")
    lines.append("----------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; replication)")
    lines.append("----------------------------------------------")
    fd = combined_fit.copy()
    fd["N"] = fd["N"].map(lambda v: fmt(v, 0))
    for c in ["R2", "Adj_R2", "Constant"]:
        fd[c] = fd[c].map(lambda v: fmt(v, 3))
    lines.append(fd.to_string())
    lines.append("")
    lines.append("DV descriptives (constructed counts; before listwise deletion)")
    lines.append("------------------------------------------------------------")
    dv_disp = dv_desc.copy()
    dv_disp["DV1"] = dv_disp["DV1"].map(lambda v: str(int(v)) if isinstance(v, (int, np.integer)) else fmt(v, 3))
    dv_disp["DV2"] = dv_disp["DV2"].map(lambda v: str(int(v)) if isinstance(v, (int, np.integer)) else fmt(v, 3))
    lines.append(dv_disp.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares in 1993 (before listwise deletion)")
    lines.append("----------------------------------------------------")
    miss_disp = miss.copy()
    miss_disp["share_missing_1993"] = miss_disp["share_missing_1993"].map(lambda v: fmt(v, 3))
    lines.append(miss_disp.to_string(index=False))
    lines.append("")
    lines.append("Analytic sample sizes (after listwise deletion)")
    lines.append("----------------------------------------------")
    lines.append(f"Model A analytic N: {len(m1_frame)}")
    lines.append(f"Model B analytic N: {len(m2_frame)}")
    lines.append("")
    lines.append("Selected frequency checks (1993)")
    lines.append("--------------------------------")
    lines.append("\n".join(freq_lines).strip() if freq_lines else "(none)")
    lines.append("")
    lines.append("Notes / known limitations")
    lines.append("-------------------------")
    lines.append("- Hispanic is approximated from ETHNIC==1 in this extract; if this does not match Bryson's Hispanic flag, coefficients may differ.")
    lines.append("- Conservative Protestant is approximated as RELIG==1 & DENOM==1 due to limited denomination detail in this extract.")
    lines.append("- DV1/DV2 allow partial completion (>=5/6; >=10/12) to reduce case loss; strict complete-case counts can be produced by switching to strict_sum().")
    lines.append("- Stars are computed from this replication's unstandardized OLS p-values (two-tailed), not copied from the paper.")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss.to_csv("./output/missingness_1993.csv", index=False)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "missingness_1993": miss,
        "dv_descriptives": dv_desc,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_fit": m1_fit,
        "modelB_fit": m2_fit,
        "modelA_analytic_frame": m1_frame,
        "modelB_analytic_frame": m2_frame,
    }