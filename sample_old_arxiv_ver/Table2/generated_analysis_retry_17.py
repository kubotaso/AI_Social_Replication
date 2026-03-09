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

    def fmt(x, nd=3):
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
        ["id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "region"]
        + minority_genres
        + remaining_genres
        + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Numeric coercion (keep id as-is)
    for c in required:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables: strict dislike counts (0..6 and 0..12)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    dv1_col = "dv1_dislike_minority_linked_6"
    dv2_col = "dv2_dislike_remaining_12"

    df[dv1_col] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df[dv2_col] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0-5) strict sum of 5 dichotomies
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1

    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls / indicators (keep missing as missing; listwise deletion later)
    # -----------------------------
    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = dummy_eq(df["sex"], 2)

    # Age already numeric as df["age"]

    # Race dummies (reference: White)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: best-effort from ETHNIC if available (extract includes 'ethnic')
    # NOTE: This is an approximation; the classic GSS Hispanic indicator is separate.
    if "ethnic" in df.columns:
        df["ethnic"] = to_num(df["ethnic"])
        # Convention used here: treat code 1 as Hispanic (common in some recodes),
        # otherwise 0; unknown codes -> missing.
        df["hispanic"] = pd.Series(np.nan, index=df.index, dtype="float64")
        m = df["ethnic"].notna()
        df.loc[m, "hispanic"] = (df.loc[m, "ethnic"] == 1).astype(float)
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant: cannot be faithfully reproduced from available broad fields in this extract.
    # Keep as missing; model will report feasibility issues if this empties the sample.
    df["cons_prot"] = np.nan

    # No religion
    df["no_religion"] = dummy_eq(df["relig"], 4)

    # Southern (mapping instruction: REGION==3)
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
        dv1_col: "Dislike of minority-linked genres (Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin) (count of 6)",
        dv2_col: "Dislike of the 12 remaining genres (count of 12)",
        "racism_score": "Racism score (0–5; strict sum of 5 dichotomies)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (approx from ETHNIC==1 if available; else missing)",
        "other_race": "Other race (1=other)",
        "cons_prot": "Conservative Protestant (NOT REPRODUCIBLE from provided fields; missing)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting
    # -----------------------------
    def fit_table2_model(dv_col, model_name):
        cols = [dv_col] + predictors
        d_raw = df[cols].copy()

        # Strict listwise deletion on full model frame
        d = d_raw.dropna(subset=cols).copy()

        # Empty sample guard: write error and return consistent schemas
        if len(d) == 0:
            miss = labelled_missingness(df, cols, labels)
            miss_fmt = miss.copy()
            miss_fmt["share_missing"] = miss_fmt["share_missing"].map(lambda v: fmt(v, 3))

            msg = [
                model_name,
                "=" * len(model_name),
                "",
                "ERROR: Analytic sample is empty after strict listwise deletion.",
                f"DV: {labels.get(dv_col, dv_col)}",
                "",
                "Most likely cause: at least one required predictor is entirely missing in the provided extract.",
                "In this implementation, 'cons_prot' is set to missing because it cannot be faithfully reconstructed from the available broad fields.",
                "",
                "Missingness in 1993 (share missing):",
                miss_fmt.to_string(index=False),
                "",
            ]
            with open(f"./output/{model_name}_ERROR.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(msg) + "\n")

            table = pd.DataFrame(
                {
                    "Independent Variable": [labels.get(p, p) for p in predictors],
                    "Std_Beta": [np.nan] * len(predictors),
                    "Replication_Sig": [""] * len(predictors),
                }
            )
            fit_stats = pd.DataFrame(
                {
                    "DV": [labels.get(dv_col, dv_col)],
                    "N": [0],
                    "R2": [np.nan],
                    "Adj_R2": [np.nan],
                    "Constant": [np.nan],
                    "Constant_Replication_Sig": [""],
                    "Dropped_predictors_no_variation": [""],
                },
                index=[model_name],
            )
            diag = pd.DataFrame(
                {
                    "variable": [dv_col] + predictors,
                    "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors],
                    "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors],
                    "unique_values_in_analytic_sample": [0] + [0] * len(predictors),
                }
            )
            diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)
            table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
            fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)
            return table, fit_stats, d, diag

        # Drop non-varying predictors (avoid singular matrices)
        kept, dropped = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)

        # Unstandardized OLS (R2, intercept, p-values)
        if len(kept) > 0:
            X_unstd = d[kept].astype(float)
            Xc = sm.add_constant(X_unstd, has_constant="add")
        else:
            Xc = pd.DataFrame({"const": np.ones(len(d), dtype=float)}, index=d.index)
        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized betas: regress z(Y) on z(X) without intercept (beta weights)
        betas = pd.Series(index=kept, dtype="float64")
        fit_beta = None
        if len(kept) > 0:
            y_z = zscore(y).rename("y_z")
            X_z = pd.DataFrame({p: zscore(d[p]) for p in kept}, index=d.index)
            dz = pd.concat([y_z, X_z], axis=1).dropna()
            if len(dz) > 0 and dz["y_z"].std(ddof=0) > 0:
                ok = True
                for p in kept:
                    if dz[p].std(ddof=0) == 0 or pd.isna(dz[p].std(ddof=0)):
                        ok = False
                        break
                if ok:
                    fit_beta = sm.OLS(dz["y_z"].astype(float), dz[kept].astype(float)).fit()
                    betas = fit_beta.params.reindex(kept)

        # Replication stars (not claimed to match paper unless exact replication)
        pvals = fit_unstd.pvalues if hasattr(fit_unstd, "pvalues") else pd.Series(dtype=float)
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        # Table in paper-like order
        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Replication_Sig": stars.get(p, ""),
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Replication_Sig": "",
                    }
                )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(getattr(fit_unstd, "rsquared", np.nan))],
                "Adj_R2": [float(getattr(fit_unstd, "rsquared_adj", np.nan))],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Replication_Sig": [const_star],
                "Dropped_predictors_no_variation": [", ".join(dropped) if dropped else ""],
            },
            index=[model_name],
        )

        # Diagnostics
        diag = pd.DataFrame(
            {
                "variable": [dv_col] + predictors,
                "label": [labels.get(dv_col, dv_col)] + [labels.get(p, p) for p in predictors],
                "share_missing_in_1993": [df[dv_col].isna().mean()] + [df[p].isna().mean() for p in predictors],
                "unique_values_in_analytic_sample": [safe_unique_count(d[dv_col])] + [safe_unique_count(d[p]) for p in predictors],
            }
        )

        # Save outputs
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS.")
        lines.append("Reported coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: regress z(Y) on z(X) with no intercept (analytic listwise sample).")
        lines.append("Stars: two-tailed p-values from unstandardized OLS (replication stars).")
        lines.append("")
        lines.append("Construction notes:")
        lines.append("- Dislike coding per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV construction: strict sum across component genres (missing if any component missing within that DV).")
        lines.append("- Racism scale: strict sum of 5 dichotomies (0–5; missing if any component missing).")
        lines.append("- Missing data: strict listwise deletion on DV and all predictors in the model frame.")
        lines.append("")
        if dropped:
            lines.append("Dropped predictors due to no variation in analytic sample:")
            for p in dropped:
                lines.append(f"- {p}: {labels.get(p, p)}")
            lines.append("")
        lines.append("Standardized coefficients (Table-2 style; replication)")
        lines.append("-----------------------------------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp.to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS; replication)")
        lines.append("----------------------------------------------")
        fs = fit_stats[
            ["N", "R2", "Adj_R2", "Constant", "Constant_Replication_Sig", "Dropped_predictors_no_variation"]
        ].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

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
        diag.to_csv(f"./output/{model_name}_diagnostics.csv", index=False)

        return table, fit_stats, d, diag

    # Run models
    m1_table, m1_fit, m1_frame, m1_diag = fit_table2_model(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame, m2_diag = fit_table2_model(dv2_col, "Table2_ModelB_Remaining12")

    # Combined summary outputs
    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "ModelA_Std_Beta": m1_table["Std_Beta"],
            "ModelA_Replication_Sig": m1_table["Replication_Sig"],
            "ModelB_Std_Beta": m2_table["Std_Beta"],
            "ModelB_Replication_Sig": m2_table["Replication_Sig"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # Missingness tables (pre-listwise)
    miss_A = labelled_missingness(df, [dv1_col] + predictors, labels)
    miss_B = labelled_missingness(df, [dv2_col] + predictors, labels)

    # Human-readable combined summary (robust to missing columns in fit_stats)
    fit_cols = [
        "DV",
        "N",
        "R2",
        "Adj_R2",
        "Constant",
        "Constant_Replication_Sig",
        "Dropped_predictors_no_variation",
    ]
    fit_cols_present = [c for c in fit_cols if c in combined_fit.columns]
    combined_fit_disp = combined_fit.copy()
    for col in ["N"]:
        if col in combined_fit_disp.columns:
            combined_fit_disp[col] = combined_fit_disp[col].map(lambda v: fmt(v, 0))
    for col in ["R2", "Adj_R2", "Constant"]:
        if col in combined_fit_disp.columns:
            combined_fit_disp[col] = combined_fit_disp[col].map(lambda v: fmt(v, 3))

    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Implementation summary")
    lines.append("----------------------")
    lines.append("- Year filter: year == 1993")
    lines.append("- DVs: strict dislike counts (missing if any component genre rating missing within the DV set)")
    lines.append("- Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing")
    lines.append("- Racism score: strict sum of 5 dichotomies (0–5; missing if any component missing)")
    lines.append("- Estimation: OLS; standardized coefficients computed as z(Y) on z(X), no intercept")
    lines.append("- Regression missing data: strict listwise deletion on DV + all predictors")
    lines.append("")
    lines.append("Combined standardized coefficients (replication)")
    lines.append("----------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; replication)")
    lines.append("----------------------------------------------")
    lines.append(combined_fit_disp[fit_cols_present].to_string())
    lines.append("")
    lines.append("Missingness shares (Model A variables; before listwise)")
    lines.append("------------------------------------------------------")
    missA = miss_A.copy()
    missA["share_missing"] = missA["share_missing"].map(lambda v: fmt(v, 3))
    lines.append(missA.to_string(index=False))
    lines.append("")
    lines.append("Missingness shares (Model B variables; before listwise)")
    lines.append("------------------------------------------------------")
    missB = miss_B.copy()
    missB["share_missing"] = missB["share_missing"].map(lambda v: fmt(v, 3))
    lines.append(missB.to_string(index=False))
    lines.append("")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss_A.to_csv("./output/missingness_modelA_labelled.csv", index=False)
    miss_B.to_csv("./output/missingness_modelB_labelled.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "missingness_modelA": miss_A,
        "missingness_modelB": miss_B,
        "modelA_analytic_frame": m1_frame,
        "modelB_analytic_frame": m2_frame,
        "modelA_diagnostics": m1_diag,
        "modelB_diagnostics": m2_diag,
    }