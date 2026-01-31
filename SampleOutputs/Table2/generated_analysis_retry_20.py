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
        missing otherwise (including DK/NA/refused, etc. after coercion).
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

    def zscore(s):
        s = to_num(s)
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - s.mean()) / sd

    def betas_from_unstd(fit, X, y):
        """
        Standardized betas from unstandardized OLS coefficients:
        beta_j = b_j * sd(x_j) / sd(y)
        """
        ysd = float(np.std(y, ddof=0))
        betas = {}
        if ysd == 0 or pd.isna(ysd):
            return pd.Series({k: np.nan for k in X.columns})
        for c in X.columns:
            xsd = float(np.std(X[c], ddof=0))
            b = float(fit.params.get(c, np.nan))
            betas[c] = b * (xsd / ysd) if (xsd != 0 and not pd.isna(xsd) and not pd.isna(b)) else np.nan
        return pd.Series(betas)

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

    # numeric coercion (leave id as-is)
    for c in df.columns:
        if c != "id":
            df[c] = to_num(df[c])

    # -----------------------------
    # Dependent variables: strict dislike counts (per instruction)
    # -----------------------------
    for c in minority_genres + remaining_genres:
        df[f"d_{c}"] = dislike_indicator(df[c])

    dv1_col = "dv1_minority6_dislikes"
    dv2_col = "dv2_remaining12_dislikes"
    df[dv1_col] = strict_sum(df, [f"d_{c}" for c in minority_genres])
    df[dv2_col] = strict_sum(df, [f"d_{c}" for c in remaining_genres])

    # -----------------------------
    # Racism score (0–5): strict sum of 5 dichotomies (per mapping)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])     # 1=yes object -> 1
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])     # 2=oppose -> 1
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])   # 2=no discrimination -> 1
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])   # 2=no education chance -> 1
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])   # 1=yes willpower -> 1
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls (faithful missing handling: do NOT fill missing dummies with 0)
    # -----------------------------
    # income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # female (SEX: 1=male, 2=female)
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # age (continuous; keep missing as missing)
    df["age"] = df["age"]

    # race dummies (reference=white). keep missing as missing.
    race_known = df["race"].isin([1, 2, 3])
    df["black"] = np.where(race_known, (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (df["race"] == 3).astype(float), np.nan)

    # hispanic (best available in this extract): ETHNIC==1 => hispanic; other observed => not hispanic; missing => missing
    df["hispanic"] = np.nan
    eth_obs = df["ethnic"].notna()
    df.loc[eth_obs, "hispanic"] = (df.loc[eth_obs, "ethnic"] == 1).astype(float)

    # no religion (RELIG==4); missing stays missing
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # southern (REGION==3); missing stays missing
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # conservative protestant: cannot fully match Bryson with broad DENOM.
    # Keep the simplest defensible proxy with faithful missing handling:
    # CONS_PROT = 1 if RELIG==1 (protestant) AND DENOM==1 (baptist); 0 if RELIG/DENOM observed and not that; else missing.
    df["cons_prot"] = np.nan
    rel_denom_obs = df["relig"].notna() & df["denom"].notna()
    df.loc[rel_denom_obs, "cons_prot"] = (
        ((df.loc[rel_denom_obs, "relig"] == 1) & (df.loc[rel_denom_obs, "denom"] == 1)).astype(float)
    )

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
        dv1_col: "Dislike of minority-linked genres (count of 6: Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin)",
        dv2_col: "Dislike of remaining genres (count of 12)",
        "racism_score": "Racism score (0–5; strict sum of 5 dichotomies)",
        "educ": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "prestg80": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age": "Age (years)",
        "black": "Black (1=Black)",
        "hispanic": "Hispanic (ETHNIC==1 if observed; else 0; missing if ETHNIC missing)",
        "other_race": "Other race (1=other)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1; missing if RELIG/DENOM missing)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Modeling
    # -----------------------------
    def fit_model(dv_col, model_name):
        model_cols = [dv_col] + predictors
        d = df[model_cols].copy()

        # Drop predictors that are entirely missing (shouldn't happen, but keep robust)
        usable = []
        dropped_all_missing = []
        for p in predictors:
            if d[p].notna().sum() == 0:
                dropped_all_missing.append(p)
            else:
                usable.append(p)

        # Listwise deletion on DV + usable predictors (faithful)
        d = d.dropna(subset=[dv_col] + usable).copy()

        # Drop no-variation predictors (protect against singular design)
        kept = []
        dropped_no_var = []
        for p in usable:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        # If too small / empty, write a diagnostic file and return placeholders
        if len(d) < 5 or len(kept) == 0:
            msg = []
            msg.append(model_name)
            msg.append("=" * len(model_name))
            msg.append("")
            msg.append("ERROR: Analytic sample too small or no usable predictors after listwise deletion.")
            msg.append(f"DV: {labels.get(dv_col, dv_col)}")
            msg.append(f"N after listwise: {len(d)}")
            msg.append("")
            msg.append("Missingness shares in 1993 (before listwise):")
            miss = df[[dv_col] + usable].isna().mean().reset_index()
            miss.columns = ["variable", "share_missing"]
            miss["label"] = miss["variable"].map(lambda v: labels.get(v, v))
            miss = miss.sort_values("share_missing", ascending=False)
            miss["share_missing"] = miss["share_missing"].map(lambda v: fmt(v, 3))
            msg.append(miss.to_string(index=False))
            msg.append("")
            msg.append("Dropped_all_missing: " + (", ".join(dropped_all_missing) if dropped_all_missing else ""))
            msg.append("Dropped_no_variation: " + (", ".join(dropped_no_var) if dropped_no_var else ""))
            with open(f"./output/{model_name}_ERROR.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(msg) + "\n")

            table = pd.DataFrame(
                {
                    "Independent Variable": [labels.get(p, p) for p in predictors],
                    "Std_Beta": [np.nan] * len(predictors),
                    "Replication_Sig": [""] * len(predictors),
                    "Included_in_model": [False] * len(predictors),
                }
            )
            fit_stats = pd.DataFrame(
                {
                    "DV": [labels.get(dv_col, dv_col)],
                    "N": [len(d)],
                    "R2": [np.nan],
                    "Adj_R2": [np.nan],
                    "Constant": [np.nan],
                    "Constant_Replication_Sig": [""],
                    "Dropped_all_missing": [", ".join(dropped_all_missing) if dropped_all_missing else ""],
                    "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
                },
                index=[model_name],
            )
            return table, fit_stats, d

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        # Standardized betas (computed from unstandardized fit; stable and matches common "beta weight" definition)
        betas = betas_from_unstd(fit, X, y)

        # Replication stars from unstandardized OLS p-values
        pvals = fit.pvalues
        stars = {p: star_from_p(pvals.get(p, np.nan)) for p in kept}
        const_star = star_from_p(pvals.get("const", np.nan))

        rows = []
        for p in predictors:
            rows.append(
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": float(betas.get(p, np.nan)) if p in betas.index else np.nan,
                    "Replication_Sig": stars.get(p, "") if p in kept else "",
                    "Included_in_model": bool(p in kept),
                }
            )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Replication_Sig": [const_star],
                "Dropped_all_missing": [", ".join(dropped_all_missing) if dropped_all_missing else ""],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        # Save human-readable model summary
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Model: OLS.")
        lines.append("Reported coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: beta_j = b_j * sd(x_j) / sd(y), using the analytic (listwise) sample.")
        lines.append("Significance stars: two-tailed p-values from unstandardized OLS (replication stars).")
        lines.append("")
        lines.append("Construction notes:")
        lines.append("- Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV is strict count across component items (missing if any component missing within that DV).")
        lines.append("- Racism score is strict sum of 5 dichotomies (missing if any component missing).")
        lines.append("- Missing data: listwise deletion on DV + predictors (no imputation of missing dummies to 0).")
        if dropped_all_missing:
            lines.append("")
            lines.append("Dropped predictors (100% missing in extract): " + ", ".join(dropped_all_missing))
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped predictors (no variation in analytic sample): " + ", ".join(dropped_no_var))
        lines.append("")
        lines.append("Standardized coefficients (Table-2 style; replication)")
        lines.append("-----------------------------------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp.to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS; replication)")
        lines.append("----------------------------------------------")
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Replication_Sig", "Dropped_all_missing", "Dropped_no_variation"]].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())
        lines.append("")
        lines.append("Unstandardized OLS diagnostics")
        lines.append("-----------------------------")
        lines.append(fit.summary().as_text())

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        return table, fit_stats, d

    # Fit both models
    m1_table, m1_fit, m1_frame = fit_model(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_frame = fit_model(dv2_col, "Table2_ModelB_Remaining12")

    # Combined table
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

    # Key diagnostics: variable missingness in 1993 and analytic N
    miss = df[[dv1_col, dv2_col] + predictors].isna().mean().reset_index()
    miss.columns = ["variable", "share_missing_1993"]
    miss["label"] = miss["variable"].map(lambda v: labels.get(v, v))
    miss = miss.sort_values("share_missing_1993", ascending=False)

    # Ensure we can see whether no_religion / hispanic actually vary (common bug source)
    freq_lines = []
    for v in ["relig", "no_religion", "ethnic", "hispanic", "region", "southern", "denom", "cons_prot"]:
        if v in df.columns:
            s = df[v]
            freq = s.value_counts(dropna=False).sort_index()
            freq_lines.append(f"{v} value counts (1993):")
            freq_lines.append(freq.to_string())
            freq_lines.append("")

    # Save combined summary
    lines = []
    lines.append("Bryson (1996) Table 2 replication attempt (1993 GSS extract provided)")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Implementation summary")
    lines.append("----------------------")
    lines.append("- Year filter: year == 1993")
    lines.append("- DVs: strict dislike counts (missing if any component rating missing within DV set)")
    lines.append("- Dislike coding: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing")
    lines.append("- Racism score: strict sum of 5 dichotomies (0–5; missing if any component missing)")
    lines.append("- Estimation: OLS; standardized betas from beta_j = b_j * sd(x_j) / sd(y)")
    lines.append("- Stars: replication stars from unstandardized OLS p-values (not the paper's stars unless the model matches)")
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
    fit_disp = combined_fit.copy()
    fit_disp["N"] = fit_disp["N"].map(lambda v: fmt(v, 0))
    for c in ["R2", "Adj_R2", "Constant"]:
        fit_disp[c] = fit_disp[c].map(lambda v: fmt(v, 3))
    lines.append(fit_disp.to_string())
    lines.append("")
    lines.append("Missingness shares in 1993 (before listwise deletion)")
    lines.append("----------------------------------------------------")
    miss_disp = miss.copy()
    miss_disp["share_missing_1993"] = miss_disp["share_missing_1993"].map(lambda v: fmt(v, 3))
    lines.append(miss_disp.to_string(index=False))
    lines.append("")
    lines.append("Selected frequency checks (1993)")
    lines.append("-------------------------------")
    lines.append("\n".join(freq_lines).strip() if freq_lines else "(none)")
    lines.append("")
    lines.append("Analytic sample sizes")
    lines.append("---------------------")
    lines.append(f"Model A (DV1) analytic N: {len(m1_frame)}")
    lines.append(f"Model B (DV2) analytic N: {len(m2_frame)}")

    with open("./output/combined_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)
    miss.to_csv("./output/missingness_1993.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "missingness_1993": miss,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_fit": m1_fit,
        "modelB_fit": m2_fit,
        "modelA_analytic_frame": m1_frame,
        "modelB_analytic_frame": m2_frame,
    }