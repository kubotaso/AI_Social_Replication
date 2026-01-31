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
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    # numeric coercion (except id)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Required columns (as available in provided extract)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_required = ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
    required = ["year", "id"] + base_required + minority_genres + remaining_genres + racism_items

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # -----------------------------
    # Helpers
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")

    def fmt(x, nd=3):
        try:
            if x is None or pd.isna(x):
                return ""
            return f"{float(x):.{nd}f}"
        except Exception:
            return ""

    def star_from_p(p):
        try:
            if p is None or pd.isna(p):
                return ""
            p = float(p)
        except Exception:
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def dislike_indicator(series):
        # 1 if 4/5; 0 if 1/2/3; else missing
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        # map specified codes to {0,1}; else missing
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        # missing if ANY component missing
        return dfin[cols].sum(axis=1, skipna=False)

    def standardized_betas_from_unstd(fit, d_analytic, ycol, xcols):
        # beta_j = b_j * SD(x_j) / SD(y), computed on analytic sample; ddof=0
        y = pd.to_numeric(d_analytic[ycol], errors="coerce").astype(float)
        sd_y = y.std(ddof=0)
        betas = {}
        for x in xcols:
            sx = pd.to_numeric(d_analytic[x], errors="coerce").astype(float)
            sd_x = sx.std(ddof=0)
            b = fit.params.get(x, np.nan)
            if pd.isna(b) or pd.isna(sd_x) or pd.isna(sd_y) or sd_x == 0 or sd_y == 0:
                betas[x] = np.nan
            else:
                betas[x] = float(b * (sd_x / sd_y))
        return pd.Series(betas)

    def value_counts_full(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.value_counts(dropna=False).sort_index()

    # -----------------------------
    # Dependent variables (strict dislike counts)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6"
    dv2 = "dv2_remaining12"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])     # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])    # 0..12

    # -----------------------------
    # Racism score (0..5; strict)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])       # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])       # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])     # 2=no -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])     # 2=no -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])     # 1=yes -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls (faithful to provided mapping; avoid ad-hoc proxies that break validity)
    # -----------------------------
    df["education"] = df["educ"]

    # Income per capita: REALINC/HOMPOP; require HOMPOP>0; missing otherwise
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    # Female: 1 if SEX==2, 0 if SEX==1, else missing
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Age
    df["age_years"] = df["age"]

    # Race indicators (reference: White=1)
    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic: NOT AVAILABLE in provided variables (do not fabricate a proxy)
    # Keep as missing; do not include in regression.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy specified in mapping: RELIG==1 & DENOM==1 (baptist)
    # To avoid inducing missingness, treat DENOM missing among Protestants as 0.
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.where(rel.notna(), 0.0, np.nan)
    prot = rel.notna() & (rel == 1)
    df.loc[prot, "cons_prot"] = np.where(den.loc[prot].notna(), (den.loc[prot] == 1).astype(float), 0.0)

    # No religion: RELIG==4
    df["no_religion"] = np.where(rel.notna(), (rel == 4).astype(float), np.nan)

    # Southern: REGION==3
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
        "racism_score",
        "education",
        "income_pc",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    dv_labels = {
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
        dv2: "Dislike of the 12 Remaining Genres (count of 12)",
    }
    var_labels = {
        "racism_score": "Racism score",
        "education": "Education",
        "income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting (listwise on DV + predictors)
    # -----------------------------
    def fit_model(dv_col, model_name, stub, paper_target_n=None):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(axis=0, how="any").copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                "Missingness shares in YEAR==1993 for model columns:\n"
                + d0.isna().mean().sort_values(ascending=False).to_string()
                + "\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop predictors with no variation (defensive)
        kept, dropped_no_var = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(fit, d, dv_col, kept)

        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": var_labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit.pvalues.get(p, np.nan)),
                        "Status": "included",
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": var_labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Sig": "",
                        "Status": "dropped (no variation)",
                    }
                )

        const = float(fit.params.get("const", np.nan))
        const_p = float(fit.pvalues.get("const", np.nan))
        rows.append(
            {
                "Independent Variable": var_labels["const"],
                "Std_Beta": const,  # intercept (unstandardized)
                "Sig": star_from_p(const_p),
                "Status": "intercept (unstandardized)",
            }
        )

        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [dv_labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [const],
                "Constant_p": [const_p],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable table output
        title = f"Bryson (1996) Table 2 replication (as feasible with provided extract) — {model_name}"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"Filter: YEAR==1993. N_1993_total={int(df.shape[0])}.")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights) for slopes, computed as b*SD(x)/SD(y) on the analytic sample.")
        lines.append("Constant: unstandardized intercept from the same OLS fit.")
        lines.append("Stars: two-tailed p-values from this OLS fit.")
        lines.append("")
        lines.append("Key implementation details:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum across 5 dichotomous items (missing if any item missing)")
        lines.append("- Income per capita: REALINC/HOMPOP with HOMPOP>0 required")
        lines.append("- Race dummies: Black=1 if RACE==2; Other race=1 if RACE==3; reference is White (RACE==1)")
        lines.append("- Conservative Protestant: RELIG==1 & DENOM==1 (baptist); protestants with missing DENOM treated as 0 to avoid extra missingness")
        lines.append("- No religion: RELIG==4; Southern: REGION==3")
        lines.append("- Hispanic: not available in provided variables; omitted rather than proxied (paper includes it)")
        lines.append("- Missing data: listwise deletion on DV and included predictors")
        if paper_target_n is not None:
            lines.append(f"- Paper target N (reference only): {int(paper_target_n)}")
        if dropped_no_var:
            lines.append(f"- Dropped for no variation: {', '.join(dropped_no_var)}")
        lines.append("")

        disp = table.copy()
        disp["Coefficient"] = disp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append("Regression table")
        lines.append("----------------")
        lines.append(disp[["Independent Variable", "Coefficient", "Sig", "Status"]].to_string(index=False))
        lines.append("")
        lines.append("Fit statistics")
        lines.append("--------------")
        fs = fit_stats.copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        fs["Constant_p"] = fs["Constant_p"].map(lambda v: fmt(v, 3))
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant", "Constant_p", "Dropped_no_variation"]].to_string(index=False))

        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))
        write_text(f"./output/{stub}_ols_unstandardized_summary.txt", fit.summary().as_text())

        # Diagnostics (helps explain N gaps without changing the model)
        diag = []
        diag.append(f"{model_name} diagnostics")
        diag.append("=" * (len(model_name) + 12))
        diag.append(f"N_1993_total: {int(df.shape[0])}")
        diag.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag.append(f"N_analytic_listwise: {int(d.shape[0])}")
        if paper_target_n is not None:
            diag.append(f"N_target_from_paper: {int(paper_target_n)}")
            diag.append(f"N_gap (analytic - target): {int(d.shape[0]) - int(paper_target_n)}")
        diag.append("")
        diag.append("Missingness shares in YEAR==1993 for model columns (descending):")
        diag.append(d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())
        diag.append("")
        diag.append("Value counts (YEAR==1993, incl. NA):")
        diag.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        diag.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag.append("\nRacism score:\n" + value_counts_full(df["racism_score"]).to_string())
        diag.append(f"\n{dv_labels.get(dv_col, dv_col)} value counts:\n" + value_counts_full(df[dv_col]).to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d

    m1_table, m1_fit, m1_d = fit_model(
        dv1, "Model 2A (Minority-linked genres: 6)", "Table2_Model2A_MinorityLinked6", paper_target_n=644
    )
    m2_table, m2_fit, m2_d = fit_model(
        dv2, "Model 2B (Remaining genres: 12)", "Table2_Model2B_Remaining12", paper_target_n=605
    )

    # -----------------------------
    # Combined outputs
    # -----------------------------
    combined = pd.DataFrame({"Independent Variable": m1_table["Independent Variable"]}).merge(
        m1_table.rename(columns={"Std_Beta": "Model2A_Coefficient", "Sig": "Model2A_Sig", "Status": "Model2A_Status"})[
            ["Independent Variable", "Model2A_Coefficient", "Model2A_Sig", "Model2A_Status"]
        ],
        on="Independent Variable",
        how="left",
    ).merge(
        m2_table.rename(columns={"Std_Beta": "Model2B_Coefficient", "Sig": "Model2B_Sig", "Status": "Model2B_Status"})[
            ["Independent Variable", "Model2B_Coefficient", "Model2B_Sig", "Model2B_Status"]
        ],
        on="Independent Variable",
        how="outer",
    )

    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    title = "Bryson (1996) Table 2 replication (as feasible with provided extract)"
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Dependent variables:")
    lines.append(f"- Model 2A DV: {dv_labels[dv1]}")
    lines.append(f"- Model 2B DV: {dv_labels[dv2]}")
    lines.append("")
    lines.append("Important limitations vs Bryson (1996) Table 2:")
    lines.append("- Hispanic indicator is not available in the provided extract and is omitted (paper includes it).")
    lines.append("- Conservative Protestant is approximated as RELIG==1 & DENOM==1 (baptist) due to limited denomination detail.")
    lines.append("")
    lines.append("Combined coefficients (slopes are standardized betas; Constant is unstandardized intercept)")
    lines.append("-----------------------------------------------------------------------------------")
    tmp = combined.copy()
    for c in ["Model2A_Coefficient", "Model2B_Coefficient"]:
        tmp[c] = tmp[c].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics")
    lines.append("--------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    fs["Constant_p"] = fs["Constant_p"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_p", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("Analytic sample sizes (computed)")
    lines.append("-------------------------------")
    lines.append(f"Model 2A analytic N: {int(m1_d.shape[0])}")
    lines.append(f"Model 2B analytic N: {int(m2_d.shape[0])}")
    lines.append("")
    lines.append("Saved outputs in ./output:")
    lines.append("- Table2_Model2A_MinorityLinked6_table2_style.txt / .csv")
    lines.append("- Table2_Model2B_Remaining12_table2_style.txt / .csv")
    lines.append("- *_ols_unstandardized_summary.txt (statsmodels OLS summary)")
    lines.append("- *_diagnostics.txt (missingness and key distributions)")
    lines.append("- combined_summary.txt / combined_table2_betas.csv / combined_fit.csv")

    write_text("./output/combined_summary.txt", "\n".join(lines))
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "model2a_table": m1_table,
        "model2b_table": m2_table,
        "model2a_analytic_sample": m1_d,
        "model2b_analytic_sample": m2_d,
    }