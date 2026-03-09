def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + normalize
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
    # Required fields
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["year", "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_items
    )
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
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        try:
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
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        # Missing if ANY component missing
        return dfin[cols].sum(axis=1, skipna=False)

    def standardized_betas_from_unstd(fit, d_analytic, ycol, xcols):
        # beta_j = b_j * SD(x_j) / SD(y) on analytic sample
        y = pd.to_numeric(d_analytic[ycol], errors="coerce").astype(float)
        sd_y = y.std(ddof=1)
        betas = {}
        for x in xcols:
            sx = pd.to_numeric(d_analytic[x], errors="coerce").astype(float)
            sd_x = sx.std(ddof=1)
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
    # Dependent variables (strict counts)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6"
    dv2 = "dv2_remaining12"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])    # 0..6, missing if any missing
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])   # 0..12, missing if any missing

    # -----------------------------
    # Racism score (0..5, strict)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrim
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no educ opp
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls / indicators (minimize unnecessary missingness)
    # -----------------------------
    df["education"] = df["educ"]

    # Income per capita: REALINC / HOMPOP (require HOMPOP>0)
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Occupational prestige
    df["occ_prestige"] = df["prestg80"]

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Age
    df["age_years"] = df["age"]

    # Race dummies (White reference; require RACE in {1,2,3})
    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic dummy (best-effort from ETHNIC without forcing extra missingness):
    # - If RACE==2 (Black), set Hispanic=0 (to keep categories closer to mutually-exclusive scheme in practice)
    # - Else if ETHNIC indicates Hispanic origin => 1
    # - Else if ETHNIC is non-missing => 0
    # Note: ETHNIC coding can vary by extract; we keep this explicit and diagnose distributions.
    eth = df["ethnic"]
    hisp_code = eth.isin(list(range(20, 30)))  # common GSS origin bucket for Hispanic
    df["hispanic"] = np.where(eth.notna(), hisp_code.astype(float), np.nan)
    df.loc[race_known & (race == 2) & df["hispanic"].notna(), "hispanic"] = 0.0

    # Conservative Protestant proxy:
    # Use RELIG and DENOM broad recode; treat missing DENOM among non-Protestants as 0.
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.where(rel.notna(), 0.0, np.nan)
    prot = rel.notna() & (rel == 1)
    df.loc[prot, "cons_prot"] = np.where(den.loc[prot].notna(), (den.loc[prot] == 1).astype(float), 0.0)

    # No religion
    df["no_religion"] = np.where(rel.notna(), (rel == 4).astype(float), np.nan)

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

    dv_labels = {
        dv1: "Dislike of minority-linked genres (Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin): count of 6",
        dv2: "Dislike of remaining genres: count of 12",
    }
    var_labels = {
        "racism_score": "Racism score",
        "education": "Education",
        "income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy)",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant (unstandardized intercept)",
    }

    # -----------------------------
    # Model fitting (OLS; betas post-estimation)
    # -----------------------------
    def fit_model(dv_col, model_name, stub, target_n=None):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Listwise deletion on DV + predictors
        d = d0.dropna(axis=0, how="any").copy()
        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                "Missingness shares in 1993 for model columns:\n"
                + d0.isna().mean().sort_values(ascending=False).to_string()
                + "\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop predictors with no variation on analytic sample
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
                        "Beta (standardized)": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit.pvalues.get(p, np.nan)),
                        "Status": "included",
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": var_labels.get(p, p),
                        "Beta (standardized)": np.nan,
                        "Sig": "",
                        "Status": "dropped (no variation)",
                    }
                )

        const = float(fit.params.get("const", np.nan))
        const_p = float(fit.pvalues.get("const", np.nan))
        rows.append(
            {
                "Independent Variable": var_labels["const"],
                "Beta (standardized)": const,  # explicitly unstandardized
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
                "Intercept_unstd": [const],
                "Intercept_p": [const_p],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        title = f"Bryson (1996) Table 2 replication — {model_name} (computed from provided 1993 extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"Filter: YEAR==1993. N_1993_total={int(df.shape[0])}.")
        lines.append(f"DV: {dv_labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Reporting convention:")
        lines.append("- Predictor rows: standardized OLS coefficients (beta weights) computed post-estimation as b*SD(x)/SD(y) on the analytic sample.")
        lines.append("- Constant row: unstandardized intercept from the same unstandardized OLS regression.")
        lines.append("- Stars: two-tailed p-values from that OLS fit.")
        lines.append("")
        lines.append("Key coding:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV counts: strict sum across component genres (missing if any component missing).")
        lines.append("- Racism score: strict sum across 5 dichotomous items (missing if any item missing).")
        lines.append("- Income per capita: REALINC/HOMPOP with HOMPOP>0.")
        lines.append("- Race dummies: Black=1 if RACE==2; Hispanic from ETHNIC; Other race=1 if RACE==3; White is implied reference.")
        lines.append("- Conservative Protestant: proxy RELIG==1 & DENOM==1 (baptist); missing DENOM among Protestants treated as 0.")
        lines.append("- No religion: RELIG==4. Southern: REGION==3.")
        lines.append("- Missing data: listwise deletion on DV and included predictors.")
        if target_n is not None:
            lines.append(f"- Paper target N (reference only): {int(target_n)}")
        if dropped_no_var:
            lines.append(f"- Dropped predictors (no variation): {', '.join(dropped_no_var)}")
        lines.append("")

        disp = table.copy()
        disp["Coefficient"] = disp["Beta (standardized)"].map(lambda v: fmt(v, 3))
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
        fs["Intercept_unstd"] = fs["Intercept_unstd"].map(lambda v: fmt(v, 3))
        fs["Intercept_p"] = fs["Intercept_p"].map(lambda v: fmt(v, 3))
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Intercept_unstd", "Intercept_p", "Dropped_no_variation"]].to_string(index=False))

        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))
        write_text(f"./output/{stub}_ols_unstandardized_summary.txt", fit.summary().as_text())

        # Diagnostics to help reconcile N differences
        diag = []
        diag.append(f"{model_name} diagnostics")
        diag.append("=" * (len(model_name) + 12))
        diag.append(f"N_1993_total: {int(df.shape[0])}")
        diag.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag.append(f"N_analytic_listwise: {int(d.shape[0])}")
        if target_n is not None:
            diag.append(f"N_target_from_paper: {int(target_n)}")
            diag.append(f"N_gap (analytic - target): {int(d.shape[0]) - int(target_n)}")
        diag.append("")
        diag.append("Missingness shares in 1993 for model columns (descending):")
        diag.append(d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())
        diag.append("")
        diag.append("Key value counts (1993, incl. NA):")
        diag.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag.append("\nETHNIC:\n" + value_counts_full(df["ethnic"]).to_string())
        diag.append("\nHispanic (constructed):\n" + value_counts_full(df["hispanic"]).to_string())
        diag.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        diag.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag.append("\nRacism score:\n" + value_counts_full(df["racism_score"]).to_string())
        diag.append(f"\n{dv_labels.get(dv_col, dv_col)} value counts:\n" + value_counts_full(df[dv_col]).to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d

    m1_table, m1_fit, m1_d = fit_model(
        dv1, "Model 1 (Minority-linked genres: 6)", "Table2_Model1_MinorityLinked6", target_n=644
    )
    m2_table, m2_fit, m2_d = fit_model(
        dv2, "Model 2 (Remaining genres: 12)", "Table2_Model2_Remaining12", target_n=605
    )

    # -----------------------------
    # Combined outputs
    # -----------------------------
    combined = pd.DataFrame({"Independent Variable": m1_table["Independent Variable"]}).merge(
        m1_table.rename(columns={"Beta (standardized)": "Model1_Coefficient", "Sig": "Model1_Sig", "Status": "Model1_Status"})[
            ["Independent Variable", "Model1_Coefficient", "Model1_Sig", "Model1_Status"]
        ],
        on="Independent Variable",
        how="left",
    ).merge(
        m2_table.rename(columns={"Beta (standardized)": "Model2_Coefficient", "Sig": "Model2_Sig", "Status": "Model2_Status"})[
            ["Independent Variable", "Model2_Coefficient", "Model2_Sig", "Model2_Status"]
        ],
        on="Independent Variable",
        how="outer",
    )

    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    def format_combined_for_text(dfin):
        out = dfin.copy()
        for c in ["Model1_Coefficient", "Model2_Coefficient"]:
            if c in out.columns:
                out[c] = out[c].map(lambda v: fmt(v, 3))
        return out

    title = "Bryson (1996) Table 2 replication (computed from provided 1993 extract)"
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Dependent variables:")
    lines.append(f"- Model 1 DV: {dv_labels[dv1]}")
    lines.append(f"- Model 2 DV: {dv_labels[dv2]}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Predictor rows are standardized OLS betas (post-estimation).")
    lines.append("- Constant row is the unstandardized intercept from the unstandardized OLS fit.")
    lines.append("- Hispanic is approximated from ETHNIC (codes 20..29). If your extract uses different codes, adjust this mapping.")
    lines.append("")
    lines.append("Combined coefficients")
    lines.append("---------------------")
    lines.append(format_combined_for_text(combined).to_string(index=False))
    lines.append("")
    lines.append("Fit statistics")
    lines.append("--------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Intercept_unstd"] = fs["Intercept_unstd"].map(lambda v: fmt(v, 3))
    fs["Intercept_p"] = fs["Intercept_p"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Intercept_unstd", "Intercept_p", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("Analytic sample sizes (computed) vs paper targets")
    lines.append("-------------------------------------------------")
    lines.append(f"Model 1 analytic N: {int(m1_d.shape[0])} (paper target: 644)")
    lines.append(f"Model 2 analytic N: {int(m2_d.shape[0])} (paper target: 605)")

    write_text("./output/combined_summary.txt", "\n".join(lines))
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "model1_table": m1_table,
        "model2_table": m2_table,
        "model1_analytic_sample": m1_d,
        "model2_analytic_sample": m2_d,
    }