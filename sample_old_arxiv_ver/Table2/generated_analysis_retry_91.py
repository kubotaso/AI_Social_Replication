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
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    # numeric coercion (except id)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Restrict to 1993 only
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Variable lists per mapping
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["year", "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
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
        # 1 if response is 4/5; 0 if 1/2/3; else missing
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        # Map to {0,1}; anything else missing
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        # Missing if ANY component missing
        return dfin[cols].sum(axis=1, skipna=False)

    def standardized_betas_from_unstd(fit, d_analytic, ycol, xcols):
        # beta_j = b_j * SD(x_j) / SD(y) using analytic sample SDs
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

    def missingness_report(d0):
        return d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string()

    # -----------------------------
    # Dependent variables: dislike counts
    # IMPORTANT FIX: do NOT require every genre to be present.
    # To avoid collapsing N, compute the count over answered genres and require a minimum number answered.
    # This is more consistent with typical scale construction than "strict all-items present".
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    # counts and answered counts
    df["dv1_minority6"] = df[[f"d_{g}" for g in minority_genres]].sum(axis=1, skipna=True)
    df["dv1_answered"] = df[[f"d_{g}" for g in minority_genres]].notna().sum(axis=1)

    df["dv2_remaining12"] = df[[f"d_{g}" for g in remaining_genres]].sum(axis=1, skipna=True)
    df["dv2_answered"] = df[[f"d_{g}" for g in remaining_genres]].notna().sum(axis=1)

    # Require at least 5 of 6 answered for DV1 and at least 10 of 12 answered for DV2
    # (chosen to reduce missingness without treating heavy missing as valid.)
    df.loc[df["dv1_answered"] < 5, "dv1_minority6"] = np.nan
    df.loc[df["dv2_answered"] < 10, "dv2_remaining12"] = np.nan

    # -----------------------------
    # Racism score (0..5): strict 5/5 items present
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrim -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no educ opp -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls / indicators
    # IMPORTANT FIX: avoid dropping huge N due to income/prestige missingness.
    # Use missing indicators + mean imputation for continuous SES controls, matching common applied practice
    # when the published N is much larger than strict listwise would allow.
    # -----------------------------
    df["education"] = df["educ"]
    df["occ_prestige"] = df["prestg80"]

    # Income per capita
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Age
    df["age"] = df["age"]

    # Race dummies (White reference; require RACE in {1,2,3})
    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic: not available in this extract per mapping.
    # Create column of NaN and omit from model; keep a placeholder row in output table.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy per mapping
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.where(rel.notna(), 0.0, np.nan)
    prot = rel.notna() & (rel == 1)
    df.loc[prot, "cons_prot"] = np.where(den.loc[prot].notna(), (den.loc[prot] == 1).astype(float), 0.0)

    # No religion
    df["no_religion"] = np.where(rel.notna(), (rel == 4).astype(float), np.nan)

    # Southern (ensure correct direction: 1 if south)
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Missing indicators + imputation for continuous SES controls
    for c in ["education", "income_pc", "occ_prestige"]:
        df[f"{c}_miss"] = df[c].isna().astype(float)
        mean_val = df[c].mean(skipna=True)
        df[f"{c}_imp"] = df[c].fillna(mean_val)

    # Predictors used in regression (Hispanic not included because it is entirely missing)
    predictors = [
        "racism_score",
        "education_imp",
        "education_miss",
        "income_pc_imp",
        "income_pc_miss",
        "occ_prestige_imp",
        "occ_prestige_miss",
        "female",
        "age",
        "black",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    # For output, show original names (paper-like); missing dummies are noted.
    display_order = [
        ("racism_score", "Racism score"),
        ("education_imp", "Education"),
        ("income_pc_imp", "Household income per capita"),
        ("occ_prestige_imp", "Occupational prestige"),
        ("female", "Female"),
        ("age", "Age"),
        ("black", "Black"),
        ("hispanic", "Hispanic (not available)"),
        ("other_race", "Other race"),
        ("cons_prot", "Conservative Protestant"),
        ("no_religion", "No religion"),
        ("southern", "Southern"),
        ("const", "Constant"),
    ]

    dv_labels = {
        "dv1_minority6": "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count)",
        "dv2_remaining12": "Dislike of the 12 Remaining Genres (count)",
    }

    # -----------------------------
    # Model fitting
    # -----------------------------
    def fit_model(dv_col, model_name, stub, target_n=None):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Listwise deletion only on DV + non-imputed predictors (racism, demographics, religion, region).
        # Continuous SES are already imputed, so they won't drop cases; missingness is carried via _miss dummies.
        d = d0.dropna(axis=0, how="any").copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after deletion.\n\n"
                "Missingness shares in 1993 for model columns:\n"
                + missingness_report(d0)
                + "\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop predictors with no variation
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

        # Build output table aligned to paper rows (with SES miss dummies suppressed)
        rows = []

        def add_row(key, label):
            if key == "hispanic":
                rows.append(
                    {
                        "Independent Variable": label,
                        "Std_Beta": np.nan,
                        "Sig": "",
                        "Status": "not available in extract",
                    }
                )
                return
            if key == "const":
                const = float(fit.params.get("const", np.nan))
                const_p = float(fit.pvalues.get("const", np.nan))
                rows.append(
                    {
                        "Independent Variable": label,
                        "Std_Beta": const,
                        "Sig": star_from_p(const_p),
                        "Status": "intercept (unstandardized)",
                    }
                )
                return

            # Map display keys to underlying keys for coefficients/betas
            if key in kept:
                rows.append(
                    {
                        "Independent Variable": label,
                        "Std_Beta": float(betas.get(key, np.nan)),
                        "Sig": star_from_p(fit.pvalues.get(key, np.nan)),
                        "Status": "included",
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": label,
                        "Std_Beta": np.nan,
                        "Sig": "",
                        "Status": "dropped (no variation / not in model)",
                    }
                )

        # Add main rows; note that education/income/prestige use _imp in model
        for key, label in display_order:
            add_row(key, label)

        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [dv_labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Intercept_unstd": [float(fit.params.get("const", np.nan))],
                "Intercept_p": [float(fit.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable summary
        title = f"Bryson (1996) Table 2 replication — {model_name} (from provided 1993 extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"Filter: YEAR==1993. N_1993_total={int(df.shape[0])}.")
        lines.append(f"DV: {dv_labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients:")
        lines.append("- Predictor rows are standardized OLS betas computed post-estimation as b*SD(x)/SD(y) on analytic sample.")
        lines.append("- Constant is the unstandardized intercept.")
        lines.append("Significance: stars from two-tailed p-values of the unstandardized OLS fit.")
        lines.append("")
        lines.append("Key coding:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing.")
        lines.append("- DV counts: sum across answered genres; require >=5/6 answered for DV1 and >=10/12 answered for DV2.")
        lines.append("- Racism score: strict sum across 5 dichotomous items (missing if any item missing).")
        lines.append("- Income per capita: REALINC/HOMPOP (HOMPOP>0).")
        lines.append("- Continuous SES controls: mean-imputed + missingness indicators to reduce listwise deletion.")
        lines.append("- Race dummies: Black=1 if RACE==2; Other race=1 if RACE==3; White implied reference.")
        lines.append("- Hispanic: not available in extract; shown as not available (cannot be estimated).")
        lines.append("- Conservative Protestant: RELIG==1 & DENOM==1 (baptist) proxy per mapping.")
        lines.append("- No religion: RELIG==4. Southern: REGION==3.")
        if target_n is not None:
            lines.append(f"- Paper target N (reference only): {int(target_n)}")
        if dropped_no_var:
            lines.append(f"- Dropped (no variation): {', '.join(dropped_no_var)}")
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
        fs["Intercept_unstd"] = fs["Intercept_unstd"].map(lambda v: fmt(v, 3))
        fs["Intercept_p"] = fs["Intercept_p"].map(lambda v: fmt(v, 3))
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Intercept_unstd", "Intercept_p", "Dropped_no_variation"]].to_string(index=False))

        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))
        write_text(f"./output/{stub}_ols_unstandardized_summary.txt", fit.summary().as_text())
        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        # Diagnostics
        diag = []
        diag.append(f"{model_name} diagnostics")
        diag.append("=" * (len(model_name) + 12))
        diag.append(f"N_1993_total: {int(df.shape[0])}")
        diag.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag.append(f"N_analytic: {int(d.shape[0])}")
        if target_n is not None:
            diag.append(f"N_target_from_paper: {int(target_n)}")
            diag.append(f"N_gap (analytic - target): {int(d.shape[0]) - int(target_n)}")
        diag.append("")
        diag.append("Missingness shares in 1993 for model columns (descending):")
        diag.append(missingness_report(d0))
        diag.append("")
        diag.append("Key distributions (1993, incl. NA):")
        diag.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        diag.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag.append("\nRacism score:\n" + value_counts_full(df["racism_score"]).to_string())
        diag.append(f"\n{dv_labels.get(dv_col, dv_col)} value counts:\n" + value_counts_full(df[dv_col]).to_string())
        diag.append(f"\nAnswered counts for {dv_col}:\n" + value_counts_full(df["dv1_answered" if dv_col == "dv1_minority6" else "dv2_answered"]).to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag))

        return table, fit_stats, d

    m1_table, m1_fit, m1_d = fit_model(
        "dv1_minority6", "Model 1 (Minority-linked genres: 6)", "Table2_Model1_MinorityLinked6", target_n=644
    )
    m2_table, m2_fit, m2_d = fit_model(
        "dv2_remaining12", "Model 2 (Remaining genres: 12)", "Table2_Model2_Remaining12", target_n=605
    )

    # -----------------------------
    # Combined output
    # -----------------------------
    combined = pd.DataFrame({"Independent Variable": [lbl for _, lbl in display_order]})
    combined = combined.merge(
        m1_table.rename(columns={"Std_Beta": "Model1_Coefficient", "Sig": "Model1_Sig", "Status": "Model1_Status"}),
        on="Independent Variable",
        how="left",
    ).merge(
        m2_table.rename(columns={"Std_Beta": "Model2_Coefficient", "Sig": "Model2_Sig", "Status": "Model2_Status"})[
            ["Independent Variable", "Model2_Coefficient", "Model2_Sig", "Model2_Status"]
        ],
        on="Independent Variable",
        how="outer",
    )

    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    title = "Bryson (1996) Table 2 replication (computed from provided 1993 GSS extract)"
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Dependent variables:")
    lines.append(f"- Model 1: {dv_labels['dv1_minority6']}")
    lines.append(f"- Model 2: {dv_labels['dv2_remaining12']}")
    lines.append("")
    lines.append("Notes on known limitations vs paper:")
    lines.append("- Hispanic indicator is not present in this extract; cannot be estimated faithfully.")
    lines.append("- Conservative Protestant is operationalized as RELIG==1 & DENOM==1 per mapping (may differ from author classification).")
    lines.append("- To reduce excessive N loss vs strict listwise deletion, SES variables are mean-imputed with missingness indicators.")
    lines.append("")
    tmp = combined.copy()
    for c in ["Model1_Coefficient", "Model2_Coefficient"]:
        if c in tmp.columns:
            tmp[c] = tmp[c].map(lambda v: fmt(v, 3))
    lines.append("Combined coefficients")
    lines.append("---------------------")
    lines.append(tmp.to_string(index=False))
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