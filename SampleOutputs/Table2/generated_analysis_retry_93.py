def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Utilities
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")

    def fmt(x, nd=3):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
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
        # 1 if 4/5; 0 if 1/2/3; missing otherwise (including DK/NA codes if present)
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        # Map to {0,1}; other values missing
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        # Missing if ANY component missing (matches "DK dropped"/listwise scale construction)
        return dfin[cols].sum(axis=1, skipna=False)

    def standardized_betas_from_unstd(fit, d_analytic, ycol, xcols):
        # beta_j = b_j * SD(x_j) / SD(y), computed on analytic sample used for fit
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
    # Load + normalize columns
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    # Numeric coercion (except id)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Restrict to 1993
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
        ["year", "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]
        + minority_genres + remaining_genres + racism_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # -----------------------------
    # Dependent variables (strict sums)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv_minority_linked"
    dv2 = "dv_remaining"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (0..5), strict 5/5 items present
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # object to >half black school: 1=yes
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # busing: 2=oppose
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # discrim: 2=no (racist-endorsing)
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # education chance: 2=no
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # motivation: 1=yes
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls / indicators
    # -----------------------------
    df["education"] = df["educ"]

    # Income per capita: REALINC/HOMPOP (HOMPOP>0)
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    df["age"] = df["age"]

    # Race dummies (reference intended: White, non-Hispanic)
    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic: best-effort from ETHNIC (extract lacks canonical HISPANIC flag).
    # Keep cases by setting 0 where ETHNIC observed, missing otherwise.
    # (This avoids artificially shrinking N by requiring ETHNIC for everyone.)
    eth = df["ethnic"]
    df["hispanic"] = np.where(eth.notna(), eth.isin([1, 2]).astype(float), np.nan)

    # If coded Hispanic, zero out other race dummies to keep "Hispanic" as separate category
    # (approximates separate Hispanic dummy with White reference; note: imperfect due to extract limits).
    hisp1 = df["hispanic"] == 1
    df.loc[hisp1 & df["black"].notna(), "black"] = 0.0
    df.loc[hisp1 & df["other_race"].notna(), "other_race"] = 0.0

    # Conservative Protestant proxy: RELIG==1 (protestant) & DENOM==1 (baptist).
    # Important: do NOT turn missing DENOM into 0; keep missing to avoid misclassification.
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.nan
    df.loc[rel.notna(), "cons_prot"] = 0.0
    df.loc[(rel == 1) & den.notna(), "cons_prot"] = (den == 1).astype(float)

    df["no_religion"] = np.where(rel.notna(), (rel == 4).astype(float), np.nan)

    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
        "racism_score",
        "education",
        "income_pc",
        "occ_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    dv_labels = {
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music",
        dv2: "Dislike of the 12 Remaining Genres",
    }

    var_labels = {
        "racism_score": "Racism score",
        "education": "Education",
        "income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting
    # -----------------------------
    def fit_model(dv_col, model_name, stub, target_n=None):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
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

        # Drop predictors with no variation (protect against singular matrix)
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

        # Build Table 2-like output: standardized betas for predictors; unstandardized constant
        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": var_labels.get(p, p),
                        "Standardized_beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit.pvalues.get(p, np.nan)),
                        "Status": "included",
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": var_labels.get(p, p),
                        "Standardized_beta": np.nan,
                        "Sig": "",
                        "Status": "dropped (no variation)",
                    }
                )

        const = float(fit.params.get("const", np.nan))
        const_p = float(fit.pvalues.get("const", np.nan))
        rows.append(
            {
                "Independent Variable": var_labels["const"],
                "Standardized_beta": const,  # intercept is not a standardized beta; kept in same column for a single table
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
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Write human-readable report
        title = f"Bryson (1996) Table 2 replication — {model_name}"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"Filter: YEAR==1993. N_1993_total={int(df.shape[0])}.")
        lines.append(f"DV: {dv_labels.get(dv_col, dv_col)}")
        lines.append("Estimator: OLS (unweighted).")
        lines.append("Displayed coefficients:")
        lines.append("- Predictors: standardized OLS coefficients (beta weights), computed as b*SD(x)/SD(y) on the analytic sample.")
        lines.append("- Constant: unstandardized intercept from the same OLS fit.")
        lines.append("Stars: two-tailed p-values from the OLS fit (* p<.05, ** p<.01, *** p<.001).")
        if target_n is not None:
            lines.append(f"Paper target N (reference only): {int(target_n)}")
        lines.append("")
        lines.append("Coding implemented:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing")
        lines.append("- DV: strict sum across component genres (missing if any component genre missing)")
        lines.append("- Racism score: strict sum across 5 dichotomous items (missing if any item missing)")
        lines.append("- Income per capita: REALINC / HOMPOP (requires HOMPOP>0)")
        lines.append("- Race dummies: Black (RACE==2), Other race (RACE==3), Hispanic (ETHNIC in {1,2} when ETHNIC observed); intended reference: White non-Hispanic")
        lines.append("- Conservative Protestant: RELIG==1 & DENOM==1 (baptist), missing if DENOM missing among Protestants")
        lines.append("- No religion: RELIG==4; Southern: REGION==3")
        lines.append("- Missing data: listwise deletion on DV + all included predictors")
        lines.append("")
        lines.append("Regression table")
        lines.append("----------------")
        disp = table.copy()
        disp["Value"] = disp["Standardized_beta"].map(lambda v: fmt(v, 3))
        lines.append(disp[["Independent Variable", "Value", "Sig", "Status"]].to_string(index=False))
        lines.append("")
        lines.append("Fit statistics")
        lines.append("--------------")
        fs = fit_stats.copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant", "Dropped_no_variation"]].to_string(index=False))
        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))

        # Save full unstandardized summary for debugging
        write_text(f"./output/{stub}_ols_unstandardized_summary.txt", fit.summary().as_text())

        # Diagnostics file to help explain N gaps
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
        diag.append("Key distributions (1993, including NA):")
        diag.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag.append("\nETHNIC:\n" + value_counts_full(df["ethnic"]).to_string())
        diag.append("\nHispanic indicator:\n" + value_counts_full(df["hispanic"]).to_string())
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
        dv1, "Model 1 (Minority-linked genres)", "Table2_Model1_MinorityLinked", target_n=644
    )
    m2_table, m2_fit, m2_d = fit_model(
        dv2, "Model 2 (Remaining genres)", "Table2_Model2_Remaining", target_n=605
    )

    # Combined table in the paper's row order
    paper_row_order = [
        "Racism score",
        "Education",
        "Household income per capita",
        "Occupational prestige",
        "Female",
        "Age",
        "Black",
        "Hispanic",
        "Other race",
        "Conservative Protestant",
        "No religion",
        "Southern",
        "Constant",
    ]

    def table_to_map(t):
        out = {}
        for _, r in t.iterrows():
            out[r["Independent Variable"]] = (r["Standardized_beta"], r["Sig"], r["Status"])
        return out

    m1_map = table_to_map(m1_table)
    m2_map = table_to_map(m2_table)

    combined_rows = []
    for lab in paper_row_order:
        b1, s1, st1 = m1_map.get(lab, (np.nan, "", "missing row"))
        b2, s2, st2 = m2_map.get(lab, (np.nan, "", "missing row"))
        combined_rows.append(
            {
                "Independent Variable": lab,
                "Model1_beta_or_const": b1,
                "Model1_Sig": s1,
                "Model1_Status": st1,
                "Model2_beta_or_const": b2,
                "Model2_Sig": s2,
                "Model2_Status": st2,
            }
        )

    combined = pd.DataFrame(combined_rows)
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    # Write combined summary
    title = "Bryson (1996) Table 2 replication — combined summary (computed from provided 1993 GSS extract)"
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Dependent variables:")
    lines.append(f"- Model 1: {dv_labels[dv1]}")
    lines.append(f"- Model 2: {dv_labels[dv2]}")
    lines.append("")
    lines.append("Note: predictor rows are standardized betas; Constant is an unstandardized intercept.")
    lines.append("")
    disp = combined.copy()
    disp["Model1_beta_or_const"] = disp["Model1_beta_or_const"].map(lambda v: fmt(v, 3))
    disp["Model2_beta_or_const"] = disp["Model2_beta_or_const"].map(lambda v: fmt(v, 3))
    lines.append("Combined coefficients")
    lines.append("---------------------")
    lines.append(
        disp[
            ["Independent Variable", "Model1_beta_or_const", "Model1_Sig", "Model2_beta_or_const", "Model2_Sig"]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("Fit statistics")
    lines.append("--------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("Analytic-sample N checkpoints (paper targets: Model 1 N=644; Model 2 N=605)")
    lines.append("----------------------------------------------------------------------------")
    lines.append(f"Model 1 analytic N: {int(m1_d.shape[0])}")
    lines.append(f"Model 2 analytic N: {int(m2_d.shape[0])}")
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