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

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # Coerce to numeric (except id)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Required columns (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["year", "id", "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_raw
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
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

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

    def dislike_indicator(series):
        """
        1 if response is 4/5 (dislike/dislike very much),
        0 if response is 1/2/3,
        missing otherwise.
        """
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def dich(series, ones, zeros):
        """Map to {0,1}; anything else missing."""
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

    def standardized_betas_from_unstd(fit, d, ycol, xcols):
        """
        Standardized betas from unstandardized coefficients:
            beta_j = b_j * SD(x_j) / SD(y)
        computed on analytic sample (ddof=0).
        """
        y = pd.to_numeric(d[ycol], errors="coerce").astype(float)
        sd_y = y.std(ddof=0)
        betas = {}
        for x in xcols:
            sx = pd.to_numeric(d[x], errors="coerce").astype(float)
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
    # Dependent variables: strict dislike counts
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv_minority_linked_6"
    dv2 = "dv_remaining_12"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (0..5): strict 5/5 items present (per mapping)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no education chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # missing if any item missing

    # -----------------------------
    # Controls / indicators
    # -----------------------------
    df["education"] = df["educ"]

    # Income per capita: REALINC/HOMPOP (HOMPOP>0)
    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    # Female indicator: 1 if SEX==2, 0 if SEX==1, else missing
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Age
    df["age_years"] = df["age"]

    # Race dummies: reference = White (RACE==1)
    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic: not available in provided extract; keep as all-missing so it is excluded safely
    df["hispanic"] = np.nan

    # Conservative Protestant (proxy, but kept simple and does not create extra missingness for non-Protestants):
    # If RELIG missing -> missing; else default 0; if Protestant then 1 iff DENOM==1 else 0 (DENOM missing => 0).
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.where(rel.notna(), 0.0, np.nan)
    prot = rel.notna() & (rel == 1)
    df.loc[prot, "cons_prot"] = np.where(den.loc[prot].notna(), (den.loc[prot] == 1).astype(float), 0.0)

    # No religion: RELIG==4 (missing if RELIG missing)
    df["no_religion"] = np.where(rel.notna(), (rel == 4).astype(float), np.nan)

    # Southern: REGION==3 (missing if REGION missing)
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictor order matches paper
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
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music",
        dv2: "Dislike of the 12 Remaining Genres",
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
        "cons_prot": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting (simple, faithful): listwise deletion across DV + available predictors
    # -----------------------------
    def fit_model(dv_col, model_name, stub, target_n=None):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Drop predictors that are entirely missing to avoid collapsing N (fixes prior runtime/sample collapse)
        usable_predictors = [p for p in predictors if d0[p].notna().any()]
        dropped_all_missing = [p for p in predictors if p not in usable_predictors]

        # Listwise deletion on DV + usable predictors
        d = d0[[dv_col] + usable_predictors].dropna(axis=0, how="any").copy()
        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                "Missingness shares in 1993 for model columns:\n"
                + d0.isna().mean().sort_values(ascending=False).to_string()
                + "\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop predictors with no variation in analytic sample
        kept, dropped_no_var = [], []
        for p in usable_predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(fit_unstd, d, dv_col, kept)

        # Output table in paper order; intercept reported as unstandardized constant (not a beta)
        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": var_labels.get(p, p),
                        "Standardized beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(float(fit_unstd.pvalues.get(p, np.nan))),
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": var_labels.get(p, p),
                        "Standardized beta": np.nan,
                        "Sig": "",
                    }
                )

        rows.append(
            {
                "Independent Variable": var_labels["const"],
                "Standardized beta": np.nan,
                "Sig": star_from_p(float(fit_unstd.pvalues.get("const", np.nan))),
            }
        )
        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [dv_labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant_unstd": [float(fit_unstd.params.get("const", np.nan))],
                "Dropped_all_missing": [", ".join(dropped_all_missing) if dropped_all_missing else ""],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable output
        title = f"Bryson (1996) Table 2 replication — {model_name} (computed from provided 1993 GSS extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {dv_labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Intercept: reported as UNSTANDARDIZED constant from the same OLS model (as in published tables).")
        lines.append("Stars: two-tailed p-values from conventional OLS in this run.")
        lines.append("")
        lines.append("Construction rules implemented:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum across 5 dichotomous items (missing if any item missing)")
        lines.append("- Income per capita: REALINC / HOMPOP (HOMPOP>0 required)")
        lines.append("- Race dummies: Black=1 if RACE==2; Other race=1 if RACE==3; reference is White (RACE==1)")
        lines.append("- Hispanic: not available in this extract; excluded automatically if all-missing")
        lines.append("")
        if target_n is not None:
            lines.append(f"Target N in paper (reference only): {int(target_n)}")
        if dropped_all_missing:
            lines.append(f"Dropped (all missing in this extract): {', '.join(dropped_all_missing)}")
        if dropped_no_var:
            lines.append(f"Dropped (no variation in analytic sample): {', '.join(dropped_no_var)}")

        lines.append("")
        lines.append("Regression table (Table 2 style)")
        lines.append("-------------------------------")
        tmp = table.copy()
        tmp["Standardized beta"] = tmp["Standardized beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp[["Independent Variable", "Standardized beta", "Sig"]].to_string(index=False))

        lines.append("")
        lines.append("Fit statistics")
        lines.append("--------------")
        fs = fit_stats.copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant_unstd"] = fs["Constant_unstd"].map(lambda v: fmt(v, 3))
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant_unstd"]].to_string(index=False))

        # SE note (paper doesn't provide SEs)
        lines.append("")
        lines.append("Note on standard errors:")
        lines.append("- Bryson (1996) Table 2 does not report standard errors; this replication does not attempt SE matching.")

        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))
        write_text(f"./output/{stub}_ols_unstandardized_summary.txt", fit_unstd.summary().as_text())

        # Diagnostics: N and missingness
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
        diag.append("Value counts (1993, including NA):")
        diag.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        diag.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag.append("\nRacism score:\n" + value_counts_full(df["racism_score"]).to_string())
        diag.append(f"\n{dv_labels.get(dv_col, dv_col)} value counts:\n" + value_counts_full(df[dv_col]).to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d, kept, dropped_all_missing, dropped_no_var, fit_unstd

    m1_table, m1_fit, m1_d, m1_kept, m1_drop_all, m1_drop_novar, _ = fit_model(
        dv1, "Model 1", "Table2_Model1_MinorityLinked6", target_n=644
    )
    m2_table, m2_fit, m2_d, m2_kept, m2_drop_all, m2_drop_novar, _ = fit_model(
        dv2, "Model 2", "Table2_Model2_Remaining12", target_n=605
    )

    # -----------------------------
    # Combined summary outputs
    # -----------------------------
    combined = m1_table.merge(
        m2_table,
        on="Independent Variable",
        how="outer",
        suffixes=(" (Model 1)", " (Model 2)"),
    )

    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    title = "Bryson (1996) Table 2 replication (computed from provided 1993 GSS extract)"
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Dependent variables:")
    lines.append(f"- Model 1: {dv_labels[dv1]}")
    lines.append(f"- Model 2: {dv_labels[dv2]}")
    lines.append("")
    lines.append("Combined regression table (standardized betas; constant is unstandardized)")
    lines.append("-----------------------------------------------------------------------")
    tmp = combined.copy()
    for c in ["Standardized beta (Model 1)", "Standardized beta (Model 2)"]:
        if c in tmp.columns:
            tmp[c] = tmp[c].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics")
    lines.append("--------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant_unstd"] = fs["Constant_unstd"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant_unstd", "Dropped_all_missing", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("Analytic-sample N checkpoints (paper targets: Model 1 N=644; Model 2 N=605)")
    lines.append("----------------------------------------------------------------------------")
    lines.append(f"Model 1 analytic N: {int(m1_d.shape[0])}")
    lines.append(f"Model 2 analytic N: {int(m2_d.shape[0])}")
    lines.append("")
    lines.append("SE note:")
    lines.append("- Published Table 2 does not provide standard errors; this output does not attempt SE matching.")

    write_text("./output/combined_summary.txt", "\n".join(lines))
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)

    return {
        "model1_table": m1_table,
        "model2_table": m2_table,
        "combined_table": combined,
        "fit_stats": combined_fit,
        "model1_analytic_sample": m1_d,
        "model2_analytic_sample": m2_d,
    }