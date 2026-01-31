def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load + basic normalization
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in the input CSV.")
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    # 1993 only
    df = df.loc[df["year"] == 1993].copy()

    # Coerce numeric (keep id as-is)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Variable lists per mapping
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    required = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"]
        + minority_genres + remaining_genres + racism_raw
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing expected columns: " + ", ".join(missing_cols))

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

    def star(p):
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
        """Map to {0,1}; anything else -> missing."""
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

    def standardized_betas(fit, d, ycol, xcols):
        """
        Standardized OLS betas (beta weights):
            beta_j = b_j * sd(X_j)/sd(Y)
        computed on the analytic sample used for fit.
        """
        y = d[ycol].astype(float)
        y_sd = y.std(ddof=0)
        out = {}
        for p in xcols:
            x = d[p].astype(float)
            x_sd = x.std(ddof=0)
            if pd.isna(y_sd) or y_sd == 0 or pd.isna(x_sd) or x_sd == 0:
                out[p] = np.nan
            else:
                out[p] = float(fit.params[p] * (x_sd / y_sd))
        return out

    def value_counts_all(s):
        return pd.Series(s).value_counts(dropna=False)

    def hispanic_from_ethnic_like(eth):
        """
        Best-effort Hispanic indicator from available 'ethnic' field.

        Rules (deterministic, conservative):
        - If ETHNIC missing -> missing.
        - If values are 1/2 only -> treat 1=Hispanic, 2=Not (common binary coding).
        - Else if there is a clear "Spanish/Hispanic block" present, mark that block:
            prefer 20-39, else 200-299, else 700-799.
        - Else -> 0 for all nonmissing.
        """
        x = pd.to_numeric(eth, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        m = x.notna()
        if not m.any():
            return out

        # start as 0 for nonmissing
        out.loc[m] = 0.0

        vals = sorted(set(x[m].dropna().unique().tolist()))
        # binary case
        if set(vals).issubset({1.0, 2.0}):
            out.loc[m] = (x.loc[m] == 1.0).astype(float)
            return out

        # integer-like values only
        xi = np.floor(x[m]).astype("Int64")
        is_intlike = (x[m] - xi.astype(float)).abs() < 1e-9
        idx = x[m].index[is_intlike.values]
        e = xi[is_intlike].astype(int)
        present = set(e.unique().tolist())

        blocks = [
            (20, 39),
            (200, 299),
            (700, 799),
        ]
        chosen = None
        for lo, hi in blocks:
            if any((v >= lo and v <= hi) for v in present):
                chosen = (lo, hi)
                break
        if chosen is None:
            return out

        lo, hi = chosen
        out.loc[idx] = e.between(lo, hi).astype(float).values
        return out

    # -----------------------------
    # Dependent variables (strict counts per mapping)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (0-5), strict 5/5 items per mapping
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls (preserve missingness; listwise deletion handled per model)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies (ref = white)
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: try to construct from 'ethnic' if available; otherwise raise (faithful replication requires it)
    if "ethnic" in df.columns:
        df["hispanic"] = hispanic_from_ethnic_like(df["ethnic"])
    else:
        raise ValueError("Expected 'ethnic' column to construct Hispanic indicator, but it was not found.")

    # Religion dummies (preserve missingness)
    # RELIG categories in the provided extract appear numeric; mapping says 4 == none.
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: limited by extract (no detailed denom tradition codes).
    # Keep the simplest available proxy but DO NOT impute missing as 0.
    # Proxy: Protestant (RELIG==1) & Baptist (DENOM==1).
    df["cons_prot"] = np.nan
    m_rel = df["relig"].notna() & df["denom"].notna()
    df.loc[m_rel, "cons_prot"] = ((df.loc[m_rel, "relig"] == 1) & (df.loc[m_rel, "denom"] == 1)).astype(float)

    # Southern (mapping instruction: REGION==3 is South). Preserve missingness.
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

    labels = {
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music",
        dv2: "Dislike of the 12 Remaining Genres",
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
    # Model fitting (keep as faithful as possible)
    # - Listwise deletion on DV + predictors
    # - Drop no-variation predictors only if necessary to avoid singular fit
    # - Stars computed from this run's unstandardized OLS p-values
    # -----------------------------
    def fit_model(dv_col, model_name, stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # If sample is unexpectedly tiny, save diagnostics to help fix upstream coding
        if d.shape[0] < 50:
            diag = []
            diag.append(f"{model_name}: analytic sample too small (N={d.shape[0]}).")
            diag.append("")
            diag.append("Missingness shares (1993) for model columns (descending):")
            diag.append(d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())
            diag.append("")
            diag.append("RELIG value counts (1993):")
            diag.append(value_counts_all(df["relig"]).to_string())
            diag.append("")
            diag.append("DENOM value counts (1993):")
            diag.append(value_counts_all(df["denom"]).to_string())
            diag.append("")
            diag.append("RACE value counts (1993):")
            diag.append(value_counts_all(df["race"]).to_string())
            diag.append("")
            if "ethnic" in df.columns:
                diag.append("ETHNIC value counts (1993, top 60):")
                diag.append(value_counts_all(df["ethnic"]).head(60).to_string())
                diag.append("")
            write_text(f"./output/{stub}_SMALLN_DIAGNOSTICS.txt", "\n".join(diag))

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                f"Missingness shares (1993) for model columns:\n"
                f"{d0.isna().mean().sort_values(ascending=False).to_string()}\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        kept, dropped = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        betas = standardized_betas(fit, d, dv_col, kept)

        table = pd.DataFrame(
            [
                {
                    "Independent Variable": labels.get(p, p),
                    "Std_Beta": (betas.get(p, np.nan) if p in kept else np.nan),
                    "Sig": (star(fit.pvalues.get(p, np.nan)) if p in kept else ""),
                }
                for p in predictors
            ]
        )

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit.nobs))],
                "R2": [float(fit.rsquared)],
                "Adj_R2": [float(fit.rsquared_adj)],
                "Constant": [float(fit.params.get("const", np.nan))],
                "Constant_Sig": [star(fit.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped) if dropped else ""],
            }
        )

        # Human-readable report
        title = f"Bryson (1996) Table 2 — {model_name} (computed from provided GSS 1993 extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)} (count)")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_j = b_j * sd(X_j)/sd(Y), computed on this model's analytic sample.")
        lines.append("Stars: two-tailed p-values from this run's unstandardized OLS model (replication stars).")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Income per capita: REALINC/HOMPOP (HOMPOP must be >0)")
        lines.append("- Southern dummy: REGION==3")
        lines.append("- Missing data: strict listwise deletion on DV + all predictors")
        if dropped:
            lines.append("")
            lines.append("Dropped due to no variation in analytic sample:")
            for p in dropped:
                lines.append(f"- {p}: {labels.get(p, p)}")
        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp[["Independent Variable", "Std_Beta", "Sig"]].to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats.copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs[["Model", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].to_string(index=False))

        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))

        # Save full OLS summary
        with open(f"./output/{stub}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Diagnostics useful for reconciliation (focus on the prior failures: N collapse and dummy variation)
        diag_lines = []
        diag_lines.append(f"{model_name} diagnostics")
        diag_lines.append("=" * (len(model_name) + 12))
        diag_lines.append(f"N_1993_total: {int(df.shape[0])}")
        diag_lines.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag_lines.append(f"N_analytic_listwise: {int(d.shape[0])}")
        diag_lines.append("")
        diag_lines.append("Missingness shares in 1993 for model columns (descending):")
        diag_lines.append(d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())
        diag_lines.append("")
        diag_lines.append("Value counts (analytic sample) for key dummies:")
        for v in ["female", "black", "hispanic", "other_race", "cons_prot", "no_religion", "southern"]:
            diag_lines.append(f"\n{v} ({labels.get(v, v)}):")
            diag_lines.append(value_counts_all(d[v]).to_string())
        diag_lines.append("")
        diag_lines.append("Underlying raw distributions (1993, pre-listwise):")
        diag_lines.append("\nRELIG value counts:\n" + value_counts_all(df["relig"]).to_string())
        diag_lines.append("\nDENOM value counts:\n" + value_counts_all(df["denom"]).to_string())
        diag_lines.append("\nREGION value counts:\n" + value_counts_all(df["region"]).to_string())
        if "ethnic" in df.columns:
            diag_lines.append("\nETHNIC value counts (top 80):\n" + value_counts_all(df["ethnic"]).head(80).to_string())
        diag_lines.append("\nRACE value counts:\n" + value_counts_all(df["race"]).to_string())
        diag_lines.append("")
        diag_lines.append("nunique (analytic sample) by predictor:")
        diag_lines.append(d[predictors].nunique(dropna=True).sort_values().to_string())
        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag_lines))

        # CSV outputs
        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d

    m1_table, m1_fit, m1_d = fit_model(dv1, "Model 1 (Minority-linked genres: 6)", "Table2_Model1_MinorityLinked6")
    m2_table, m2_fit, m2_d = fit_model(dv2, "Model 2 (Remaining genres: 12)", "Table2_Model2_Remaining12")

    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "Model1_Std_Beta": m1_table["Std_Beta"],
            "Model1_Sig": m1_table["Sig"],
            "Model2_Std_Beta": m2_table["Std_Beta"],
            "Model2_Sig": m2_table["Sig"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    def dv_descriptives(series):
        s = pd.to_numeric(series, errors="coerce")
        return {
            "N": int(s.notna().sum()),
            "Mean": float(s.mean()) if s.notna().any() else np.nan,
            "SD": float(s.std(ddof=0)) if s.notna().any() else np.nan,
            "Min": float(s.min()) if s.notna().any() else np.nan,
            "P25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "Median": float(s.quantile(0.50)) if s.notna().any() else np.nan,
            "P75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "Max": float(s.max()) if s.notna().any() else np.nan,
        }

    dv_desc = pd.DataFrame(
        [
            {"Sample": "All 1993 (DV constructed nonmissing)", "DV": labels[dv1], **dv_descriptives(df[dv1])},
            {"Sample": "All 1993 (DV constructed nonmissing)", "DV": labels[dv2], **dv_descriptives(df[dv2])},
            {"Sample": "Model 1 analytic sample", "DV": labels[dv1], **dv_descriptives(m1_d[dv1])},
            {"Sample": "Model 2 analytic sample", "DV": labels[dv2], **dv_descriptives(m2_d[dv2])},
        ]
    )

    # Summary text
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars (from this run)")
    lines.append("--------------------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["Model1_Std_Beta"] = tmp["Model1_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["Model2_Std_Beta"] = tmp["Model2_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; from this run)")
    lines.append("--------------------------------------------------")
    fs = combined_fit.copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].to_string(index=False))
    lines.append("")
    lines.append("DV descriptives (counts)")
    lines.append("------------------------")
    dv_desc_fmt = dv_desc.copy()
    for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
        dv_desc_fmt[c] = dv_desc_fmt[c].map(lambda v: fmt(v, 3))
    dv_desc_fmt["N"] = dv_desc_fmt["N"].map(lambda v: fmt(v, 0))
    lines.append(dv_desc_fmt[["Sample", "DV", "N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"]].to_string(index=False))
    lines.append("")
    lines.append("Implementation notes")
    lines.append("--------------------")
    lines.append("- Strict listwise deletion is applied separately per model (DV + all predictors).")
    lines.append("- Hispanic is constructed from the available 'ethnic' field using deterministic rules (see diagnostics).")
    lines.append("- Conservative Protestant is a coarse proxy (RELIG==1 & DENOM==1) because the extract lacks finer tradition coding.")
    lines.append("- No missing values are imputed to 0 for dummies; missingness is preserved to better match listwise handling.")

    write_text("./output/combined_summary.txt", "\n".join(lines))

    # Save combined artifacts
    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)
    dv_desc.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc,
        "model1_table": m1_table,
        "model2_table": m2_table,
        "model1_analytic_sample": m1_d,
        "model2_analytic_sample": m2_d,
    }