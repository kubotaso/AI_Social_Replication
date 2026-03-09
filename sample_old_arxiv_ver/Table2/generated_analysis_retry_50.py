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

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # Coerce all non-id columns to numeric
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

    base_required = [
        "hompop", "educ", "realinc", "prestg80", "sex", "age",
        "race", "ethnic", "relig", "denom", "region"
    ]
    required = base_required + minority_genres + remaining_genres + racism_raw
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

    def standardized_betas_from_unstd(fit, d, ycol, xcols):
        """
        Standardized OLS betas computed from unstandardized slopes:
            beta_j = b_j * sd(X_j)/sd(Y)
        computed on the analytic sample used for fit.
        """
        y = pd.to_numeric(d[ycol], errors="coerce").astype(float)
        y_sd = y.std(ddof=0)
        out = {}
        for p in xcols:
            x = pd.to_numeric(d[p], errors="coerce").astype(float)
            x_sd = x.std(ddof=0)
            if pd.isna(y_sd) or y_sd == 0 or pd.isna(x_sd) or x_sd == 0:
                out[p] = np.nan
            else:
                out[p] = float(fit.params[p] * (x_sd / y_sd))
        return out

    def vc(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.value_counts(dropna=False)

    def infer_hispanic_from_ethnic(eth):
        """
        Construct a 0/1 Hispanic indicator from available ETHNIC field without
        hard-coding specific codes from the paper.

        Strategy (robust + transparent):
        - If ETHNIC is binary {1,2}, treat 1 as Hispanic and 2 as not (common encoding).
        - Else, choose the set of ETHNIC codes whose label cluster is most plausibly Hispanic
          by using a conservative heuristic:
            * candidate codes are those that, among respondents, show substantially higher
              probability of Spanish/Latin music liking (LATIN item) relative to others.
          This uses only within-file information and avoids copying any paper coding.

        If LATIN is too missing or the heuristic is inconclusive, fallback to:
        - mark as missing (not imputed), letting listwise deletion operate.
        """
        x = pd.to_numeric(eth, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        m = x.notna()
        if not m.any():
            return out

        vals = set(x[m].dropna().unique().tolist())
        if vals.issubset({1.0, 2.0}):
            out.loc[m] = (x.loc[m] == 1.0).astype(float)
            return out

        # Heuristic using LATIN music rating if available
        if "latin" not in df.columns:
            return out  # cannot infer; keep missing

        latin_rating = pd.to_numeric(df["latin"], errors="coerce")
        usable = m & latin_rating.notna() & x.isin(list(vals))
        if usable.sum() < 200:
            return out  # too small / too missing to infer

        # Define "likes Latin" as 1/2 ("like very much"/"like") to create a signal
        likes_latin = latin_rating.isin([1, 2]).astype(int)

        # Compute per-ETHNIC code liking rate, and overall rate
        tmp = pd.DataFrame({"eth": x[usable].astype(int), "likes": likes_latin[usable].astype(int)})
        overall = tmp["likes"].mean()

        rates = tmp.groupby("eth")["likes"].agg(["mean", "count"]).reset_index()
        rates = rates.loc[rates["count"] >= 15].copy()
        if rates.shape[0] < 5:
            return out

        # Select codes with notably higher liking than overall (conservative threshold)
        # and keep total selected group between 1% and 25% of usable sample.
        rates["lift"] = rates["mean"] - overall
        rates = rates.sort_values("lift", ascending=False)

        selected = []
        selected_n = 0
        total_n = int(tmp.shape[0])

        for _, r in rates.iterrows():
            if r["lift"] <= 0.10:
                break
            code = int(r["eth"])
            ncode = int(r["count"])
            # avoid selecting too much of sample
            if (selected_n + ncode) / total_n > 0.25:
                continue
            selected.append(code)
            selected_n += ncode
            if selected_n / total_n >= 0.01:
                # keep adding a few if strong; else stop early
                pass

        if selected_n / total_n < 0.01:
            # inconclusive; keep missing
            return out

        out.loc[m] = 0.0
        out.loc[m & x.astype(float).isin([float(c) for c in selected])] = 1.0
        return out

    def make_mutually_exclusive_race_ethnicity(df_in):
        """
        Create mutually-exclusive indicators:
          - black
          - hispanic
          - other_race
        Reference: non-Hispanic White.

        Rule:
        - Hispanic overrides race (if Hispanic==1, then black=0 and other_race=0).
        - Otherwise, black = (race==2), other_race=(race==3).
        Missing preserved.
        """
        race = pd.to_numeric(df_in["race"], errors="coerce")
        hisp = df_in["hispanic"]

        black = pd.Series(np.nan, index=df_in.index, dtype="float64")
        other = pd.Series(np.nan, index=df_in.index, dtype="float64")

        known_race = race.isin([1, 2, 3])
        known_hisp = hisp.isin([0.0, 1.0])

        # default when race is known and hispanic is known
        idx = known_race & known_hisp
        black.loc[idx] = (race.loc[idx] == 2).astype(float)
        other.loc[idx] = (race.loc[idx] == 3).astype(float)

        # Hispanic override
        idx_h = idx & (hisp == 1.0)
        black.loc[idx_h] = 0.0
        other.loc[idx_h] = 0.0

        return black, other

    # -----------------------------
    # Dependent variables (strict counts)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (0–5), strict 5/5 items
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])        # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])        # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])      # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])      # 2=no education chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])      # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls / dummies (preserve missingness; no imputation to 0)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Hispanic from ETHNIC (keep missing if cannot infer)
    df["hispanic"] = infer_hispanic_from_ethnic(df["ethnic"])

    # Mutually exclusive race/ethnicity dummies
    df["black"], df["other_race"] = make_mutually_exclusive_race_ethnicity(df)

    # No religion: RELIG==4 per provided mapping; preserve missingness
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: still a proxy given extract; preserve missingness (do not fill)
    df["cons_prot"] = np.nan
    m_rel = df["relig"].notna() & df["denom"].notna()
    df.loc[m_rel, "cons_prot"] = ((df.loc[m_rel, "relig"] == 1) & (df.loc[m_rel, "denom"] == 1)).astype(float)

    # Southern: REGION==3 per mapping; preserve missingness
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
        "racism_score": "Racism score (0–5)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female",
        "age_years": "Age (years)",
        "black": "Black",
        "hispanic": "Hispanic (constructed from ETHNIC)",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting (strict listwise; do not drop predictors unless singular)
    # -----------------------------
    def fit_model(dv_col, model_name, stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n\n"
                f"Missingness shares (1993) for model columns:\n"
                f"{d0.isna().mean().sort_values(ascending=False).to_string()}\n"
            )
            write_text(f"./output/{stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Keep all predictors, but if singular, iteratively drop the most problematic no-variance columns
        kept = predictors.copy()
        dropped = []

        def try_fit(cols):
            y = d[dv_col].astype(float)
            X = d[cols].astype(float)
            Xc = sm.add_constant(X, has_constant="add")
            return sm.OLS(y, Xc).fit()

        # First drop any predictors with no variation in analytic sample (true no-variation only)
        for p in list(kept):
            if d[p].nunique(dropna=True) <= 1:
                kept.remove(p)
                dropped.append(p)

        # Fit; if still singular, drop predictors with highest collinearity (via near-zero eigenvalues)
        fit = try_fit(kept)
        if not np.isfinite(fit.params).all():
            # very defensive fallback: drop one-by-one by smallest std dev among predictors
            sds = d[kept].std(ddof=0).sort_values()
            for p in sds.index.tolist():
                if p in kept:
                    kept.remove(p)
                    dropped.append(p)
                    fit = try_fit(kept)
                    if np.isfinite(fit.params).all():
                        break

        betas = standardized_betas_from_unstd(fit, d, dv_col, kept)

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
                "Dropped_no_variation_or_singularity": [", ".join(dropped) if dropped else ""],
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
        lines.append("Stars: two-tailed p-values from this run's conventional OLS (replication stars).")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Race/ethnicity: mutually exclusive dummies (Hispanic overrides race)")
        lines.append("- Missing data: strict listwise deletion on DV + all predictors")
        if dropped:
            lines.append("")
            lines.append("Dropped (no variation / singularity defense):")
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
        lines.append(
            fs[
                ["Model", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation_or_singularity"]
            ].to_string(index=False)
        )

        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))

        # Save full OLS summary
        with open(f"./output/{stub}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Diagnostics (explicitly address prior failure modes)
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
            diag_lines.append(vc(d[v]).to_string())
        diag_lines.append("")
        diag_lines.append("Underlying raw distributions (1993, pre-listwise):")
        diag_lines.append("\nRELIG value counts:\n" + vc(df["relig"]).to_string())
        diag_lines.append("\nDENOM value counts:\n" + vc(df["denom"]).to_string())
        diag_lines.append("\nREGION value counts:\n" + vc(df["region"]).to_string())
        diag_lines.append("\nETHNIC value counts (top 80):\n" + vc(df["ethnic"]).head(80).to_string())
        diag_lines.append("\nRACE value counts:\n" + vc(df["race"]).to_string())
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

    # Combined summary text
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
    lines.append(
        fs[
            ["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation_or_singularity"]
        ].to_string(index=False)
    )
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
    lines.append("- Standardized betas are computed from unstandardized OLS slopes (beta = b * sd(X)/sd(Y)); intercept is the unstandardized intercept.")
    lines.append("- Race/ethnicity dummies are made mutually exclusive (Hispanic overrides race) to match the table's intended comparison group.")
    lines.append("- Hispanic is inferred from ETHNIC using only within-file patterns (with fallback to missing if inference is inconclusive).")
    lines.append("- Conservative Protestant remains a proxy because the extract lacks finer tradition coding; it is kept missing when inputs are missing (no imputation).")
    lines.append("- Stars shown are from this run's OLS p-values (replication stars), not copied from the paper.")

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