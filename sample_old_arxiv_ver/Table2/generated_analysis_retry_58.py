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

    df = df.loc[df["year"] == 1993].copy()

    # Coerce all non-id fields to numeric
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
    racism_items_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    # Conservative Protestant / Hispanic availability is extract-dependent
    required_min = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "region"]
        + minority_genres
        + remaining_genres
        + racism_items_raw
    )
    missing_cols = [c for c in required_min if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in extract: {missing_cols}")

    # Optional columns
    has_hisp = "hispanic" in df.columns
    has_ethnic = "ethnic" in df.columns
    has_denom = "denom" in df.columns

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

    def dislike_indicator(x):
        """
        1 if response is 4/5 (dislike/dislike very much),
        0 if response is 1/2/3,
        missing otherwise.
        """
        x = pd.to_numeric(x, errors="coerce")
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

    def value_counts_full(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.value_counts(dropna=False).sort_index()

    def standardized_betas_from_unstd(fit, d, ycol, xcols):
        """
        Standardized betas on analytic sample:
          beta_j = b_j * SD(x_j) / SD(y)
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

    def build_hispanic_indicator(dfin):
        """
        Prefer an explicit 'hispanic' column if present.
        Otherwise, attempt to derive from 'ethnic' ONLY in a conservative way:
        - If ETHNIC is binary-like (0/1 or 1/2), map to 0/1.
        - Else, if ETHNIC contains common GSS-style 4-category codes (1..4), treat codes 2/3 as Hispanic.
        - If still unclear, return all-missing (do not fabricate).
        """
        if "hispanic" in dfin.columns:
            h = pd.to_numeric(dfin["hispanic"], errors="coerce")
            # Accept common codings: 1/2, 0/1
            vals = sorted(h.dropna().unique().tolist())
            out = pd.Series(np.nan, index=h.index, dtype="float64")
            if set(vals).issubset({0, 1}):
                out.loc[h.notna()] = h.loc[h.notna()].astype(float)
                return out
            if set(vals).issubset({1, 2}):
                out.loc[h.notna()] = (h.loc[h.notna()] == 1).astype(float)
                return out
            # If already 0/1-ish but with stray codes, only map known and leave others missing
            out.loc[h.isin([0, 1])] = h.loc[h.isin([0, 1])].astype(float)
            out.loc[h.isin([1, 2])] = (h.loc[h.isin([1, 2])] == 1).astype(float)
            return out

        if "ethnic" not in dfin.columns:
            return pd.Series(np.nan, index=dfin.index, dtype="float64")

        e = pd.to_numeric(dfin["ethnic"], errors="coerce")
        out = pd.Series(np.nan, index=e.index, dtype="float64")
        vals = sorted(e.dropna().unique().tolist())

        # Binary-like: 0/1
        if set(vals).issubset({0, 1}):
            out.loc[e.notna()] = e.loc[e.notna()].astype(float)
            return out

        # Binary-like: 1/2 (often 1=yes,2=no or vice versa; cannot assume)
        # Only accept if distribution is clearly "minority flag" with small 1's
        if set(vals).issubset({1, 2}):
            # Try 1==Hispanic first (common for yes/no items)
            cand1 = pd.Series(np.nan, index=e.index, dtype="float64")
            cand1.loc[e.notna()] = (e.loc[e.notna()] == 1).astype(float)
            if cand1.dropna().nunique() == 2 and 0 < (cand1 == 1).sum() < (cand1 == 0).sum():
                return cand1
            # Try 2==Hispanic (less likely, but possible)
            cand2 = pd.Series(np.nan, index=e.index, dtype="float64")
            cand2.loc[e.notna()] = (e.loc[e.notna()] == 2).astype(float)
            if cand2.dropna().nunique() == 2 and 0 < (cand2 == 1).sum() < (cand2 == 0).sum():
                return cand2
            return out

        # 4-category style (often something like 1=not hisp, 2=mex, 3=pr, 4=cuban/other)
        if set(vals).issubset({1, 2, 3, 4}):
            out.loc[e.notna()] = e.loc[e.notna()].isin([2, 3, 4]).astype(float)
            if out.dropna().nunique() == 2:
                return out

        # Unknown scheme: do not guess
        return out

    def build_cons_prot(dfin):
        """
        Conservative Protestant is not fully reconstructable with this extract.
        We implement the best *non-imputing* proxy possible:
        - If denom exists: CONS_PROT = 1 if RELIG==1 & DENOM==1 (Baptist), else 0, missing if RELIG or DENOM missing.
        - If denom is absent: return all-missing (do not fabricate).
        """
        if "denom" not in dfin.columns:
            return pd.Series(np.nan, index=dfin.index, dtype="float64")
        rel = pd.to_numeric(dfin["relig"], errors="coerce")
        den = pd.to_numeric(dfin["denom"], errors="coerce")
        out = pd.Series(np.nan, index=dfin.index, dtype="float64")
        m = rel.notna() & den.notna()
        out.loc[m] = ((rel.loc[m] == 1) & (den.loc[m] == 1)).astype(float)
        return out

    # -----------------------------
    # Dependent variables: strict complete-case dislike counts
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6 strict
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12 strict

    # -----------------------------
    # Racism score: strict 5/5 items, sum to 0..5
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls: preserve missingness (no imputing missing to 0)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    race = df["race"]
    df["black"] = np.where(race.isin([1, 2, 3]), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.isin([1, 2, 3]), (race == 3).astype(float), np.nan)

    # Hispanic (prefer explicit field; else cautious derivation from ETHNIC; else missing)
    df["hispanic_ind"] = build_hispanic_indicator(df)

    # Religion dummies (keep missing as missing)
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant (proxy if denom exists; keep missing as missing)
    df["cons_prot"] = build_cons_prot(df)

    # Southern per mapping instruction (keep missing as missing)
    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    predictors = [
        "racism_score",
        "education",
        "income_pc",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic_ind",
        "other_race",
        "cons_prot",
        "no_religion",
        "southern",
    ]

    labels = {
        dv1: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music (count of 6)",
        dv2: "Dislike of the 12 Remaining Genres (count of 12)",
        "racism_score": "Racism score (0–5; strict 5 items, sum)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (SEX==2)",
        "age_years": "Age (years)",
        "black": "Black (RACE==2)",
        "hispanic_ind": "Hispanic indicator (prefer 'hispanic', else cautious from 'ethnic')",
        "other_race": "Other race (RACE==3)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1; missing if RELIG/DENOM missing)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting (strict listwise deletion, no silent dropping)
    # -----------------------------
    def fit_model(dv_col, model_name, stub):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Predictors that are all-missing in extract: mark unavailable and remove from model frame,
        # BUT keep track and report explicitly (do not replace with zeros).
        unavailable = [p for p in predictors if d0[p].isna().all()]
        usable_predictors = [p for p in predictors if p not in unavailable]

        # Strict listwise deletion on DV + usable predictors
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

        # Drop non-varying predictors (avoid singular matrices)
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

        # Standardized betas from unstandardized fit
        betas = standardized_betas_from_unstd(fit_unstd, d, dv_col, kept)

        # Build table rows in original order
        rows = []
        for p in predictors:
            if p in unavailable:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": "dropped (unavailable)"})
            elif p in dropped_no_var:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": "dropped (no variation)"})
            elif p in kept:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star_from_p(fit_unstd.pvalues.get(p, np.nan)),
                        "Status": "included",
                    }
                )
            else:
                rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": "", "Status": "dropped (other)"})

        table = pd.DataFrame(rows)

        fit_stats = pd.DataFrame(
            {
                "Model": [model_name],
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [star_from_p(fit_unstd.pvalues.get("const", np.nan))],
                "Dropped_unavailable": [", ".join(unavailable) if unavailable else ""],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            }
        )

        # Human-readable report
        title = f"Bryson (1996) Table 2 — {model_name} (computed from provided GSS 1993 extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Standardization: beta_j = b_j * SD(x_j) / SD(y) computed on analytic sample.")
        lines.append("Stars: two-tailed p-values from unstandardized OLS in this run (replication stars).")
        lines.append("")
        lines.append("Construction rules:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict 5/5 dichotomous items; sum -> 0..5 (missing if any item missing)")
        lines.append("- Missing data in regressions: strict listwise deletion on DV + included predictors")
        lines.append("")
        lines.append("Standardized coefficients")
        lines.append("------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp[["Independent Variable", "Std_Beta", "Sig", "Status"]].to_string(index=False))
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
                ["Model", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_unavailable", "Dropped_no_variation"]
            ].to_string(index=False)
        )
        write_text(f"./output/{stub}_table2_style.txt", "\n".join(lines))

        # Save full OLS summary
        with open(f"./output/{stub}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit_unstd.summary().as_text())
            f.write("\n")

        # Diagnostics to debug N mismatches
        diag_lines = []
        diag_lines.append(f"{model_name} diagnostics")
        diag_lines.append("=" * (len(model_name) + 12))
        diag_lines.append(f"N_1993_total: {int(df.shape[0])}")
        diag_lines.append(f"N_with_nonmissing_DV: {int(df[dv_col].notna().sum())}")
        diag_lines.append(f"N_analytic_listwise: {int(d.shape[0])}")
        diag_lines.append("")
        diag_lines.append("Missingness shares in 1993 for model columns (descending):")
        diag_lines.append(d0.isna().mean().sort_values(ascending=False).map(lambda v: fmt(v, 3)).to_string())

        diag_lines.append("\n\nUnderlying raw value counts (1993, pre-listwise; including NA):")
        diag_lines.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        if "denom" in df.columns:
            diag_lines.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag_lines.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag_lines.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        if "hispanic" in df.columns:
            diag_lines.append("\nHISPANIC (raw):\n" + value_counts_full(df["hispanic"]).to_string())
        if "ethnic" in df.columns:
            diag_lines.append("\nETHNIC (raw):\n" + value_counts_full(df["ethnic"]).to_string())
        diag_lines.append("\nHispanic indicator used (hispanic_ind):\n" + value_counts_full(df["hispanic_ind"]).to_string())
        diag_lines.append("\nRacism components missingness:\n" + df[racism_comp + ["racism_score"]].isna().mean().map(lambda v: fmt(v, 3)).to_string())
        diag_lines.append("\nRacism score value counts:\n" + value_counts_full(df["racism_score"]).to_string())
        diag_lines.append("\nDV value counts:\n" + value_counts_full(df[dv_col]).to_string())

        write_text(f"./output/{stub}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{stub}_fit.csv", index=False)

        return table, fit_stats, d, unavailable, dropped_no_var, kept

    m1_table, m1_fit, m1_d, m1_unavail, m1_novar, m1_kept = fit_model(
        dv1, "Model A (Minority-linked genres: 6)", "Table2_ModelA_MinorityLinked6"
    )
    m2_table, m2_fit, m2_d, m2_unavail, m2_novar, m2_kept = fit_model(
        dv2, "Model B (Remaining genres: 12)", "Table2_ModelB_Remaining12"
    )

    # -----------------------------
    # Combined outputs
    # -----------------------------
    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "ModelA_Std_Beta": m1_table["Std_Beta"],
            "ModelA_Sig": m1_table["Sig"],
            "ModelA_Status": m1_table["Status"],
            "ModelB_Std_Beta": m2_table["Std_Beta"],
            "ModelB_Sig": m2_table["Sig"],
            "ModelB_Status": m2_table["Status"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0, ignore_index=True)

    def dv_desc(series):
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

    dv_desc_df = pd.DataFrame(
        [
            {"Sample": "All 1993", "DV": labels[dv1], **dv_desc(df[dv1])},
            {"Sample": "All 1993", "DV": labels[dv2], **dv_desc(df[dv2])},
            {"Sample": "Model A analytic sample", "DV": labels[dv1], **dv_desc(m1_d[dv1])},
            {"Sample": "Model B analytic sample", "DV": labels[dv2], **dv_desc(m2_d[dv2])},
        ]
    )

    # Human-readable combined summary
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (computed from provided GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Key implementation choices (to match published specification as closely as extract allows)")
    lines.append("-------------------------------------------------------------------------------------")
    lines.append("- Year restriction: YEAR==1993")
    lines.append("- DVs: strict complete-case counts of dislikes (4/5 => dislike) across specified genres")
    lines.append("- Racism: strict 5/5 items required; summed to integer 0–5")
    lines.append("- Missing data in regressions: strict listwise deletion (DV + all included predictors)")
    lines.append("- Standardized coefficients: beta_j = b_j * SD(x_j) / SD(y) on analytic sample")
    lines.append("- Stars: from this run's OLS p-values (paper does not report SEs)")
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars (from this run)")
    lines.append("--------------------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
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
            ["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_unavailable", "Dropped_no_variation"]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("DV descriptives (counts)")
    lines.append("------------------------")
    dvf = dv_desc_df.copy()
    dvf["N"] = dvf["N"].map(lambda v: fmt(v, 0))
    for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
        dvf[c] = dvf[c].map(lambda v: fmt(v, 3))
    lines.append(dvf[["Sample", "DV", "N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"]].to_string(index=False))

    lines.append("")
    lines.append("Quick sample-size checkpoints (target from paper: N=644 for Model A, N=605 for Model B)")
    lines.append("--------------------------------------------------------------------------------------")
    lines.append(f"Model A analytic N: {int(m1_d.shape[0])}")
    lines.append(f"Model B analytic N: {int(m2_d.shape[0])}")
    lines.append("")
    lines.append("Predictors kept after dropping unavailable/no-variation")
    lines.append("------------------------------------------------------")
    lines.append(f"Model A kept: {', '.join(m1_kept) if m1_kept else '(none)'}")
    lines.append(f"Model B kept: {', '.join(m2_kept) if m2_kept else '(none)'}")

    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=False)
    dv_desc_df.to_csv("./output/dv_descriptives.csv", index=False)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "dv_descriptives": dv_desc_df,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_analytic_sample": m1_d,
        "modelB_analytic_sample": m2_d,
    }