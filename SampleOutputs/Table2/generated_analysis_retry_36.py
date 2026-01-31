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

    # Restrict to 1993 only
    df = df.loc[df["year"] == 1993].copy()

    # Coerce to numeric except id
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

    required = (
        ["hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region", "ethnic"]
        + minority_genres + remaining_genres + racism_items_raw
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing expected columns needed for this replication attempt: "
            + ", ".join(missing_cols)
        )

    # -----------------------------
    # Helpers
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")

    def dislike_indicator(series):
        """
        1 if response is 4/5 (dislike/dislike very much),
        0 if response is 1/2/3,
        otherwise missing.
        """
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
        return dfin[cols].sum(axis=1, skipna=False)

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

    def fmt(x, nd=3):
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

    def value_counts_all(s):
        return pd.Series(s).value_counts(dropna=False)

    def standardized_betas_from_fit(fit, d, ycol, xcols):
        """
        Conventional standardized OLS betas:
        beta_j = b_j * sd(X_j) / sd(Y), computed on analytic sample used for fit.
        """
        y = d[ycol].astype(float)
        y_sd = y.std(ddof=0)
        betas = {}
        for p in xcols:
            x = d[p].astype(float)
            x_sd = x.std(ddof=0)
            if pd.isna(y_sd) or y_sd == 0 or pd.isna(x_sd) or x_sd == 0:
                betas[p] = np.nan
            else:
                betas[p] = float(fit.params[p] * (x_sd / y_sd))
        return betas

    def hispanic_from_ethnic(series):
        """
        Try to construct a Hispanic indicator from ETHNIC without hard-coding unknown codes.
        Priority:
          1) If ETHNIC is already a flag {0,1} or {1,2} => map accordingly.
          2) If not, attempt to detect a small-category "hispanic" group by:
             - treating integer-like values only
             - selecting values that are relatively rare but not extremely rare,
               and that correspond to typical "origin codes" blocks (commonly 20-29 in some recodes).
        If detection fails, return all-missing.
        """
        eth = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=eth.index, dtype="float64")

        nonmiss = eth.dropna()
        if nonmiss.empty:
            return out

        u = set(np.unique(nonmiss.values))
        if u.issubset({0.0, 1.0}):
            out.loc[eth.notna()] = eth.loc[eth.notna()].astype(float)
            return out

        if u.issubset({1.0, 2.0}):
            out.loc[eth.notna()] = (eth.loc[eth.notna()] == 1.0).astype(float)
            return out

        # integer-like check
        flo = np.floor(nonmiss.values)
        int_like = np.isfinite(nonmiss.values) & (np.abs(nonmiss.values - flo) < 1e-9)
        if not int_like.any():
            return out

        observed_int = pd.Series(flo[int_like].astype(int)).value_counts()
        observed_vals = set(observed_int.index.tolist())

        # Heuristic: if codes 20-29 exist, use them as Hispanic-origin bucket
        band = set(range(20, 30))
        if len(observed_vals.intersection(band)) > 0:
            eth_int = pd.Series(np.floor(eth.values), index=eth.index)
            ok = eth.notna() & (np.abs(eth - eth_int) < 1e-9)
            out.loc[ok] = eth_int.loc[ok].astype(int).isin(list(band)).astype(float)
            return out

        # Otherwise: try to find a small set of "minority origin" codes that are not {97/98/99}-like
        # and appear with nontrivial frequency.
        # Keep integer codes with frequency between 0.5% and 15% (roughly plausible subgroup).
        n = int(nonmiss.shape[0])
        freq = observed_int / max(n, 1)
        candidates = set(freq[(freq >= 0.005) & (freq <= 0.15)].index.tolist())
        # Drop likely "misc/unknown" buckets: very high codes often represent "other" in some codings
        candidates = {c for c in candidates if c >= 0 and c <= 90}

        if not candidates:
            return out

        # Choose the smallest contiguous candidate block if any; else choose smallest few codes
        cand_sorted = sorted(candidates)
        blocks = []
        start = prev = cand_sorted[0]
        for v in cand_sorted[1:]:
            if v == prev + 1:
                prev = v
            else:
                blocks.append((start, prev))
                start = prev = v
        blocks.append((start, prev))
        # pick the block with smallest width (most "coded category group"), tie-breaker: lowest start
        blocks = sorted(blocks, key=lambda t: (t[1] - t[0], t[0]))
        chosen = set(range(blocks[0][0], blocks[0][1] + 1))

        eth_int_all = pd.Series(np.floor(eth.values), index=eth.index)
        ok_all = eth.notna() & (np.abs(eth - eth_int_all) < 1e-9)
        out.loc[ok_all] = eth_int_all.loc[ok_all].astype(int).isin(list(chosen)).astype(float)
        return out

    # -----------------------------
    # DVs: strict counts (missing if any component missing)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6_dislikes"
    dv2 = "dv2_remaining12_dislikes"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score: strict 5/5 items (missing if any component missing)
    # Direction per mapping instruction (higher = more racist)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no edu chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # 0..5

    # -----------------------------
    # Controls (preserve missing for listwise deletion)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    # Female
    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    # Age
    df["age_years"] = df["age"]

    # Race dummies (reference=White). Missing preserved if race not in {1,2,3}.
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: derive from ETHNIC using cautious mapping; missing preserved.
    df["hispanic"] = hispanic_from_ethnic(df["ethnic"])

    # Religion variables: preserve missing (do NOT fill with 0)
    # No religion: RELIG==4 (per provided extract documentation). Missing preserved.
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant: keep simple proxy only with available fields, missing preserved.
    # (If this proxy is not what the paper used, results will differ; but do not impute missing.)
    df["cons_prot"] = np.nan
    m = df["relig"].notna() & df["denom"].notna()
    df.loc[m, "cons_prot"] = ((df.loc[m, "relig"] == 1) & (df.loc[m, "denom"] == 1)).astype(float)

    # Southern: REGION==3 (per provided extract documentation). Missing preserved.
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
        dv1: "Dislike of minority-linked genres (count of 6)",
        dv2: "Dislike of remaining genres (count of 12)",
        "racism_score": "Racism score (0–5)",
        "education": "Education (years)",
        "income_pc": "Household income per capita (REALINC/HOMPOP)",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (SEX==2)",
        "age_years": "Age (years)",
        "black": "Black (RACE==2)",
        "hispanic": "Hispanic (derived from ETHNIC via cautious mapping)",
        "other_race": "Other race (RACE==3)",
        "cons_prot": "Conservative Protestant (proxy: RELIG==1 & DENOM==1; missing preserved)",
        "no_religion": "No religion (RELIG==4)",
        "southern": "Southern (REGION==3)",
        "const": "Constant",
    }

    # -----------------------------
    # Model fitting: strict listwise deletion on DV + all predictors
    # -----------------------------
    def fit_model(dv_col, model_name):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()
        d = d0.dropna(subset=model_cols).copy()

        # If listwise deletion is too strict, fail with diagnostics rather than silently proceed
        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after strict listwise deletion.\n\n"
                f"Missingness shares (1993) for candidate model columns:\n"
                f"{d0.isna().mean().sort_values(ascending=False).to_string()}\n\n"
                f"RELIG value counts:\n{value_counts_all(df['relig']).to_string()}\n\n"
                f"REGION value counts:\n{value_counts_all(df['region']).to_string()}\n\n"
                f"ETHNIC value counts:\n{value_counts_all(df['ethnic']).to_string()}\n"
            )
            write_text(f"./output/{model_name}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop no-variation predictors in analytic sample (prevents singular matrix)
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

        betas = standardized_betas_from_fit(fit, d, dv_col, kept)

        # Table in the Table 2 variable order (show blanks for dropped terms)
        rows = []
        for p in predictors:
            if p in kept:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": betas.get(p, np.nan),
                        "Sig": star(fit.pvalues.get(p, np.nan)),
                        "Note": "",
                    }
                )
            else:
                rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": np.nan,
                        "Sig": "",
                        "Note": "dropped (no variation in analytic sample)" if p in dropped_no_var else "",
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
                "Constant_Sig": [star(fit.pvalues.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        # Save human-readable output
        title = f"Bryson (1996) Table 2 replication attempt — {model_name}"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Beta computation: beta_j = b_j * sd(X_j)/sd(Y) on the analytic (listwise) sample.")
        lines.append("Significance stars: two-tailed p-values from the unstandardized OLS regression (this run).")
        lines.append("")
        lines.append("Construction rules used:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append("- Income_pc: REALINC/HOMPOP (HOMPOP must be >0)")
        lines.append("- Missing data: strict listwise deletion on DV + all predictors")
        if dropped_no_var:
            lines.append("")
            lines.append("Dropped due to no variation in analytic sample:")
            for p in dropped_no_var:
                lines.append(f"- {p}: {labels.get(p, p)}")

        lines.append("")
        lines.append("Standardized coefficients (Table 2 style)")
        lines.append("---------------------------------------")
        tmp = table.copy()
        tmp["Std_Beta"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append(tmp[["Independent Variable", "Std_Beta", "Sig", "Note"]].to_string(index=False))

        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())

        write_text(f"./output/{model_name}_table2_style.txt", "\n".join(lines))
        with open(f"./output/{model_name}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n")

        # Diagnostics to track N and coding
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
        diag_lines.append("Value counts in analytic sample (key dummies):")
        for v in ["female", "black", "hispanic", "other_race", "cons_prot", "no_religion", "southern"]:
            diag_lines.append(f"\n{v} ({labels.get(v, v)}):")
            diag_lines.append(value_counts_all(d[v]).to_string())
        write_text(f"./output/{model_name}_diagnostics.txt", "\n".join(diag_lines))

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        return table, fit_stats, d

    # Fit both models
    m1_table, m1_fit, m1_d = fit_model(dv1, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit, m2_d = fit_model(dv2, "Table2_ModelB_Remaining12")

    # Combined output
    combined = pd.DataFrame(
        {
            "Independent Variable": m1_table["Independent Variable"],
            "ModelA_Std_Beta": m1_table["Std_Beta"],
            "ModelA_Sig": m1_table["Sig"],
            "ModelB_Std_Beta": m2_table["Std_Beta"],
            "ModelB_Sig": m2_table["Sig"],
        }
    )
    combined_fit = pd.concat([m1_fit, m2_fit], axis=0)

    # Quickchecks (pre-listwise) to spot coding problems immediately
    qc = []
    qc.append("Quickcheck distributions (1993 sample, pre-listwise)")
    qc.append("====================================================")
    qc.append("")
    qc.append("RELIG value counts:")
    qc.append(value_counts_all(df["relig"]).to_string())
    qc.append("")
    qc.append("DENOM value counts:")
    qc.append(value_counts_all(df["denom"]).to_string())
    qc.append("")
    qc.append("REGION value counts:")
    qc.append(value_counts_all(df["region"]).to_string())
    qc.append("")
    qc.append("ETHNIC value counts:")
    qc.append(value_counts_all(df["ethnic"]).to_string())
    qc.append("")
    qc.append("Derived dummies value counts (pre-listwise):")
    for v in ["female", "black", "hispanic", "other_race", "cons_prot", "no_religion", "southern"]:
        qc.append(f"\n{v} ({labels.get(v, v)}):")
        qc.append(value_counts_all(df[v]).to_string())

    qc.append("")
    qc.append("DV1 distribution (minority-linked 6):")
    qc.append(value_counts_all(df[dv1]).sort_index().to_string())
    qc.append("")
    qc.append("DV2 distribution (remaining 12):")
    qc.append(value_counts_all(df[dv2]).sort_index().to_string())
    qc.append("")
    qc.append("Racism score distribution:")
    qc.append(value_counts_all(df["racism_score"]).sort_index().to_string())

    write_text("./output/quickcheck_distributions.txt", "\n".join(qc))

    # Human-readable combined summary
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (GSS 1993 extract provided)"
    lines.append(title)
    lines.append("=" * len(title))
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
    fs = combined_fit[["DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs.to_string())
    lines.append("")
    lines.append("Notes:")
    lines.append("- Missing values are preserved; models use strict listwise deletion on DV + all predictors.")
    lines.append("- Hispanic is derived from ETHNIC using a cautious, data-driven mapping; if ETHNIC is not a Hispanic-origin indicator in this extract, results will differ.")
    lines.append("- Conservative Protestant uses a limited proxy (RELIG==1 & DENOM==1) because more detailed denomination tradition coding is not available in this extract.")
    write_text("./output/combined_summary.txt", "\n".join(lines))

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
        "modelA_analytic_sample": m1_d,
        "modelB_analytic_sample": m2_d,
    }