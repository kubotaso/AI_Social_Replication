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

    # Coerce to numeric where possible (keep id as-is)
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Variable lists (per mapping)
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items_raw = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    core_required = [
        "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"
    ]
    optional = ["ethnic"]  # best-effort Hispanic (extract lacks a dedicated hispanic flag)

    required = core_required + minority_genres + remaining_genres + racism_items_raw
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    for c in optional:
        if c not in df.columns:
            df[c] = np.nan

    # -----------------------------
    # Helpers
    # -----------------------------
    def write_text(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

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
        """Map to {0,1}; anything else missing."""
        x = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(list(zeros))] = 0.0
        out.loc[x.isin(list(ones))] = 1.0
        return out

    def strict_sum(dfin, cols):
        """Row sum; missing if ANY component missing."""
        return dfin[cols].sum(axis=1, skipna=False)

    def z(s):
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean()
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

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

    def vc(s):
        return pd.Series(s).value_counts(dropna=False)

    # -----------------------------
    # DV construction (strict counts, missing if any component missing)
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1_col = "dv1_minority_linked_6"
    dv2_col = "dv2_remaining_12"
    df[dv1_col] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6
    df[dv2_col] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12

    # -----------------------------
    # Racism score (0–5) STRICT 5/5 items (missing if any missing)
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no discrimination -> 1; 1=yes -> 0
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no edu chance -> 1; 1=yes -> 0
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes willpower -> 1; 2=no -> 0
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)  # 0..5, strict

    # -----------------------------
    # Controls (preserve missing; listwise in model)
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)
    df["age_years"] = df["age"]

    # Race dummies: reference is White; preserve missing if race unknown
    df["black"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 2).astype(float), np.nan)
    df["other_race"] = np.where(df["race"].isin([1, 2, 3]), (df["race"] == 3).astype(float), np.nan)

    # Hispanic: best-effort from ETHNIC; preserve missing if ETHNIC missing
    eth = df["ethnic"]

    def make_hisp_from_codes(codes):
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        m = eth.notna()
        out.loc[m] = eth.loc[m].isin(list(codes)).astype(float)
        return out

    # Candidate code-sets (unknown coding in this extract).
    hisp_defs = {
        "eth_in_1_2_3_4": set([1, 2, 3, 4]),
        "eth_in_20_21_22": set([20, 21, 22]),
        "eth_in_29_30_31": set([29, 30, 31]),
    }
    hisp_candidates = {name: make_hisp_from_codes(codes) for name, codes in hisp_defs.items()}

    def candidate_score(s):
        # Prefer: nontrivial variation, moderate prevalence, and many observed values.
        m = s.notna()
        if m.sum() == 0:
            return -np.inf
        p = s.loc[m].mean()
        if p <= 0 or p >= 0.5:
            return -1e12 + float(m.sum())
        return float(m.sum()) - 1000.0 * abs(p - 0.10)

    best_name = max(hisp_candidates.keys(), key=lambda k: candidate_score(hisp_candidates[k]))
    df["hispanic"] = hisp_candidates[best_name]

    # Conservative Protestant proxy per mapping instruction: RELIG==1 & DENOM==1; preserve missing
    df["cons_prot"] = np.nan
    m = df["relig"].notna() & df["denom"].notna()
    df.loc[m, "cons_prot"] = ((df.loc[m, "relig"] == 1) & (df.loc[m, "denom"] == 1)).astype(float)

    # No religion: RELIG==4; preserve missing
    df["no_religion"] = np.where(df["relig"].notna(), (df["relig"] == 4).astype(float), np.nan)

    # Southern: REGION==3; preserve missing
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
        dv1_col: "Dislike of Rap, Reggae, Blues/R&B, Jazz, Gospel, and Latin Music",
        dv2_col: "Dislike of the 12 Remaining Genres",
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
    # Fit model: standardized betas via z(Y) on z(X); stars from unstandardized OLS p-values
    # (This matches typical "beta weights + conventional OLS significance tests".)
    # -----------------------------
    def fit_table2(dv_col, model_name):
        model_cols = [dv_col] + predictors
        d0 = df[model_cols].copy()

        # Strict listwise deletion on ALL model columns (faithful; no missing->0 recodes)
        d = d0.dropna(subset=model_cols).copy()

        if d.shape[0] == 0:
            msg = (
                f"{model_name}: analytic sample is empty after listwise deletion.\n"
                f"Missingness shares (1993) for model columns:\n{d0.isna().mean().sort_values(ascending=False).to_string()}\n"
            )
            write_text(f"./output/{model_name}_ERROR.txt", [msg])
            raise ValueError(msg)

        # Drop no-variation predictors (avoid singular matrix); keep a record
        kept, dropped_no_var = [], []
        for p in predictors:
            if d[p].nunique(dropna=True) <= 1:
                dropped_no_var.append(p)
            else:
                kept.append(p)

        y = d[dv_col].astype(float)
        X = d[kept].astype(float)

        # Unstandardized OLS for intercept + p-values
        Xc = sm.add_constant(X, has_constant="add")
        fit_unstd = sm.OLS(y, Xc).fit()

        # Standardized regression for betas: z(Y) on z(X), no intercept
        yz = z(y)
        Xz = pd.DataFrame({c: z(X[c]) for c in kept})
        dz = pd.concat([yz.rename("y"), Xz], axis=1).dropna()

        yz2 = dz["y"].astype(float)
        Xz2 = dz[kept].astype(float)
        fit_std = sm.OLS(yz2, Xz2).fit()
        betas = fit_std.params.reindex(kept)

        # Stars from unstandardized OLS p-values (conventional)
        pvals_unstd = fit_unstd.pvalues.reindex(["const"] + kept)

        table_rows = []
        for p in predictors:
            if p in kept:
                table_rows.append(
                    {
                        "Independent Variable": labels.get(p, p),
                        "Std_Beta": float(betas.get(p, np.nan)),
                        "Sig": star(pvals_unstd.get(p, np.nan)),
                    }
                )
            else:
                table_rows.append({"Independent Variable": labels.get(p, p), "Std_Beta": np.nan, "Sig": ""})
        table = pd.DataFrame(table_rows)

        fit_stats = pd.DataFrame(
            {
                "DV": [labels.get(dv_col, dv_col)],
                "N": [int(round(fit_unstd.nobs))],
                "R2": [float(fit_unstd.rsquared)],
                "Adj_R2": [float(fit_unstd.rsquared_adj)],
                "Constant": [float(fit_unstd.params.get("const", np.nan))],
                "Constant_Sig": [star(pvals_unstd.get("const", np.nan))],
                "Dropped_no_variation": [", ".join(dropped_no_var) if dropped_no_var else ""],
            },
            index=[model_name],
        )

        # Save human-readable table
        lines = []
        title = f"Bryson (1996) Table 2 replication attempt — {model_name}"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights).")
        lines.append("Betas computed as: regress z(Y) on z(X) with no intercept.")
        lines.append("Significance stars: two-tailed p-values from the unstandardized OLS regression.")
        lines.append("")
        lines.append("Construction rules used:")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; else missing")
        lines.append("- DV: strict sum across component genres (missing if any component missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any component missing)")
        lines.append(f"- Hispanic: constructed from ETHNIC (rule selected='{best_name}'); see quickcheck_distributions.txt")
        lines.append("- Conservative Protestant: (RELIG==1 & DENOM==1), missing preserved")
        lines.append("- No religion: (RELIG==4), missing preserved")
        lines.append("- Southern: (REGION==3), missing preserved")
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
        lines.append(tmp.to_string(index=False))
        lines.append("")
        lines.append("Fit statistics (unstandardized OLS)")
        lines.append("---------------------------------")
        fs = fit_stats[["N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].copy()
        fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
        fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
        fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
        fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
        lines.append(fs.to_string())
        write_text(f"./output/{model_name}_table2_style.txt", lines)

        # Save regression diagnostics
        with open(f"./output/{model_name}_ols_unstandardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit_unstd.summary().as_text())
            f.write("\n")
        with open(f"./output/{model_name}_ols_standardized_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit_std.summary().as_text())
            f.write("\n")

        table.to_csv(f"./output/{model_name}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{model_name}_fit.csv", index=True)

        # Diagnostics (to pinpoint N collapse and dropped dummies)
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
            diag_lines.append(f"\n{v} ({labels.get(v,v)}):")
            diag_lines.append(vc(d[v]).to_string())
        write_text(f"./output/{model_name}_diagnostics.txt", diag_lines)

        return table, fit_stats

    m1_table, m1_fit = fit_table2(dv1_col, "Table2_ModelA_MinorityLinked6")
    m2_table, m2_fit = fit_table2(dv2_col, "Table2_ModelB_Remaining12")

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

    # -----------------------------
    # Quickcheck distributions (debug Hispanic & no_religion variation and missingness)
    # -----------------------------
    qc_lines = []
    qc_lines.append("Quickcheck distributions (1993 sample, pre-listwise)")
    qc_lines.append("====================================================")
    qc_lines.append("")
    qc_lines.append("ETHNIC value counts:")
    qc_lines.append(vc(df["ethnic"]).to_string())
    qc_lines.append("")
    qc_lines.append(f"Hispanic rule selected: {best_name}")
    qc_lines.append("Hispanic value counts:")
    qc_lines.append(vc(df["hispanic"]).to_string())
    qc_lines.append("")
    qc_lines.append("RELIG value counts:")
    qc_lines.append(vc(df["relig"]).to_string())
    qc_lines.append("")
    qc_lines.append("No religion value counts:")
    qc_lines.append(vc(df["no_religion"]).to_string())
    qc_lines.append("")
    qc_lines.append("REGION value counts:")
    qc_lines.append(vc(df["region"]).to_string())
    qc_lines.append("")
    qc_lines.append("Southern value counts:")
    qc_lines.append(vc(df["southern"]).to_string())
    qc_lines.append("")
    qc_lines.append("DENOM value counts:")
    qc_lines.append(vc(df["denom"]).to_string())
    qc_lines.append("")
    qc_lines.append("Conservative Protestant value counts:")
    qc_lines.append(vc(df["cons_prot"]).to_string())
    qc_lines.append("")
    qc_lines.append("DV1 (minority-linked 6) count distribution:")
    qc_lines.append(vc(df[dv1_col]).sort_index().to_string())
    qc_lines.append("")
    qc_lines.append("DV2 (remaining 12) count distribution:")
    qc_lines.append(vc(df[dv2_col]).sort_index().to_string())
    write_text("./output/quickcheck_distributions.txt", qc_lines)

    # -----------------------------
    # Combined summary
    # -----------------------------
    lines = []
    title = "Bryson (1996) Table 2 replication attempt (GSS 1993 extract)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Combined standardized coefficients (beta weights) and significance stars (from this replication)")
    lines.append("-------------------------------------------------------------------------------------------")
    tmp = combined.copy()
    tmp["ModelA_Std_Beta"] = tmp["ModelA_Std_Beta"].map(lambda v: fmt(v, 3))
    tmp["ModelB_Std_Beta"] = tmp["ModelB_Std_Beta"].map(lambda v: fmt(v, 3))
    lines.append(tmp.to_string(index=False))
    lines.append("")
    lines.append("Fit statistics (unstandardized OLS; this replication)")
    lines.append("-----------------------------------------------------")
    fs = combined_fit[["DV", "N", "R2", "Adj_R2", "Constant", "Constant_Sig", "Dropped_no_variation"]].copy()
    fs["N"] = fs["N"].map(lambda v: fmt(v, 0))
    fs["R2"] = fs["R2"].map(lambda v: fmt(v, 3))
    fs["Adj_R2"] = fs["Adj_R2"].map(lambda v: fmt(v, 3))
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    lines.append(fs.to_string())
    lines.append("")
    lines.append("Notes:")
    lines.append("- This code preserves missing values (no missing->0 imputation for dummies). Models use strict listwise deletion.")
    lines.append("- If N is far below the published table, it reflects missingness in this extract (often ETHNIC, RELIG/DENOM, racism items, income, or music items).")
    lines.append("- Hispanic is best-effort from ETHNIC because a dedicated Hispanic indicator is not present in the provided variable list; see ./output/quickcheck_distributions.txt.")
    write_text("./output/combined_summary.txt", lines)

    combined.to_csv("./output/combined_table2_betas.csv", index=False)
    combined_fit.to_csv("./output/combined_fit.csv", index=True)

    return {
        "combined_table2_betas": combined,
        "combined_fit": combined_fit,
        "modelA_table": m1_table,
        "modelB_table": m2_table,
    }