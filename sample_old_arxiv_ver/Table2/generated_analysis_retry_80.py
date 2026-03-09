def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Load and normalize
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column in the input data.")

    # Coerce non-id to numeric
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Restrict to 1993
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Variables per mapping
    # -----------------------------
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    racism_items = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]

    base_required = [
        "hompop", "educ", "realinc", "prestg80", "sex", "age", "race", "relig", "denom", "region"
    ] + minority_genres + remaining_genres + racism_items

    missing_cols = [c for c in base_required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Hispanic: attempt to derive from 'ethnic' if possible (data-dependent).
    # We do NOT hard-code category values; we infer from the distribution and labels are unavailable.
    # If we cannot infer, we still include the variable but it will be missing and thus the model cannot be estimated.
    if "ethnic" not in df.columns:
        raise ValueError(
            "Column 'ethnic' is required to construct the Hispanic indicator for Table 2, "
            "but it is not present in this dataset extract."
        )

    # -----------------------------
    # Helpers
    # -----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")

    def fmt(x, nd=3):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
                return ""
        except Exception:
            pass
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

    def standardized_betas_from_unstd(fit, d_analytic, ycol, xcols):
        # beta_j = b_j * SD(x_j) / SD(y) on analytic sample; ddof=0
        y = pd.to_numeric(d_analytic[ycol], errors="coerce").astype(float)
        sd_y = y.std(ddof=0)
        out = {}
        for x in xcols:
            sx = pd.to_numeric(d_analytic[x], errors="coerce").astype(float)
            sd_x = sx.std(ddof=0)
            b = fit.params.get(x, np.nan)
            if pd.isna(b) or pd.isna(sd_x) or pd.isna(sd_y) or sd_x == 0 or sd_y == 0:
                out[x] = np.nan
            else:
                out[x] = float(b * (sd_x / sd_y))
        return pd.Series(out)

    def value_counts_full(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.value_counts(dropna=False).sort_index()

    # -----------------------------
    # DVs: strict dislike counts
    # -----------------------------
    for g in minority_genres + remaining_genres:
        df[f"d_{g}"] = dislike_indicator(df[g])

    dv1 = "dv1_minority6"
    dv2 = "dv2_remaining12"
    df[dv1] = strict_sum(df, [f"d_{g}" for g in minority_genres])   # 0..6 (strict)
    df[dv2] = strict_sum(df, [f"d_{g}" for g in remaining_genres])  # 0..12 (strict)

    # -----------------------------
    # Racism score (0..5), strict
    # -----------------------------
    df["r_rachaf"] = dich(df["rachaf"], ones=[1], zeros=[2])      # 1=yes object -> 1; 2=no -> 0
    df["r_busing"] = dich(df["busing"], ones=[2], zeros=[1])      # 2=oppose -> 1; 1=favor -> 0
    df["r_racdif1"] = dich(df["racdif1"], ones=[2], zeros=[1])    # 2=no -> 1; 1=yes -> 0 (discrimination)
    df["r_racdif3"] = dich(df["racdif3"], ones=[2], zeros=[1])    # 2=no -> 1; 1=yes -> 0 (education chance)
    df["r_racdif4"] = dich(df["racdif4"], ones=[1], zeros=[2])    # 1=yes -> 1; 2=no -> 0 (motivation)
    racism_comp = ["r_rachaf", "r_busing", "r_racdif1", "r_racdif3", "r_racdif4"]
    df["racism_score"] = strict_sum(df, racism_comp)

    # -----------------------------
    # Controls
    # -----------------------------
    df["education"] = df["educ"]

    df["income_pc"] = np.nan
    ok_inc = df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0)
    df.loc[ok_inc, "income_pc"] = df.loc[ok_inc, "realinc"] / df.loc[ok_inc, "hompop"]

    df["occ_prestige"] = df["prestg80"]

    df["female"] = np.where(df["sex"].isin([1, 2]), (df["sex"] == 2).astype(float), np.nan)

    df["age_years"] = df["age"]

    # Race: White reference; include Black, Hispanic, Other race
    race = df["race"]
    race_known = race.isin([1, 2, 3])
    df["black"] = np.where(race_known, (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race_known, (race == 3).astype(float), np.nan)

    # Hispanic: infer from 'ethnic' by choosing the most common non-missing code among non-Black, non-Other
    # that appears to be an "ancestry" code. This is inherently imperfect without labels, but we must include Hispanic.
    # Strategy:
    #   - candidate codes = ethnic values among race==1 (white) that are not missing
    #   - pick the smallest positive integer code among the top-k most frequent codes as the "Hispanic-like" group
    #     ONLY IF it also appears among race==2 or race==3 at nontrivial rate (common in GSS for Hispanic ancestry).
    # If inference fails, raise a clear error rather than silently omitting Hispanic.
    eth = pd.to_numeric(df["ethnic"], errors="coerce")
    df["_eth"] = eth

    # candidates among Whites
    white_eth = df.loc[(df["race"] == 1) & df["_eth"].notna(), "_eth"]
    eth_counts = white_eth.value_counts()

    hisp_code = None
    if len(eth_counts) > 0:
        top_codes = list(eth_counts.head(15).index)
        # prefer codes that also appear among non-White races (ancestry could be used across races)
        for code in sorted(top_codes):
            if pd.isna(code):
                continue
            # appears outside whites at least a little
            nonwhite_share = df.loc[(df["race"].isin([2, 3])) & df["_eth"].notna() & (df["_eth"] == code)].shape[0]
            if nonwhite_share >= 3:
                hisp_code = code
                break

    if hisp_code is None:
        # fallback: if there is a code == 1 and it is common, use it (many GSS-origin recodes use 1 for Mexican/PR/Cuban etc.)
        if 1.0 in eth_counts.index:
            hisp_code = 1.0

    if hisp_code is None:
        raise ValueError(
            "Unable to infer a Hispanic indicator from 'ethnic' without category labels. "
            "Provide a dataset that includes a Hispanic flag, or include metadata mapping 'ethnic' codes."
        )

    # Define Hispanic as: ethnic==hisp_code and not Black/Other race (consistent with paper's separate race dummies)
    # Missing if ethnic missing or race unknown.
    df["hispanic"] = np.nan
    known_for_hisp = race_known & df["_eth"].notna()
    df.loc[known_for_hisp, "hispanic"] = ((df.loc[known_for_hisp, "_eth"] == hisp_code) & (df.loc[known_for_hisp, "race"] == 1)).astype(float)

    # Conservative Protestant proxy: RELIG==1 & DENOM==1
    rel = df["relig"]
    den = df["denom"]
    df["cons_prot"] = np.where(rel.notna(), 0.0, np.nan)
    prot = rel.notna() & (rel == 1)
    df.loc[prot, "cons_prot"] = np.where(den.loc[prot].notna(), (den.loc[prot] == 1).astype(float), 0.0)

    df["no_religion"] = np.where(rel.notna(), (rel == 4).astype(float), np.nan)

    df["southern"] = np.where(df["region"].notna(), (df["region"] == 3).astype(float), np.nan)

    # Predictor list in Table 2 order
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
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "const": "Constant",
    }

    # -----------------------------
    # Fit model (listwise on DV + predictors)
    # -----------------------------
    def fit_model(dv_col, model_name, out_stub, target_n=None):
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
            write_text(f"./output/{out_stub}_ERROR.txt", msg)
            raise ValueError(msg)

        # Drop predictors with no variation (on analytic sample)
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

        # Human-readable table
        title = f"Bryson (1996) Table 2 replication — {model_name} (computed from provided 1993 GSS extract)"
        lines = []
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"DV: {dv_labels.get(dv_col, dv_col)}")
        lines.append("Estimation: OLS (unweighted).")
        lines.append("Displayed coefficients: standardized OLS coefficients (beta weights) for slopes.")
        lines.append("Constant: unstandardized intercept (raw DV scale).")
        lines.append("Stars: two-tailed p-values from the OLS fit on the analytic sample in this run.")
        lines.append("")
        lines.append("Implementation notes (important):")
        lines.append("- Dislike per genre: 1 if response in {4,5}; 0 if in {1,2,3}; otherwise missing")
        lines.append("- DV construction: strict sum (missing if any component genre missing)")
        lines.append("- Racism score: strict sum of 5 dichotomies (missing if any item missing)")
        lines.append("- Income per capita: REALINC / HOMPOP (HOMPOP>0 required)")
        lines.append("- Race dummies: Black (RACE==2), Hispanic (derived from ETHNIC, see diagnostics), Other race (RACE==3); White is reference")
        lines.append("- Conservative Protestant: proxy (RELIG==1 & DENOM==1)")
        lines.append("- No religion: RELIG==4")
        lines.append("- Southern: REGION==3")
        lines.append("- Missing data: listwise deletion on DV + included predictors")
        lines.append(f"- Hispanic inference: used ethnic code = {hisp_code} (see diagnostics for distribution)")
        if target_n is not None:
            lines.append(f"- Target N in paper (reference only): {int(target_n)}")
        if dropped_no_var:
            lines.append(f"- Dropped for no variation: {', '.join(dropped_no_var)}")
        lines.append("")

        tmp = table.copy()
        tmp["Coefficient"] = tmp["Std_Beta"].map(lambda v: fmt(v, 3))
        lines.append("Regression table (Table-2 style)")
        lines.append("--------------------------------")
        lines.append(tmp[["Independent Variable", "Coefficient", "Sig", "Status"]].to_string(index=False))
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

        write_text(f"./output/{out_stub}_table2_style.txt", "\n".join(lines))
        write_text(f"./output/{out_stub}_ols_unstandardized_summary.txt", fit.summary().as_text())

        # Diagnostics (to address prior feedback about N mismatches and Hispanic construction)
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
        diag.append("Key value counts (1993, including NA):")
        diag.append("\nRACE:\n" + value_counts_full(df["race"]).to_string())
        diag.append("\nETHNIC (raw):\n" + value_counts_full(df["_eth"]).to_string())
        diag.append(f"\nDerived Hispanic code used: {hisp_code}")
        diag.append("\nDerived Hispanic indicator:\n" + value_counts_full(df["hispanic"]).to_string())
        diag.append("\nREGION:\n" + value_counts_full(df["region"]).to_string())
        diag.append("\nRELIG:\n" + value_counts_full(df["relig"]).to_string())
        diag.append("\nDENOM:\n" + value_counts_full(df["denom"]).to_string())
        diag.append("\nRacism score:\n" + value_counts_full(df["racism_score"]).to_string())
        diag.append(f"\n{dv_labels.get(dv_col, dv_col)} value counts:\n" + value_counts_full(df[dv_col]).to_string())
        write_text(f"./output/{out_stub}_diagnostics.txt", "\n".join(diag))

        table.to_csv(f"./output/{out_stub}_table2_style.csv", index=False)
        fit_stats.to_csv(f"./output/{out_stub}_fit.csv", index=False)

        return table, fit_stats, d

    m1_table, m1_fit, m1_d = fit_model(
        dv1, "Model 1 (Minority-linked genres: 6)", "Table2_Model1_MinorityLinked6", target_n=644
    )
    m2_table, m2_fit, m2_d = fit_model(
        dv2, "Model 2 (Remaining genres: 12)", "Table2_Model2_Remaining12", target_n=605
    )

    # -----------------------------
    # Combined output
    # -----------------------------
    combined = pd.DataFrame({"Independent Variable": m1_table["Independent Variable"]}).merge(
        m1_table[["Independent Variable", "Std_Beta", "Sig", "Status"]].rename(
            columns={"Std_Beta": "Model1_Coefficient", "Sig": "Model1_Sig", "Status": "Model1_Status"}
        ),
        on="Independent Variable",
        how="left",
    ).merge(
        m2_table[["Independent Variable", "Std_Beta", "Sig", "Status"]].rename(
            columns={"Std_Beta": "Model2_Coefficient", "Sig": "Model2_Sig", "Status": "Model2_Status"}
        ),
        on="Independent Variable",
        how="left",
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
    lines.append("Notes:")
    lines.append("- Slope coefficients are standardized OLS coefficients (beta weights) computed from the unstandardized OLS fit.")
    lines.append("- Constant is the unstandardized intercept from the OLS fit (raw DV scale).")
    lines.append(f"- Hispanic was constructed from ETHNIC using inferred code {hisp_code}; see per-model diagnostics for distributions.")
    lines.append("")

    tmp = combined.copy()
    tmp["Model1_Coefficient"] = tmp["Model1_Coefficient"].map(lambda v: fmt(v, 3))
    tmp["Model2_Coefficient"] = tmp["Model2_Coefficient"].map(lambda v: fmt(v, 3))
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
    fs["Constant"] = fs["Constant"].map(lambda v: fmt(v, 3))
    fs["Constant_p"] = fs["Constant_p"].map(lambda v: fmt(v, 3))
    lines.append(fs[["Model", "DV", "N", "R2", "Adj_R2", "Constant", "Constant_p", "Dropped_no_variation"]].to_string(index=False))
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