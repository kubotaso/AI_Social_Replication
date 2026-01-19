def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing-code handling for this extract:
        - Coerce to numeric
        - Treat common DK/NA/refusal sentinels as missing
        - For Likert 1-5, we will also enforce range later.
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(sentinels)))
        return x

    def likert_dislike_indicator(x):
        """
        Music items are 1-5: 4/5 = dislike, 1/2/3 = not dislike.
        Anything outside 1..5 (or missing codes) -> NaN.
        """
        x = clean_gss_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def bin_from_codes(x, true_codes, false_codes):
        x = clean_gss_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all_items=False):
        """
        Sum of item-level dislike indicators.
        Paper notes DK treated as missing; then cases with missing excluded.
        We implement the least-restrictive faithful choice to avoid collapsing N:
        - Require at least 1 non-missing item (min_count=1), not necessarily all 6/12,
          because "DK treated as missing in construction" does not necessarily imply
          all items must be answered to compute a count.
        If you want strict complete-item counts, set require_all_items=True.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        min_count = len(items) if require_all_items else 1
        return mat.sum(axis=1, min_count=min_count)

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def ols_with_posthoc_betas(df_model, dv, xcols, model_name):
        """
        Fit OLS on unstandardized variables (with intercept), then compute standardized
        betas post-hoc for slopes: beta_j = b_j * sd(x_j)/sd(y).
        This keeps intercept comparable (unstandardized), aligning with common "standardized
        coefficients table" conventions that still reports a constant.
        """
        needed = [dv] + xcols
        d = df_model[needed].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)
        X = d[xcols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # Post-hoc standardized betas for slopes
        y_sd = y.std(ddof=0)
        betas = {}
        for c in xcols:
            x_sd = X[c].std(ddof=0)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = model.params[c] * (x_sd / y_sd)

        beta_series = pd.Series(betas)
        pvals = model.pvalues.reindex(["const"] + xcols)

        # "paper style" table: standardized betas for slopes, intercept unstandardized
        rows = []
        rows.append(("Constant", model.params.get("const", np.nan), pvals.get("const", np.nan), ""))  # no stars convention varies; we compute but leave blank below
        for c in xcols:
            rows.append((c, beta_series.get(c, np.nan), pvals.get(c, np.nan), stars(pvals.get(c, np.nan))))

        paper_tbl = pd.DataFrame(rows, columns=["term", "coef", "p_value", "stars"]).set_index("term")

        # Full table for diagnostics (computed from our estimation; not from paper)
        full_tbl = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        # add standardized betas aligned to full table rows
        full_tbl["beta_std_posthoc"] = np.nan
        for c in xcols:
            full_tbl.loc[c, "beta_std_posthoc"] = beta_series.get(c, np.nan)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        return model, paper_tbl, full_tbl, fit, d.index

    def write_text_tables(model, paper_tbl, full_tbl, fit_df, model_name):
        # statsmodels summary
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFIT:\n")
            f.write(fit_df.to_string(index=False))
            f.write("\n")

        # Paper-style table (standardized slopes, unstandardized constant)
        # Do not star the constant in this "paper match" output (paper conventions vary).
        paper_out = paper_tbl.copy()
        if "Constant" in paper_out.index:
            paper_out.loc["Constant", "stars"] = ""
        paper_out["coef_with_stars"] = paper_out["coef"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "") + paper_out["stars"]
        paper_out_disp = paper_out[["coef_with_stars"]]

        with open(f"./output/{model_name}_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Paper-style output: standardized coefficients (slopes) + stars; Constant is unstandardized.\n")
            f.write("Stars: * p<.05, ** p<.01, *** p<.001 (from this re-estimation; paper may differ if sample/weights differ).\n\n")
            f.write(paper_out_disp.to_string())
            f.write("\n")

        # Full diagnostics table
        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write("Full regression output (computed from this re-estimation; SEs not reported in Bryson Table 2).\n\n")
            f.write(full_tbl.to_string(float_format=lambda v: f"{v: .6f}"))
            f.write("\n")

    # -----------------------------
    # Load and filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain columns 'year' and 'id'.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct DVs
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Least-restrictive faithful construction to avoid artificial N collapse:
    # count of dislikes among available (non-missing) items; listwise deletion happens at model stage.
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_items=False)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_items=False)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = bin_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = bin_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = bin_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = bin_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = bin_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)

    # Require all five components to avoid changing scale meaning
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # RHS controls
    # -----------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing columns for income per capita: realinc and/or hompop")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = bin_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race indicators (White reference)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables; do NOT proxy using ethnic.
    # To keep the model runnable and avoid zero-variance errors, include a 0/1 column of zeros
    # only if it has variance (it won't), so we drop it from estimation but keep placeholder for output mapping.
    df["hispanic"] = np.nan  # explicit missing: cannot be estimated from provided extract

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing columns for religion/denomination: relig and/or denom")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    no_rel = (relig == 4).astype(float)
    no_rel = pd.Series(no_rel, index=df.index).astype(float)
    no_rel.loc[relig.isna()] = np.nan
    df["no_religion"] = no_rel

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Fit models (Table 2)
    # Note: hispanic cannot be estimated; we omit it from X to avoid runtime errors.
    # This is a data limitation of the provided extract.
    # -----------------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # not available in provided extract
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Guard against accidental zero-variance dummies due to subsetting:
    # if any predictor becomes constant in the estimation sample, drop it for that model but report.
    def fit_with_auto_drop(df_in, dv, xcols, model_name):
        base = df_in[[dv] + xcols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if base.shape[0] == 0:
            raise ValueError(f"{model_name}: no complete cases for DV and predictors.")

        keep = []
        dropped = []
        for c in xcols:
            v = base[c]
            if v.nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                keep.append(c)

        model, paper_tbl, full_tbl, fit_df, used_idx = ols_with_posthoc_betas(df_in, dv, keep, model_name)

        # Write an extra note about dropped predictors (if any)
        note_path = f"./output/{model_name}_notes.txt"
        with open(note_path, "w", encoding="utf-8") as f:
            f.write("Notes about estimation sample / predictors.\n")
            f.write(f"DV: {dv}\n")
            f.write(f"Initial predictors requested: {xcols}\n")
            f.write(f"Predictors dropped due to zero variance after listwise deletion: {dropped}\n")
            f.write(f"Final predictors used: {keep}\n")
            f.write(f"N used: {int(model.nobs)}\n")

        return model, paper_tbl, full_tbl, fit_df, used_idx, dropped, keep

    mA, paperA, fullA, fitA, idxA, droppedA, keepA = fit_with_auto_drop(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    write_text_tables(mA, paperA, fullA, fitA, "Table2_ModelA_dislike_minority6")

    mB, paperB, fullB, fitB, idxB, droppedB, keepB = fit_with_auto_drop(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )
    write_text_tables(mB, paperB, fullB, fitB, "Table2_ModelB_dislike_other12")

    # -----------------------------
    # Human-readable overview
    # -----------------------------
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("GSS 1993 Table 2-style replication (computed from provided extract)\n")
        f.write("Models: OLS with unstandardized intercept; standardized betas computed post-hoc for slopes.\n")
        f.write("Important: If a Table 2 covariate is not present in the extract (e.g., Hispanic), it cannot be estimated.\n")
        f.write("DV construction: dislike=1 for response 4/5 on 1-5 scale; counts sum available (non-missing) items.\n")
        f.write("Missing handling: listwise deletion at model stage on DV and included predictors.\n\n")

        f.write("Model A DV: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\nDropped predictors (zero variance): " + str(droppedA) + "\n")
        f.write("\nPaper-style coefficients (standardized slopes):\n")
        f.write(paperA.assign(coef_with_stars=paperA["coef"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "") + paperA["stars"])[["coef_with_stars"]].to_string())
        f.write("\n\n")

        f.write("Model B DV: count of dislikes among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\nDropped predictors (zero variance): " + str(droppedB) + "\n")
        f.write("\nPaper-style coefficients (standardized slopes):\n")
        f.write(paperB.assign(coef_with_stars=paperB["coef"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "") + paperB["stars"])[["coef_with_stars"]].to_string())
        f.write("\n")

    # -----------------------------
    # Return results
    # -----------------------------
    results = {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }
    return results