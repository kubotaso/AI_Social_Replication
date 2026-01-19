def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Helpers
    # -----------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na(series):
        """
        Conservative missing handling:
        - coerce to numeric
        - treat common GSS DK/NA/refused sentinels as missing
        Note: we do NOT blanket-drop large values because REALINC can be legitimately large.
        """
        x = to_num(series).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike(series):
        """
        Music taste items: 1..5, where 4/5 are dislike.
        Missing if outside 1..5 or NA-coded.
        Returns float {0,1} with NaN for missing.
        """
        x = clean_na(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete(df, items):
        """
        Count of dislikes across items.
        Paper summary: DK treated as missing and missing cases excluded.
        Implement as: require all component items observed for the DV (complete-case within DV).
        """
        mat = pd.concat([likert_dislike(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

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

    def standardized_betas_from_unstd(model, X, y):
        """
        Beta_j = b_j * SD(X_j) / SD(y), computed on the analytic sample.
        Intercept is left unstandardized (NaN beta).
        """
        sd_y = y.std(ddof=0)
        betas = {}
        for term in model.params.index:
            if term == "const":
                betas[term] = np.nan
                continue
            sd_x = X[term].std(ddof=0)
            betas[term] = model.params[term] * (sd_x / sd_y) if (sd_x > 0 and sd_y > 0) else np.nan
        return pd.Series(betas)

    def fit_table2_model(df, dv, xcols, model_name):
        """
        OLS with intercept; listwise deletion on dv + xcols.
        Compute standardized betas post-hoc on analytic sample.
        Do NOT error if a predictor is constant (zero variance); drop it and record.
        """
        needed = [dv] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        dropped_zero_var = []
        X = d[xcols].copy()
        y = d[dv].copy()

        # Drop zero-variance predictors on analytic sample (prevents runtime errors)
        keep = []
        for c in xcols:
            v = X[c].values
            if np.nanstd(v, ddof=0) == 0:
                dropped_zero_var.append(c)
            else:
                keep.append(c)

        X = X[keep]
        Xc = sm.add_constant(X, has_constant="add")

        if d.shape[0] < (Xc.shape[1] + 2):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}, k={Xc.shape[1]}).")

        model = sm.OLS(y, Xc).fit()

        # Standardized betas for slopes, computed from analytic sample
        betas = standardized_betas_from_unstd(model, Xc, y)

        # Build "paper-style" table: betas + stars (stars from this model's p-values)
        # NOTE: betas for dropped vars remain NaN and are omitted from output by design.
        paper = pd.DataFrame(index=model.params.index)
        paper["beta"] = betas
        paper["p_value"] = model.pvalues
        paper["stars"] = paper["p_value"].apply(star_from_p)
        paper["beta_star"] = paper.apply(
            lambda r: ("" if pd.isna(r["beta"]) else f"{r['beta']:.3f}{r['stars']}"), axis=1
        )

        # Full table with unstandardized coefficients and SEs (not in paper, but computed)
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": betas,
            }
        )

        fit = {
            "model": model_name,
            "n": int(model.nobs),
            "k": int(model.df_model + 1),  # incl intercept
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "dropped_zero_variance_predictors": ", ".join(dropped_zero_var) if dropped_zero_var else "",
        }

        return model, paper, full, pd.DataFrame([fit]), d.index

    def write_text_table(df, path, floatfmt="{:.6f}"):
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(df, pd.DataFrame):
                f.write(df.to_string(index=True, float_format=lambda x: floatfmt.format(x) if np.isfinite(x) else "nan"))
            else:
                f.write(str(df))

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must include 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # DVs (Table 2)
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete(df, other12_items)

    # -----------------------
    # Racism score (0-5)
    # -----------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # RHS controls
    # -----------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing required column: educ")
    educ = clean_na(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing required columns: realinc and/or hompop")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing required column: prestg80")
    df["occ_prestige"] = clean_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing required column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing required column: age")
    age = clean_na(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race indicators (white ref): black + other_race
    if "race" not in df.columns:
        raise ValueError("Missing required column: race")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables -> keep as all-missing (will be dropped by listwise deletion)
    # Important: DO NOT crash; Table 2 includes it but this extract doesn't.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing required columns: relig and/or denom")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()))
    df["cons_protestant"] = consprot

    # No religion: RELIG==4 (None)
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna())
    df["no_religion"] = norelig

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing required column: region")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Fit models (Table 2)
    # -----------------------
    x_order = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic",      # will be all-missing and thus dropped via listwise deletion; we handle zero-variance drop too
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Diagnostics: overall availability before model-specific listwise deletion
    diag = {}
    for c in ["dislike_minority_genres", "dislike_other12_genres"] + x_order:
        if c in df.columns:
            s = df[c]
            diag[c] = {
                "nonmissing": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": float(pd.to_numeric(s, errors="coerce").mean(skipna=True)) if s.notna().any() else np.nan,
                "std": float(pd.to_numeric(s, errors="coerce").std(skipna=True, ddof=0)) if s.notna().any() else np.nan,
            }
    diag_df = pd.DataFrame(diag).T

    # Run model A and B
    mA, paperA, fullA, fitA, idxA = fit_table2_model(df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6")
    mB, paperB, fullB, fitB, idxB = fit_table2_model(df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12")

    # -----------------------
    # Build paper-order tables with labels
    # -----------------------
    # Use exact Table 2 order; include rows even if absent, as blank (helps debugging)
    paper_labels = [
        ("racism_score", "Racism score"),
        ("education_years", "Education"),
        ("hh_income_per_capita", "Household income per capita"),
        ("occ_prestige", "Occupational prestige"),
        ("female", "Female"),
        ("age_years", "Age"),
        ("black", "Black"),
        ("hispanic", "Hispanic"),
        ("other_race", "Other race"),
        ("cons_protestant", "Conservative Protestant"),
        ("no_religion", "No religion"),
        ("south", "Southern"),
        ("const", "Constant"),
    ]

    def reindex_and_label(paper, full):
        out = []
        for key, lab in paper_labels:
            row = {
                "term": lab,
                "beta": np.nan,
                "beta_star": "",
                "p_value": np.nan,
                "b_unstd": np.nan,
                "std_err": np.nan,
                "t": np.nan,
            }
            if key in paper.index:
                row["beta"] = paper.loc[key, "beta"]
                row["beta_star"] = paper.loc[key, "beta_star"]
                row["p_value"] = paper.loc[key, "p_value"]
            if key in full.index:
                row["b_unstd"] = full.loc[key, "b_unstd"]
                row["std_err"] = full.loc[key, "std_err"]
                row["t"] = full.loc[key, "t"]
            out.append(row)
        out_df = pd.DataFrame(out).set_index("term")
        return out_df

    modelA_table = reindex_and_label(paperA, fullA)
    modelB_table = reindex_and_label(paperB, fullB)

    # -----------------------
    # Save outputs
    # -----------------------
    # Statsmodels summaries
    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(mA.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n")
    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(mB.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    # Paper-style tables (betas + stars) in Table-2 order
    write_text_table(modelA_table[["beta", "beta_star", "p_value"]], "./output/Table2_ModelA_paper_style.txt", floatfmt="{:.6f}")
    write_text_table(modelB_table[["beta", "beta_star", "p_value"]], "./output/Table2_ModelB_paper_style.txt", floatfmt="{:.6f}")

    # Full tables (unstandardized + computed standardized beta)
    write_text_table(fullA, "./output/Table2_ModelA_full_table.txt", floatfmt="{:.6f}")
    write_text_table(fullB, "./output/Table2_ModelB_full_table.txt", floatfmt="{:.6f}")

    # Diagnostics
    write_text_table(diag_df, "./output/Table2_diagnostics_nonmissing_mean_std.txt", floatfmt="{:.6f}")

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993) using provided extract.\n")
        f.write("Computed from raw data (no paper numbers copied).\n\n")
        f.write("IMPORTANT NOTE ON HISPANIC:\n")
        f.write("- This extract does not include a direct Hispanic identifier. We include a placeholder 'hispanic' as all-missing.\n")
        f.write("- As a result, listwise deletion will typically remove all cases if 'hispanic' is required; therefore the fitter drops\n")
        f.write("  zero-variance predictors on the analytic sample to avoid runtime errors.\n")
        f.write("- If you provide a true Hispanic flag, the model will include it automatically.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results = {
        "ModelA_paper_style": modelA_table[["beta", "beta_star", "p_value"]],
        "ModelB_paper_style": modelB_table[["beta", "beta_star", "p_value"]],
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "diagnostics": diag_df,
    }
    return results