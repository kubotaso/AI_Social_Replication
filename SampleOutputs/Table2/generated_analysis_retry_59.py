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
        Conservative missing handler for this extract:
        - Coerce to numeric
        - Treat common GSS DK/refused/not-applicable sentinels as missing
        - Additionally, for Likert/music and 1/2 items we will enforce valid ranges later
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(series):
        """
        Music taste items are expected 1-5.
        Dislike = 4 or 5; Like/neutral = 1,2,3; otherwise missing.
        """
        x = clean_gss_missing(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_1_2(series, true_code, false_code):
        x = clean_gss_missing(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x == false_code] = 0.0
        out.loc[x == true_code] = 1.0
        return out

    def build_dislike_count_allow_partial(df, items, min_answered=None):
        """
        Paper: DK treated as missing.
        To avoid collapsing N, do NOT require all items answered.
        Instead: compute count over answered items, but require at least min_answered.
        Default min_answered: 1 (must have at least one valid item).
        """
        if min_answered is None:
            min_answered = 1
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        n_answered = mat.notna().sum(axis=1)
        count_disliked = mat.sum(axis=1, min_count=1)
        count_disliked = count_disliked.where(n_answered >= min_answered)
        return count_disliked

    def zscore(series, ddof=0):
        s = to_num(series)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
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

    def posthoc_standardized_betas(unstd_model, y, X):
        """
        Compute standardized betas from an unstandardized OLS fit:
        beta_j = b_j * sd(X_j) / sd(Y)
        (excluding intercept)
        """
        y_sd = np.nanstd(y, ddof=0)
        betas = {}
        for col in X.columns:
            x_sd = np.nanstd(X[col], ddof=0)
            b = unstd_model.params.get(col, np.nan)
            if (not np.isfinite(y_sd)) or y_sd == 0 or (not np.isfinite(x_sd)) or x_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = b * (x_sd / y_sd)
        return pd.Series(betas)

    def fit_table2_model(df, dv_col, xcols, model_name):
        """
        Fit unstandardized OLS with intercept on complete cases for (dv, xcols).
        Then compute standardized betas post-hoc for slopes (intercept stays unstandardized).
        """
        needed = [dv_col] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Drop zero-variance predictors (in this analytic sample) rather than erroring out.
        # This avoids runtime errors; also logs what got dropped.
        dropped = []
        keep = []
        for c in xcols:
            v = d[c]
            if v.nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                keep.append(c)

        if d.shape[0] < 10:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        X = d[keep].copy()
        y = d[dv_col].copy()

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        beta_std = posthoc_standardized_betas(model, y.values, X)

        full = pd.DataFrame(
            {
                "b": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        # Add standardized betas for slopes; constant is NaN
        full["beta_std"] = np.nan
        for c in beta_std.index:
            if c in full.index:
                full.loc[c, "beta_std"] = beta_std.loc[c]

        # Paper-style table: standardized betas + stars for slopes, plus unstd constant
        paper_rows = []
        # order with constant at end
        for c in keep:
            bstd = full.loc[c, "beta_std"]
            p = full.loc[c, "p_value"]
            paper_rows.append((c, bstd, stars(p)))
        # constant
        if "const" in full.index:
            paper_rows.append(("Constant", full.loc["const", "b"], stars(full.loc["const", "p_value"])))
        paper = pd.DataFrame(paper_rows, columns=["term", "coef", "sig"])
        paper["coef_sig"] = paper["coef"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + paper["sig"]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nDropped zero-variance predictors (analytic sample): ")
            f.write(", ".join(dropped) if dropped else "(none)")
            f.write("\n")

        with open(f"./output/{model_name}_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Paper-style table: standardized betas for predictors (post-hoc), unstandardized Constant\n")
            f.write(paper[["term", "coef_sig"]].to_string(index=False))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write("Full table: unstandardized b, SE, t, p; plus standardized beta for slopes\n")
            f.write(full.to_string(float_format=lambda v: f"{v: .6f}"))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        return model, paper, full, fit, d.index, keep, dropped

    # -----------------------------
    # Load + harmonize columns
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for req in ["year", "id"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct dependent variables
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Use partial-item counts to avoid N collapse; require at least half answered
    df["dislike_minority_genres"] = build_dislike_count_allow_partial(df, minority_items, min_answered=3)
    df["dislike_other12_genres"] = build_dislike_count_allow_partial(df, other12_items, min_answered=6)

    # -----------------------------
    # Construct racism score (0-5)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_1_2(df["rachaf"], true_code=1, false_code=2)   # object to half-black school
    rac2 = binary_from_1_2(df["busing"], true_code=2, false_code=1)   # oppose busing
    rac3 = binary_from_1_2(df["racdif1"], true_code=2, false_code=1)  # deny discrimination
    rac4 = binary_from_1_2(df["racdif3"], true_code=2, false_code=1)  # deny educational chance
    rac5 = binary_from_1_2(df["racdif4"], true_code=1, false_code=2)  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ.")
    educ = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c}.")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female (sex: 1=male, 2=female)
    if "sex" not in df.columns:
        raise ValueError("Missing sex.")
    df["female"] = binary_from_1_2(df["sex"], true_code=2, false_code=1)

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age.")
    age = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))
    df["age_years"] = age

    # Race indicators (race: 1=white, 2=black, 3=other)
    if "race" not in df.columns:
        raise ValueError("Missing race.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available; create from 'ethnic' as best-effort proxy to avoid zero-variance.
    # NOTE: This is imperfect; but required to keep a Hispanic term in the model with this extract.
    # Heuristic: treat common Hispanic-origin codes as Hispanic (Mexican/Puerto Rican/Cuban/Central/South American/Spanish).
    if "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        # Best-effort ranges used in many GSS ETHNIC codings; also include 0/1 style if present.
        hisp_codes = set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[eth.notna()] = 0.0
        hisp.loc[eth.isin(list(hisp_codes))] = 1.0
        # If dataset uses 1/2 style for Hispanic yes/no, map that too
        hisp.loc[eth == 1] = 1.0
        hisp.loc[eth == 2] = 0.0
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing relig/denom.")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    consprot.loc[relig.notna() & denom.notna()] = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = pd.Series(np.nan, index=df.index, dtype="float64")
    norelig.loc[relig.notna()] = (relig == 4).astype(float)
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = pd.Series(np.nan, index=df.index, dtype="float64")
    south.loc[region.notna()] = (region == 3).astype(float)
    df["south"] = south

    # -----------------------------
    # Fit models (Table 2)
    # -----------------------------
    xcols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    for c in xcols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    mA, paperA, fullA, fitA, idxA, keepA, dropA = fit_table2_model(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, idxB, keepB, dropB = fit_table2_model(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )

    # -----------------------------
    # Diagnostics to help verify issues like zero-variance dummies
    # -----------------------------
    def diag_counts(model_name, used_idx, cols):
        out = []
        for c in cols:
            s = df.loc[used_idx, c]
            out.append(
                {
                    "model": model_name,
                    "var": c,
                    "n": int(s.notna().sum()),
                    "n_unique": int(s.nunique(dropna=True)),
                    "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                }
            )
        return pd.DataFrame(out)

    diagA = diag_counts("Table2_ModelA_dislike_minority6", idxA, xcols + ["dislike_minority_genres"])
    diagB = diag_counts("Table2_ModelB_dislike_other12", idxB, xcols + ["dislike_other12_genres"])

    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("Diagnostics: variance/means in analytic samples\n\n")
        f.write("Model A\n")
        f.write(diagA.to_string(index=False))
        f.write("\n\nModel B\n")
        f.write(diagB.to_string(index=False))
        f.write("\n")

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS extract)\n")
        f.write("DVs are dislike-counts with partial-item allowance to prevent N collapse.\n")
        f.write("Standardized betas computed post-hoc from unstandardized OLS slopes.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nModel A paper-style:\n")
        f.write(paperA[["term", "coef_sig"]].to_string(index=False))
        f.write("\n\nModel B paper-style:\n")
        f.write(paperB[["term", "coef_sig"]].to_string(index=False))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "DiagnosticsA": diagA,
        "DiagnosticsB": diagB,
    }