def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_numeric(x):
        """
        Conservative missing-code cleaning for this extract:
        - coerce non-numeric to NaN
        - treat common GSS sentinel codes as NaN
        """
        x = to_num(x)
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 scale.
          1-3 => not disliked (0)
          4-5 => disliked (1)
        Anything else or sentinel => missing.
        """
        x = clean_gss_numeric(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_numeric(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_dislike_count(df, items, require_all_items=True):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_all_items:
            # replicate: treat DK/refused/etc. as missing; exclude cases with any missing items
            return mat.sum(axis=1, min_count=len(items))
        # alternative (not used): allow partials
        return mat.sum(axis=1, min_count=1)

    def add_sig_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_standardized_ols(df, dv, xcols, model_name, dv_label):
        """
        Standardized betas computed by z-scoring DV and all predictors (including dummies)
        on the estimation (complete-case) sample, then OLS with intercept.
        """
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        # guardrail: avoid silent empty samples
        if d.shape[0] == 0:
            raise ValueError(f"{model_name}: no complete cases after listwise deletion for DV+predictors.")
        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = zscore(d[dv])
        Xz = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)

        # Drop predictors that become all-NaN or zero-variance after standardization (should be rare)
        keep = []
        dropped = []
        for c in Xz.columns:
            col = Xz[c]
            if col.notna().all() and np.isfinite(col.std(ddof=0)) and col.std(ddof=0) > 0:
                keep.append(c)
            else:
                dropped.append(c)
        Xz = Xz[keep]

        if Xz.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors dropped (no usable variation). Dropped={dropped}")

        X = sm.add_constant(Xz, has_constant="add")
        model = sm.OLS(y, X).fit()

        # Build paper-like table: standardized beta + stars (no SE column in exported table)
        rows = []
        for term in ["const"] + keep:
            beta = float(model.params.get(term, np.nan))
            p = float(model.pvalues.get(term, np.nan))
            rows.append(
                {
                    "term": term,
                    "beta_std": beta,
                    "p_value": p,
                    "sig": add_sig_stars(p) if np.isfinite(p) else "",
                    "reported": (f"{beta:.3f}{add_sig_stars(p)}") if np.isfinite(beta) and np.isfinite(p) else "",
                }
            )
        tab = pd.DataFrame(rows).set_index("term")

        # Fit block
        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_label,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_predictors_after_zscore": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save files
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv_label}\n\n")
            f.write("NOTE: Table 2 in Bryson reports standardized coefficients and stars; it does NOT report SEs.\n")
            f.write("This output includes p-values only because they are computed from the microdata regression.\n\n")
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        # Human-readable coefficient table in Table-2-like form (no SE column)
        pretty = tab.copy()
        pretty = pretty[["beta_std", "sig", "reported"]]
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv_label}\n\n")
            f.write("Standardized OLS coefficients (beta). Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed)\n\n")
            f.write(pretty.to_string(float_format=lambda x: f"{x:.6f}"))
            f.write("\n")

        return model, tab, fit

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Year filter (1993)
    if "year" not in df.columns:
        raise ValueError("Required column 'year' not found.")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.shape[0] == 0:
        raise ValueError("No rows with YEAR==1993 found.")

    # -------------------------
    # Construct DVs (counts of disliked genres)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Require complete responses across the relevant items per DV (DK treated as missing; cases excluded)
    df["dv_dislike_6_minority"] = build_dislike_count(df, minority_items, require_all_items=True)
    df["dv_dislike_12_remaining"] = build_dislike_count(df, other12_items, require_all_items=True)

    # -------------------------
    # Construct Racism score (0-5 additive; all 5 items required)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls (as available in this extract)
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column 'educ' (EDUC).")
    educ = clean_gss_numeric(df["educ"]).where(clean_gss_numeric(df["educ"]).between(0, 20))
    df["education"] = educ

    # HH income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' needed for household income per capita.")
    realinc = clean_gss_numeric(df["realinc"])
    hompop = clean_gss_numeric(df["hompop"]).where(clean_gss_numeric(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column 'prestg80' (PRESTG80).")
    df["occ_prestige"] = clean_gss_numeric(df["prestg80"])

    # Female: SEX (1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing column 'sex' (SEX).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column 'age' (AGE).")
    df["age"] = clean_gss_numeric(df["age"]).where(clean_gss_numeric(df["age"]).between(18, 89))

    # Race indicators from RACE (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing column 'race' (RACE).")
    race = clean_gss_numeric(df["race"]).where(clean_gss_numeric(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: NOT AVAILABLE in provided variables. Keep as all-NaN so we do NOT include it.
    # (We do not proxy with ETHNIC per instructions.)
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' needed for religion variables.")
    relig = clean_gss_numeric(df["relig"])
    denom = clean_gss_numeric(df["denom"])
    df["cons_protestant"] = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    df.loc[relig.isna() | denom.isna(), "cons_protestant"] = np.nan

    # No religion: RELIG==4
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column 'region' (REGION).")
    region = clean_gss_numeric(df["region"]).where(clean_gss_numeric(df["region"]).isin([1, 2, 3, 4]))
    df["southern"] = (region == 3).astype(float)
    df.loc[region.isna(), "southern"] = np.nan

    # -------------------------
    # Model specs (faithful but feasible with provided data)
    # IMPORTANT: Hispanic omitted because not present in this extract.
    # -------------------------
    xcols = [
        "racism_score",
        "education",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age",
        "black",
        "other_race",
        "cons_protestant",
        "no_religion",
        "southern",
    ]

    # Missingness diagnostics (for debugging sample-size collapse)
    diag_cols = ["dv_dislike_6_minority", "dv_dislike_12_remaining"] + xcols
    miss = pd.DataFrame(
        {
            "missing_n": df[diag_cols].isna().sum(),
            "nonmissing_n": df[diag_cols].notna().sum(),
            "missing_pct": (df[diag_cols].isna().mean() * 100.0),
            "mean_nonmissing": df[diag_cols].mean(numeric_only=True),
        }
    )
    with open("./output/Table2_missingness.txt", "w", encoding="utf-8") as f:
        f.write("Missingness diagnostics (YEAR==1993; before model-wise listwise deletion)\n\n")
        f.write(miss.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    # -------------------------
    # Fit models (standardized betas)
    # -------------------------
    modelA, tabA, fitA = fit_standardized_ols(
        df=df,
        dv="dv_dislike_6_minority",
        xcols=xcols,
        model_name="Table2_Model1_Dislike_Rap_Reggae_Blues_Jazz_Gospel_Latin",
        dv_label="Dislike count: Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin (0-6)",
    )

    modelB, tabB, fitB = fit_standardized_ols(
        df=df,
        dv="dv_dislike_12_remaining",
        xcols=xcols,
        model_name="Table2_Model2_Dislike_12_Remaining_Genres",
        dv_label="Dislike count: 12 remaining genres (0-12)",
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 style replication attempt from microdata (GSS 1993 extract).\n")
        f.write("Outputs: standardized OLS betas computed by z-scoring DV and predictors on each model's estimation sample.\n")
        f.write("Note: SEs are not part of Bryson Table 2; they are not reported in the coefficient tables here.\n")
        f.write("Note: Hispanic indicator is not available in this provided extract; therefore it is omitted.\n\n")
        f.write("Model 1 fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel 2 fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    # Return dict of DataFrames
    return {
        "Model1_table": tabA,
        "Model2_table": tabB,
        "Model1_fit": fitA,
        "Model2_fit": fitB,
        "missingness": miss,
    }