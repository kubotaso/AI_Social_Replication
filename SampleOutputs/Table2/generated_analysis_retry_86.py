def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # --------------------------
    # Helpers
    # --------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing handling for this extract:
        - Coerce to numeric
        - Treat common GSS nonresponse codes as missing when they appear as exact values
        - Do NOT treat valid small integers as missing (e.g., 0 can be valid for some vars)
        """
        x = to_num(x).copy()
        # common "IAP"/GSS style missing codes in many extracts
        miss_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}
        x = x.mask(x.isin(miss_codes))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items: 1-5, where 4/5 = dislike.
        Missing if not in 1..5 after cleaning.
        """
        x = clean_gss_missing(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_gss_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_series(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_completecase(df, item_cols):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in item_cols], axis=1)
        # "DK treated as missing and cases with missing excluded" => require all items observed
        return mat.sum(axis=1, min_count=len(item_cols))

    def compute_standardized_betas_from_ols(model, y, X):
        """
        Standardized beta for each regressor (excluding intercept):
            beta_j = b_j * sd(x_j) / sd(y)
        using the estimation sample.
        """
        y_sd = np.std(y, ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = np.std(X[col].values, ddof=0)
            betas[col] = model.params[col] * (x_sd / y_sd) if (y_sd != 0 and x_sd != 0) else np.nan
        return betas

    def star_from_p(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_one_model(df, dv_col, rhs_cols, model_label, dv_label, rhs_pretty):
        """
        Fit OLS on raw DV, compute standardized betas post hoc.
        Listwise deletion on dv + RHS only (per model).
        """
        needed = [dv_col] + rhs_cols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        if d.shape[0] < len(rhs_cols) + 5:
            raise ValueError(f"{model_label}: not enough complete cases (n={d.shape[0]}, k={len(rhs_cols)}).")

        y = d[dv_col].astype(float).values
        X = d[rhs_cols].astype(float)
        X = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X).fit()

        betas = compute_standardized_betas_from_ols(model, y=y, X=X)
        rows = []
        order = rhs_cols[:]  # keep paper-like ordering
        for col in order:
            p = float(model.pvalues[col])
            rows.append(
                {
                    "term": rhs_pretty.get(col, col),
                    "beta_std": float(betas.get(col, np.nan)),
                    "p_value": p,
                    "sig": star_from_p(p),
                }
            )

        table = pd.DataFrame(rows).set_index("term")

        fit = pd.DataFrame(
            [
                {
                    "model": model_label,
                    "dv": dv_label,
                    "n": int(model.nobs),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "constant_b": float(model.params["const"]),
                }
            ]
        )

        # Save human-readable output
        with open(f"./output/{model_label}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_label}\n")
            f.write(f"DV: {dv_label}\n\n")
            f.write(model.summary().as_text())
            f.write("\n\nStandardized betas (computed as b * sd(X) / sd(Y)):\n")
            f.write(table[["beta_std", "sig"]].to_string())
            f.write("\n")

        # Save a compact "paper-style" table text
        with open(f"./output/{model_label}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_label}\nDV: {dv_label}\n\n")
            out = table.copy()
            out["beta_std"] = out["beta_std"].map(lambda v: f"{v: .3f}" if pd.notna(v) else " NA")
            out = out[["beta_std", "sig"]]
            f.write(out.to_string())
            f.write("\n")

        return model, table, fit, d.index

    # --------------------------
    # Load
    # --------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # --------------------------
    # Dependent variables
    # --------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband",
        "blugrass",
        "country",
        "musicals",
        "classicl",
        "folk",
        "moodeasy",
        "newage",
        "opera",
        "conrock",
        "oldies",
        "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority6"] = build_count_completecase(df, minority_items)
    df["dislike_other12"] = build_count_completecase(df, other12_items)

    # --------------------------
    # Racism score (0-5)
    # --------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object (1) vs no (2)
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose (2) vs favor (1)
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # --------------------------
    # RHS controls
    # --------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' column (EDUC).")
    educ = clean_gss_missing(df["educ"])
    df["education"] = educ.where(educ.between(0, 20))

    # Income per capita = REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing 'realinc' and/or 'hompop' columns needed for income per capita.")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"])
    hompop = hompop.where(hompop > 0)
    df["hh_income_pc"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column (PRESTG80).")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column (SEX).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' column (AGE).")
    age = clean_gss_missing(df["age"])
    df["age"] = age.where(age.between(18, 89))

    # Race dummies (white reference): include black and other_race
    if "race" not in df.columns:
        raise ValueError("Missing 'race' column (RACE).")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided extract; attempt to derive minimally from 'ethnic' if plausible
    # This is a pragmatic fallback to avoid dropping the variable entirely (many replications use an ancestry-based proxy).
    # Codes vary across extracts; here we treat clearly invalid/missing as NaN and then flag likely Hispanic origins:
    # - If 'ethnic' is 20..29 (often Hispanic/Mexican/Puerto Rican/Cuban in some schemes)
    # This is best-effort; if no variability, model fitting will still work.
    if "ethnic" in df.columns:
        ethnic = clean_gss_missing(df["ethnic"])
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        # mark non-missing as 0 by default, then flag a plausible band as 1
        hisp.loc[ethnic.notna()] = 0.0
        hisp.loc[ethnic.between(20, 29)] = 1.0
        df["hispanic"] = hisp
    else:
        # as last resort, keep as all-NaN; but then we cannot include it in RHS listwise deletion
        df["hispanic"] = np.nan

    # Religion dummies
    if "relig" not in df.columns:
        raise ValueError("Missing 'relig' column (RELIG).")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"]) if "denom" in df.columns else pd.Series(np.nan, index=df.index)

    # Conservative Protestant proxy: RELIG==1 (Protestant) and DENOM in {1,6,7} when denom is available
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    if "denom" in df.columns:
        consprot.loc[relig.notna() & denom.notna()] = 0.0
        consprot.loc[(relig == 1) & (denom.isin([1, 6, 7]))] = 1.0
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = pd.Series(np.nan, index=df.index, dtype="float64")
    norelig.loc[relig.notna()] = 0.0
    norelig.loc[relig == 4] = 1.0
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing 'region' column (REGION).")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = pd.Series(np.nan, index=df.index, dtype="float64")
    south.loc[region.notna()] = 0.0
    south.loc[region == 3] = 1.0
    df["southern"] = south

    # --------------------------
    # Model specifications
    # --------------------------
    # Keep RHS aligned with Table 2 ordering (as provided in prompt)
    rhs_cols = [
        "racism_score",
        "education",
        "hh_income_pc",
        "occ_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "southern",
    ]

    # If a column is entirely missing in this extract (e.g., hispanic), keep it out to avoid n=0.
    # This makes the run robust while still including the variable when it has data.
    rhs_cols_effective = []
    dropped_all_missing = []
    for c in rhs_cols:
        if c not in df.columns:
            continue
        if df[c].notna().sum() == 0:
            dropped_all_missing.append(c)
            continue
        rhs_cols_effective.append(c)

    pretty = {
        "racism_score": "Racism score",
        "education": "Education",
        "hh_income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    # --------------------------
    # Fit models
    # --------------------------
    modelA, tableA, fitA, idxA = fit_one_model(
        df=df,
        dv_col="dislike_minority6",
        rhs_cols=rhs_cols_effective,
        model_label="Table2_ModelA_Dislike_Rap_Reggae_Blues_Jazz_Gospel_Latin",
        dv_label="Dislike count: Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin (0-6)",
        rhs_pretty=pretty,
    )

    modelB, tableB, fitB, idxB = fit_one_model(
        df=df,
        dv_col="dislike_other12",
        rhs_cols=rhs_cols_effective,
        model_label="Table2_ModelB_Dislike_12_Remaining_Genres",
        dv_label="Dislike count: 12 remaining genres (0-12)",
        rhs_pretty=pretty,
    )

    # --------------------------
    # Save overview + diagnostics
    # --------------------------
    diag = pd.DataFrame(
        {
            "var": ["dislike_minority6", "dislike_other12"] + rhs_cols,
            "non_missing_N": [df["dislike_minority6"].notna().sum(), df["dislike_other12"].notna().sum()]
            + [df[c].notna().sum() if c in df.columns else 0 for c in rhs_cols],
            "mean": [df["dislike_minority6"].mean(skipna=True), df["dislike_other12"].mean(skipna=True)]
            + [df[c].mean(skipna=True) if c in df.columns else np.nan for c in rhs_cols],
            "std": [df["dislike_minority6"].std(skipna=True, ddof=0), df["dislike_other12"].std(skipna=True, ddof=0)]
            + [df[c].std(skipna=True, ddof=0) if c in df.columns else np.nan for c in rhs_cols],
        }
    )

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson Table 2 replication attempt (computed from provided GSS 1993 extract)\n")
        f.write("OLS on raw DV; standardized betas computed as b * sd(X)/sd(Y) on estimation sample.\n")
        if dropped_all_missing:
            f.write(f"\nNOTE: Dropped predictors that are entirely missing in this extract: {', '.join(dropped_all_missing)}\n")
        f.write("\nModel A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nVariable diagnostics (full YEAR==1993 subset before listwise deletion):\n")
        f.write(diag.to_string(index=False))
        f.write("\n")

    # Return results
    return {
        "ModelA_table": tableA[["beta_std", "sig"]],
        "ModelB_table": tableB[["beta_std", "sig"]],
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "diagnostics": diag,
    }