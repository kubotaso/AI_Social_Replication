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

    def clean_na_codes(x):
        """
        Conservative missing-code handling for numeric GSS extracts.
        We only blank out common sentinel values; we do NOT do aggressive trimming that can halve N.
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 where 4/5 indicate dislike.
        - 1,2,3 -> 0
        - 4,5   -> 1
        - else/missing/sentinel -> NaN
        """
        x = clean_na_codes(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_na_codes(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Build count of dislikes across items.
        Paper note: DK treated as missing and missing cases excluded.
        Implement: require all component items observed (complete-case) for DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def weighted_mean(x, w):
        return (w * x).sum() / w.sum()

    def weighted_var(x, w):
        mu = weighted_mean(x, w)
        return (w * (x - mu) ** 2).sum() / w.sum()

    def compute_standardized_betas_from_fit(fit, X, y, w=None):
        """
        Standardized beta_j = b_j * SD(X_j)/SD(y), computed on the SAME estimation sample.
        Intercept has no standardized beta (NaN).
        If weights are provided, use weighted SDs (population-style, consistent with weighted_var above).
        """
        params = fit.params.copy()
        beta = pd.Series(index=params.index, dtype="float64")
        beta.loc[:] = np.nan

        if w is None:
            y_sd = y.std(ddof=0)
        else:
            y_sd = np.sqrt(weighted_var(y, w))

        for term in params.index:
            if term == "const":
                beta.loc[term] = np.nan
                continue
            xj = X[term]
            if w is None:
                x_sd = xj.std(ddof=0)
            else:
                x_sd = np.sqrt(weighted_var(xj, w))
            if (not np.isfinite(x_sd)) or x_sd == 0 or (not np.isfinite(y_sd)) or y_sd == 0:
                beta.loc[term] = np.nan
            else:
                beta.loc[term] = params.loc[term] * (x_sd / y_sd)
        return beta

    def fit_table2_model(df, dv, x_terms_ordered, model_name, w_col=None):
        """
        Fit OLS (or WLS if w_col given), then compute standardized betas (slopes only).
        Output ONLY standardized betas + stars + unstandardized constant (as typical in sociology tables).
        Also save a clear note: SEs not available in Table 2; stars computed from replication p-values.
        """
        cols_needed = [dv] + x_terms_ordered + ([w_col] if w_col else [])
        d = df[cols_needed].copy()

        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")
        if d.shape[0] < len(x_terms_ordered) + 5:
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion: n={d.shape[0]}")

        # Build design matrix in fixed order
        X = d[x_terms_ordered].copy()
        # Fail fast if any predictor has zero variance in THIS analytic sample
        zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Check coding/sample restrictions."
            )

        Xc = sm.add_constant(X, has_constant="add")
        y = d[dv].astype(float)

        if w_col is None:
            fit = sm.OLS(y, Xc).fit()
            w = None
        else:
            w = d[w_col].astype(float)
            fit = sm.WLS(y, Xc, weights=w).fit()

        # Standardized betas computed from this same estimation sample
        beta = compute_standardized_betas_from_fit(fit, Xc, y, w=w)

        # Assemble "paper-style" table in the exact row order requested
        # (include constant as unstandardized coefficient; standardized beta left blank for constant)
        rows = []
        for term in ["racism_score",
                     "education_years",
                     "hh_income_per_capita",
                     "occ_prestige",
                     "female",
                     "age",
                     "black",
                     "hispanic",
                     "other_race",
                     "cons_protestant",
                     "no_religion",
                     "south",
                     "const"]:
            if term == "const":
                b0 = float(fit.params.get("const", np.nan))
                p0 = float(fit.pvalues.get("const", np.nan))
                rows.append(
                    {
                        "term": "Constant",
                        "coef": b0,
                        "stars": stars_from_p(p0),
                    }
                )
            else:
                b = float(beta.get(term, np.nan))
                p = float(fit.pvalues.get(term, np.nan))
                rows.append(
                    {
                        "term": term,
                        "coef": b,
                        "stars": stars_from_p(p),
                    }
                )

        paper = pd.DataFrame(rows)

        # Validate row labels present and no accidental NaN label
        if paper["term"].isna().any():
            raise ValueError(f"{model_name}: table assembly failed (NaN term label).")
        if paper.shape[0] != 13:
            raise ValueError(f"{model_name}: expected 13 rows (12 predictors + constant), got {paper.shape[0]}.")

        # Fit stats
        fit_stats = pd.DataFrame(
            [{
                "model": model_name,
                "n": int(round(fit.nobs)),
                "r2": float(fit.rsquared),
                "adj_r2": float(fit.rsquared_adj),
            }]
        )

        # Save outputs as human-readable text
        note = (
            "NOTE: Table 2 in the paper does not report SEs. "
            "Stars here are computed from replication-model p-values "
            "(two-tailed; * p<.05, ** p<.01, *** p<.001) on the same estimation sample as the betas.\n"
        )
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(note)
            # Display with coef + stars combined, matching common paper layout
            disp = paper.copy()
            def fmt_row(r):
                if pd.isna(r["coef"]):
                    return ""
                return f"{r['coef']:.3f}{r['stars']}"
            disp["coef_star"] = disp.apply(fmt_row, axis=1)
            out = disp[["term", "coef_star"]].to_string(index=False)
            f.write(out + "\n")
            f.write("\nFit:\n")
            f.write(fit_stats.to_string(index=False) + "\n")

        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(note + "\n")
            f.write("Replication regression summary (SEs/p-values are replication outputs; not printed in Table 2):\n\n")
            f.write(fit.summary().as_text() + "\n")

        return paper, fit_stats

    # -----------------------
    # Load and filter
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Build dependent variables
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------
    # Racism score (0-5)
    # -----------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls
    # -----------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    df["education_years"] = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))

    # Household income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required income column: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age (do not enforce 18+ to avoid unnecessary N loss)
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    df["age"] = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(0, 89))

    # Race dummies: black / other (white ref), plus hispanic proxy not available in this extract.
    # IMPORTANT: We must not invent a proxy. If not present, we cannot include it.
    # However, the replication harness expects the Table 2 specification; this dataset lacks a hispanic flag.
    # To keep the model runnable and faithful to available data, we:
    #   - include a hispanic column if it exists, else set to 0/NaN? Setting to 0 creates zero-variance risk.
    # Best choice: if absent, create from 'ethnic' ONLY if it is explicitly a Hispanic indicator (it is not here),
    # so we will instead create as missing and then explicitly error with a clear message.
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not provided in this extract. If a hispanic column exists, use it; else stop with a clear error.
    if "hispanic" in df.columns:
        hisp = clean_na_codes(df["hispanic"])
        # Expect 0/1 or 1/2 style; handle common patterns
        if set(hisp.dropna().unique()).issubset({0, 1}):
            df["hispanic"] = hisp.astype(float)
        else:
            # Try 1=yes 2=no
            df["hispanic"] = binary_from_codes(hisp, true_codes=[1], false_codes=[2])
    else:
        raise ValueError(
            "This dataset extract does not include a Hispanic identifier, but Table 2 requires it. "
            "Provide a dataset with a proper Hispanic flag/variable, or add it to the input."
        )

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Fit models (Table 2 spec)
    # -----------------------
    x_order = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Ensure all x columns exist
    for c in x_order:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    paperA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6", w_col=None
    )
    paperB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12", w_col=None
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (1993 GSS extract): standardized betas + stars.\n")
        f.write("SEs not available in Table 2; stars computed from replication-model p-values.\n\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False) + "\n\n")
        f.write("Model B DV: dislike_other12_genres (12 remaining genres)\n")
        f.write(fitB.to_string(index=False) + "\n")

    return {
        "ModelA_table": paperA,
        "ModelB_table": paperB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }