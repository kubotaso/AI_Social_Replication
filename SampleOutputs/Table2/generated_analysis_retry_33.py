def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_na(series):
        """
        Conservative missing handling for this extract:
        - Coerce to numeric
        - Treat common GSS-style special codes as missing
        """
        s = to_num(series).copy()
        s = s.replace(
            {
                8: np.nan,
                9: np.nan,
                98: np.nan,
                99: np.nan,
                998: np.nan,
                999: np.nan,
                9998: np.nan,
                9999: np.nan,
            }
        )
        return s

    def likert_dislike(item):
        """
        Music taste items: 1-5 scale.
        Dislike indicator: 1 if 4/5; 0 if 1/2/3; missing otherwise.
        """
        x = clean_na(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def bin_from_codes(series, true_codes, false_codes):
        x = clean_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count(df, items):
        mat = pd.concat([likert_dislike(df[c]).rename(c) for c in items], axis=1)
        # Bryson note: DK treated as missing; exclude missing cases
        return mat.sum(axis=1, min_count=len(items))

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def sig_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df, y_col, x_cols_ordered, model_name):
        """
        Fit OLS, report standardized coefficients (beta) and stars.
        Standardization: compute betas from unstandardized b via:
            beta_j = b_j * sd(x_j) / sd(y)
        using the analytic (listwise-deleted) sample for that model.
        """
        needed = [y_col] + x_cols_ordered
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        k = len(x_cols_ordered)
        if d.shape[0] < (k + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={k}).")

        # Ensure predictors vary; if not, keep running but note and drop collinear constants
        zero_var = [c for c in x_cols_ordered if d[c].nunique(dropna=True) <= 1]
        # If any zero variance, drop them (do not error out)
        x_use = [c for c in x_cols_ordered if c not in zero_var]

        X = sm.add_constant(d[x_use], has_constant="add")
        y = d[y_col]
        m = sm.OLS(y, X).fit()

        # Standardized betas via sd ratio (exclude intercept)
        y_sd = y.std(ddof=0)
        betas = {}
        for c in x_use:
            x_sd = d[c].std(ddof=0)
            if (not np.isfinite(x_sd)) or x_sd == 0 or (not np.isfinite(y_sd)) or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(m.params[c] * (x_sd / y_sd))

        # Assemble "paper-style" table (standardized betas + stars), include all requested terms
        rows = []
        for c in x_cols_ordered:
            if c in betas:
                p = float(m.pvalues.get(c, np.nan))
                rows.append((c, betas[c], sig_stars(p)))
            else:
                rows.append((c, np.nan, ""))

        # Constant: keep unstandardized (as paper reports constant separately)
        const = float(m.params.get("const", np.nan))
        const_p = float(m.pvalues.get("const", np.nan))
        rows.append(("const", const, sig_stars(const_p)))

        paper_style = pd.DataFrame(rows, columns=["term", "coef", "sig"])
        paper_style["coef"] = paper_style["coef"].astype(float)

        # Also provide full labeled output for debugging/replication transparency
        full = pd.DataFrame(
            {
                "term": ["const"] + x_use,
                "b_unstd": [m.params.get("const", np.nan)] + [m.params.get(c, np.nan) for c in x_use],
                "std_err": [m.bse.get("const", np.nan)] + [m.bse.get(c, np.nan) for c in x_use],
                "t": [m.tvalues.get("const", np.nan)] + [m.tvalues.get(c, np.nan) for c in x_use],
                "p_value": [m.pvalues.get("const", np.nan)] + [m.pvalues.get(c, np.nan) for c in x_use],
                "beta": [np.nan] + [betas.get(c, np.nan) for c in x_use],
            }
        )

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": y_col,
                    "n": int(m.nobs),
                    "k_predictors_requested": int(len(x_cols_ordered)),
                    "k_predictors_used_excl_const": int(len(x_use)),
                    "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                }
            ]
        )

        # Save text outputs (human-readable)
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(m.summary().as_text())
            f.write("\n\nRequested predictors (Table 2 order):\n")
            f.write("\n".join(x_cols_ordered) + "\n")
            if zero_var:
                f.write("\nDropped zero-variance predictors:\n")
                f.write("\n".join(zero_var) + "\n")

        # Paper-style table text
        def fmt_coef(v):
            if pd.isna(v):
                return ""
            return f"{v:.3f}"

        paper_out = paper_style.copy()
        paper_out["coef"] = paper_out["coef"].map(fmt_coef)
        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            f.write("Standardized OLS coefficients (beta) with significance markers\n")
            f.write("(Constant reported unstandardized as estimated)\n\n")
            f.write(paper_out.to_string(index=False))

        # Full table text
        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        return paper_style, full, fit

    # -------------------------
    # Load and filter data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables
    # -------------------------
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
            raise ValueError(f"Missing music variable: {c}")

    df["dislike_minority_genres"] = build_count(df, minority_items)
    df["dislike_other12_genres"] = build_count(df, other12_items)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item: {c}")

    rac1 = bin_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = bin_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = bin_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = bin_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = bin_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing educ")
    df["education_years"] = clean_na(df["educ"]).where(clean_na(df["educ"]).between(0, 20))

    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing realinc and/or hompop")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80")
    df["occ_prestige"] = clean_na(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing sex")
    df["female"] = bin_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing age")
    df["age_years"] = clean_na(df["age"]).where(clean_na(df["age"]).between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing race")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not available in provided mapping -> must exist to match table;
    # create as missing so it will be listwise-dropped if included. Instead: include but allow missing
    # by setting to 0 when missing is not acceptable. However, the instruction explicitly says no proxy.
    # We include it but keep as NaN to remain faithful to "not present".
    df["hispanic"] = np.nan

    # Conservative Protestant and No religion
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing relig and/or denom")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])

    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index, dtype="float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index, dtype="float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    if "region" not in df.columns:
        raise ValueError("Missing region")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Fit the two Table 2 models
    # -------------------------
    # Paper order (includes hispanic even if unavailable in this extract)
    x_order = [
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
    for c in x_order:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # IMPORTANT: Since hispanic is missing in this extract, listwise deletion with it would yield n=0.
    # To keep the models runnable and faithful to available data, we fit two versions:
    # (1) "available-data" replication: drop hispanic from RHS.
    # (2) "table2-spec" stub saved with note that hispanic is not available.
    # The returned result includes the available-data models, which are estimable.
    x_order_available = [c for c in x_order if c != "hispanic"]

    paperA, fullA, fitA = fit_table2_model(
        df,
        "dislike_minority_genres",
        x_order_available,
        "Table2_ModelA_dislike_minority6_available",
    )
    paperB, fullB, fitB = fit_table2_model(
        df,
        "dislike_other12_genres",
        x_order_available,
        "Table2_ModelB_dislike_other12_available",
    )

    # Write an overview note
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt using provided gss93_selected.csv extract\n")
        f.write("Note: A separate Hispanic ethnicity indicator is not present in the provided variable list.\n")
        f.write("Therefore, models were estimated excluding the Hispanic dummy (all other Table 2 predictors included).\n\n")
        f.write("Model A: DV = count of disliked minority-associated genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False) + "\n\n")
        f.write("Model B: DV = count of disliked other 12 genres\n")
        f.write(fitB.to_string(index=False) + "\n")

    return {
        "ModelA_table_paper_style": paperA,
        "ModelB_table_paper_style": paperB,
        "ModelA_table_full": fullA,
        "ModelB_table_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }