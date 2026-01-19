def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_missing(x):
        """
        Conservative missing cleaning:
        - coerce non-numeric to NaN
        - treat common GSS sentinel codes as NaN
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(sentinels)))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 scale.
        dislike = 1 if 4 or 5
        like/neutral = 0 if 1,2,3
        missing otherwise
        """
        x = clean_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_dislike_count(df, items, require_complete=True):
        """
        Count of disliked genres across a set of items.
        DK/NA are missing at item level.
        If require_complete=True: DV is missing unless all items observed (paper-style listwise DV).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_complete:
            return mat.sum(axis=1, min_count=len(items))
        else:
            # not used, but kept for completeness
            return mat.sum(axis=1, min_count=1)

    def format_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_unstandardized_and_betas(df, dv, xcols, model_name):
        """
        Fit unstandardized OLS with intercept using listwise deletion on dv+xcols.
        Compute standardized betas post-hoc: beta_j = b_j * sd(x_j) / sd(y).
        (Intercept remains unstandardized.)
        """
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(how="any").copy()

        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: too few complete cases (n={d.shape[0]}, k={len(xcols)}).")

        # Drop zero-variance predictors (but do NOT error; report what was dropped)
        dropped_zero_var = []
        kept = []
        for c in xcols:
            v = to_num(d[c])
            sd = v.std(skipna=True, ddof=0)
            if not np.isfinite(sd) or sd == 0:
                dropped_zero_var.append(c)
            else:
                kept.append(c)

        if len(kept) == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance after listwise deletion.")

        y = to_num(d[dv])
        X = pd.DataFrame({c: to_num(d[c]) for c in kept}, index=d.index)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # Standardized betas post-hoc
        y_sd = y.std(skipna=True, ddof=0)
        betas = {}
        for c in kept:
            x_sd = X[c].std(skipna=True, ddof=0)
            betas[c] = model.params[c] * (x_sd / y_sd) if (np.isfinite(x_sd) and np.isfinite(y_sd) and y_sd != 0) else np.nan

        # Paper-style table: standardized betas for slopes, unstandardized intercept
        paper_rows = []
        # Preserve original xcols order, inserting NaN for dropped predictors (to keep table structure)
        for c in xcols:
            if c in kept:
                paper_rows.append(
                    {
                        "term": c,
                        "beta": float(betas[c]),
                        "stars": format_stars(model.pvalues.get(c, np.nan)),
                        "p_value": float(model.pvalues.get(c, np.nan)),
                    }
                )
            else:
                paper_rows.append({"term": c, "beta": np.nan, "stars": "", "p_value": np.nan})

        intercept = float(model.params.get("const", np.nan))
        intercept_p = float(model.pvalues.get("const", np.nan))
        intercept_row = pd.DataFrame(
            [{"term": "const", "beta": np.nan, "stars": format_stars(intercept_p), "p_value": intercept_p}]
        )

        paper = pd.DataFrame(paper_rows)
        paper = pd.concat([paper, intercept_row], ignore_index=True)

        full = pd.DataFrame(
            {
                "term": model.params.index,
                "b": model.params.values,
                "std_err": model.bse.values,
                "t": model.tvalues.values,
                "p_value": model.pvalues.values,
            }
        )
        # add beta_std for non-const terms (const beta not defined)
        full["beta_std_posthoc"] = np.nan
        for c in kept:
            full.loc[full["term"] == c, "beta_std_posthoc"] = betas[c]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped_zero_var) if dropped_zero_var else "",
                    "dv_mean": float(y.mean()),
                    "dv_sd": float(y.std(ddof=0)),
                    "intercept_unstd": intercept,
                }
            ]
        )

        return model, paper, full, fit, d.index

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year == 1993
    if "year" not in df.columns:
        raise ValueError("Missing column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # DVs (Table 2)
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Require complete genre responses within each DV, matching "DK treated as missing and cases excluded"
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_complete=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_complete=True)

    # ----------------------------
    # Racism scale (0-5)
    # ----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # RHS controls
    # ----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(clean_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies from RACE: 1 white, 2 black, 3 other
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_missing(df["race"]).where(clean_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not available in provided variables; keep column but do not force failure.
    # Set to NaN so it is excluded by listwise deletion only if included; to keep model runnable,
    # we include it but allow it to be dropped for zero-variance/NaN by fit routine.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns:
        raise ValueError("Missing column: relig")
    relig = clean_missing(df["relig"])
    denom = clean_missing(df["denom"]) if "denom" in df.columns else pd.Series(np.nan, index=df.index)
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_missing(df["region"]).where(clean_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # ----------------------------
    # Run the two Table 2 models (same RHS)
    # ----------------------------
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

    results = {}

    def save_outputs(model, paper, full, fit, model_name):
        # Save summary
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        # Save "paper-style" table: standardized betas + stars; intercept unstandardized is in fit
        paper_out = paper.copy()
        paper_out["beta_star"] = paper_out.apply(
            lambda r: ("" if pd.isna(r["beta"]) else f"{r['beta']:.3f}{r['stars']}"), axis=1
        )
        paper_view = paper_out[["term", "beta_star"]].copy()

        with open(f"./output/{model_name}_paper_table.txt", "w", encoding="utf-8") as f:
            f.write("Table 2 style output: standardized coefficients (betas) for slopes; const is unstandardized in fit.\n")
            f.write(paper_view.to_string(index=False))
            f.write("\n\nNote: stars are based on p-values from this re-estimation.\n")

        # Save full regression table (clearly labeled as computed)
        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write("Full OLS output (computed from dataset; not reported in the paper's Table 2):\n")
            f.write(full.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
            f.write("\n")

        # Save fit
        fit.to_csv(f"./output/{model_name}_fit.csv", index=False)

        return paper_view, full

    mA, paperA, fullA, fitA, idxA = fit_unstandardized_and_betas(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    paperA_view, fullA_out = save_outputs(mA, paperA, fullA, fitA, "Table2_ModelA_dislike_minority6")

    mB, paperB, fullB, fitB, idxB = fit_unstandardized_and_betas(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )
    paperB_view, fullB_out = save_outputs(mB, paperB, fullB, fitB, "Table2_ModelB_dislike_other12")

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("GSS 1993 Table 2 replication attempt: OLS with post-hoc standardized betas.\n")
        f.write("DVs are dislike counts; DK/NA treated as missing; DV requires complete responses within the genre set.\n")
        f.write("Important: 'hispanic' is not available in provided columns; it will be dropped by zero-variance/NaN handling.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results["ModelA_paper"] = paperA_view
    results["ModelB_paper"] = paperB_view
    results["ModelA_full"] = fullA_out
    results["ModelB_full"] = fullB_out
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    return results