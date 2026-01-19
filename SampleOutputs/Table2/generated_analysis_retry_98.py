def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_missing(x):
        """
        Conservative NA-code cleaning for this extract.
        Treat common GSS sentinel codes as missing.
        """
        x = to_num(x).copy()
        na_codes = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(na_codes))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 scale.
        Dislike = 4 or 5; Like/Neutral = 1,2,3; otherwise missing.
        """
        x = clean_missing(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_missing(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_count_completecase(df, items):
        """
        Sum of binary dislike indicators; require ALL components observed (complete-case DV),
        matching 'DK treated as missing and missing cases excluded' style.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def stars_from_p(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardized_ols(df, dv, xcols, model_name, ordered_terms):
        """
        OLS with standardized coefficients computed as:
            beta_j = b_j * sd(x_j) / sd(y)
        using estimation-sample SDs (ddof=0).
        Intercept is unstandardized (on DV scale).
        """
        needed = [dv] + xcols
        d = df[needed].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        # Ensure no zero-variance predictors in estimation sample
        zero_var = [c for c in xcols if d[c].std(ddof=0) == 0 or not np.isfinite(d[c].std(ddof=0))]
        if zero_var:
            # This is a real issue; write a diagnostic and then drop them so code runs.
            diag_path = f"./output/{model_name}_zero_variance_predictors.txt"
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write("Zero-variance or undefined-variance predictors in estimation sample (dropped):\n")
                for c in zero_var:
                    f.write(f"- {c}\n")
                f.write(f"\nEstimation sample size before drop: n={d.shape[0]}\n")
            x_use = [c for c in xcols if c not in zero_var]
        else:
            x_use = list(xcols)

        X = sm.add_constant(d[x_use], has_constant="add")
        y = d[dv]

        model = sm.OLS(y, X).fit()

        # Compute standardized betas from unstandardized b
        y_sd = y.std(ddof=0)
        betas = {}
        for c in x_use:
            x_sd = d[c].std(ddof=0)
            betas[c] = model.params[c] * (x_sd / y_sd) if (np.isfinite(x_sd) and x_sd > 0 and np.isfinite(y_sd) and y_sd > 0) else np.nan

        # Build output table in paper order; include all requested terms even if not estimable
        rows = []
        for term in ordered_terms:
            if term == "const":
                rows.append(
                    {
                        "term": "Constant",
                        "b_unstd": float(model.params["const"]),
                        "beta_std": np.nan,
                        "p_value": float(model.pvalues["const"]) if "const" in model.pvalues else np.nan,
                        "stars": stars_from_p(float(model.pvalues["const"])) if "const" in model.pvalues else "",
                    }
                )
            else:
                if term in model.params.index:
                    p = float(model.pvalues[term])
                    rows.append(
                        {
                            "term": term,
                            "b_unstd": float(model.params[term]),
                            "beta_std": float(betas.get(term, np.nan)),
                            "p_value": p,
                            "stars": stars_from_p(p),
                        }
                    )
                else:
                    rows.append(
                        {
                            "term": term,
                            "b_unstd": np.nan,
                            "beta_std": np.nan,
                            "p_value": np.nan,
                            "stars": "",
                        }
                    )

        tab = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(len(model.params)),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Standardized betas are computed from unstandardized coefficients as b * sd(x)/sd(y).\n")
            f.write("NOTE: p-values/stars are computed from this replication's OLS; the paper does not report SEs/p-values.\n")

        # Save a clean table (keep p/stars but clearly labeled)
        tab_to_save = tab.copy()
        tab_to_save["beta_std"] = tab_to_save["beta_std"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        tab_to_save["b_unstd"] = tab_to_save["b_unstd"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        tab_to_save["p_value"] = tab_to_save["p_value"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_to_save.to_string(index=False))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))

        return tab, fit, model

    # -------------------------
    # Load
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year==1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables
    # -------------------------
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

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half Black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_missing(df["educ"]).where(clean_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita: realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(clean_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    df["age_years"] = clean_missing(df["age"]).where(clean_missing(df["age"]).between(18, 89))

    # Race indicators (white reference; include black and other_race)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_missing(df["race"]).where(clean_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided columns. Keep as missing; do NOT proxy.
    df["hispanic"] = np.nan

    # Religion dummies (attempt to avoid dropping no_religion by not standardizing via zscore)
    # Conservative Protestant proxy: RELIG==1 (Protestant) and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_missing(df["relig"])
    denom = clean_missing(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_missing(df["region"]).where(clean_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Model spec
    # -------------------------
    # Keep Table 2 order. (We cannot estimate Hispanic with this extract; keep it in ordered output but it will be NaN.)
    ordered_predictors = [
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
    ordered_terms = ["const"] + ordered_predictors

    # Use only predictors available/meaningful in data for fitting (drop all-missing columns like hispanic)
    # This keeps the code runnable while still reporting a placeholder row for Hispanic.
    xcols_fit = []
    for c in ordered_predictors:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")
        # Drop if entirely missing
        if df[c].notna().sum() == 0:
            continue
        xcols_fit.append(c)

    results = {}

    tabA, fitA, modelA = standardized_ols(
        df,
        dv="dislike_minority_genres",
        xcols=xcols_fit,
        model_name="Table2_ModelA_dislike_minority6",
        ordered_terms=ordered_terms,
    )
    tabB, fitB, modelB = standardized_ols(
        df,
        dv="dislike_other12_genres",
        xcols=xcols_fit,
        model_name="Table2_ModelB_dislike_other12",
        ordered_terms=ordered_terms,
    )

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS with standardized betas (b*sd(x)/sd(y)).\n")
        f.write("Important: This extract has no Hispanic identifier; the Hispanic row is reported as missing (NaN).\n")
        f.write("Important: The paper reports standardized coefficients + stars but does not report SEs/p-values.\n")
        f.write("Here, stars are computed from this replication's OLS p-values (two-tailed): *<.05 **<.01 ***<.001.\n\n")
        f.write("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin) count of dislikes (0-6)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres (12 remaining genres) count of dislikes (0-12)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results