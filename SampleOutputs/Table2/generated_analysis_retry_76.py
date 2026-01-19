def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # ---- helpers ----
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def is_missing_code(x):
        # Conservative GSS-style missing codes. Keep this minimal to avoid over-deletion.
        # Handles common: 8/9, 98/99, 998/999, 9998/9999 plus negatives.
        return (
            pd.isna(x)
            | (x < 0)
            | x.isin([8, 9, 98, 99, 998, 999, 9998, 9999])
        )

    def clean_numeric(series):
        x = to_num(series).copy()
        x = x.mask(is_missing_code(x))
        return x

    def likert_dislike(series):
        # 1-5 where 4/5 are dislikes; others 0. Missing if not 1-5 or missing-code.
        x = clean_numeric(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_numeric(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_complete_case(df_, items):
        # Paper: DK treated as missing; cases with missing excluded.
        # Implement as COMPLETE CASE across all items in the count.
        mat = pd.concat([likert_dislike(df_[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardize_betas_from_unstd_fit(fit, X, y):
        # Standardized beta = b * sd(x)/sd(y), intercept excluded.
        y_sd = np.std(y, ddof=0)
        betas = {}
        for name in X.columns:
            if name == "const":
                continue
            x_sd = np.std(X[name], ddof=0)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[name] = np.nan
            else:
                betas[name] = fit.params.get(name, np.nan) * (x_sd / y_sd)
        return pd.Series(betas, dtype="float64")

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

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ---- required base ----
    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain YEAR and ID (case-insensitive).")

    df["year"] = clean_numeric(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ---- build DVs (complete-case within each DV only) ----
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # ---- racism score (0-5), complete-case across its 5 items ----
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])    # object to >half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])    # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])   # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])   # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])   # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ---- predictors ----
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    df["education_years"] = clean_numeric(df["educ"]).where(lambda x: x.between(0, 20))

    # HH income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_numeric(df["realinc"])
    hompop = clean_numeric(df["hompop"]).where(lambda x: x > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_numeric(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    df["age_years"] = clean_numeric(df["age"]).where(lambda x: x.between(18, 89))

    # Race: black/other, white reference
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_numeric(df["race"]).where(lambda x: x.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables -> cannot construct faithfully.
    # We keep it as missing and exclude from models to avoid wiping N.
    df["hispanic"] = np.nan

    # Religion: Conservative Protestant proxy and No religion
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_numeric(df["relig"])
    denom = clean_numeric(df["denom"])

    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    )
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_numeric(df["region"]).where(lambda x: x.isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Weighting: GSS weight not provided in available variables. Proceed unweighted.
    weight_note = "No weight variable provided in dataset extract; models estimated unweighted."

    # ---- model runner ----
    def run_model(dv_col, model_name, xcols, include_hispanic=False):
        cols = [dv_col] + xcols
        d = df[cols].copy()

        # Listwise deletion per model on DV and included predictors
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < len(xcols) + 2:
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv_col].astype(float)

        X = d[xcols].astype(float).copy()
        X = sm.add_constant(X, has_constant="add")

        fit = sm.OLS(y, X).fit()

        # Standardized betas computed from unstandardized fit
        betas = standardize_betas_from_unstd_fit(fit, X, y)

        # Stars from p-values (replication); table output will NOT show p-values/SEs.
        pvals = fit.pvalues.drop(labels=["const"], errors="ignore")
        beta_star = []
        for term in betas.index:
            beta_star.append(f"{betas.loc[term]: .3f}{stars_from_p(pvals.get(term, np.nan))}")
        beta_star = pd.Series(beta_star, index=betas.index, name="beta_std")

        # Assemble table in paper-like order
        table_order = [
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
        # Keep only those actually estimated (hispanic will be absent)
        ordered_terms = [t for t in table_order if t in beta_star.index]

        out = pd.DataFrame(
            {
                "beta_std": beta_star.loc[ordered_terms].values,
            },
            index=ordered_terms
        )
        out.index.name = "term"

        fit_stats = pd.DataFrame([{
            "model": model_name,
            "n": int(fit.nobs),
            "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "constant_b": float(fit.params.get("const", np.nan)),
        }])

        # Write files
        write_text(f"./output/{model_name}_summary.txt", fit.summary().as_text() + "\n\n" + weight_note + "\n")
        write_text(f"./output/{model_name}_table.txt", out.to_string())
        write_text(f"./output/{model_name}_fit.txt", fit_stats.to_string(index=False))

        return out, fit_stats, fit

    # Predictors actually available for faithful estimation (exclude hispanic because missing)
    xcols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    tabA, fitA, fitobjA = run_model("dislike_minority_genres", "Table2_ModelA_dislike_minority6", xcols)
    tabB, fitB, fitobjB = run_model("dislike_other12_genres", "Table2_ModelB_dislike_other12", xcols)

    # Overview
    overview = []
    overview.append("Table 2 replication attempt (Bryson 1996; GSS 1993)")
    overview.append("OLS estimated on unstandardized variables; standardized betas computed as b * SD(X)/SD(Y).")
    overview.append("Stars based on model p-values (two-tailed): * p<.05, ** p<.01, *** p<.001.")
    overview.append(weight_note)
    overview.append("")
    overview.append("Notes:")
    overview.append("- DV construction uses complete-case across the genre items in each DV, treating DK/NA codes as missing.")
    overview.append("- Hispanic dummy cannot be constructed from the provided variable list; therefore it is excluded (would otherwise destroy N).")
    overview.append("")
    overview.append("Model A: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)")
    overview.append(fitA.to_string(index=False))
    overview.append("")
    overview.append("Model B: dislike_other12_genres (Big band, Bluegrass, Country, Musicals, Classical, Folk, Mood/easy, New age, Opera, Contemporary rock, Oldies, Heavy metal)")
    overview.append(fitB.to_string(index=False))
    overview_text = "\n".join(overview) + "\n"
    write_text("./output/Table2_overview.txt", overview_text)

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }