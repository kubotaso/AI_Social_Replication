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
        Conservative GSS-style missing cleaning:
        - Converts to numeric
        - Treats common sentinel codes as missing
        - Leaves other values untouched
        """
        s = to_num(x).copy()
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

    def likert_dislike_indicator(series):
        """
        Music battery: 1..5; dislike = 4 or 5; like/neutral = 1..3.
        Anything else treated as missing.
        """
        x = clean_missing(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_recode(series, true_codes, false_codes):
        x = clean_missing(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        # require ALL items observed (DK treated as missing, cases excluded)
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

    def standardized_beta(b_unstd, x_sd, y_sd):
        if pd.isna(b_unstd) or pd.isna(x_sd) or pd.isna(y_sd) or y_sd == 0 or x_sd == 0:
            return np.nan
        return b_unstd * (x_sd / y_sd)

    def fit_table2_model(df, dv, x_terms, model_name):
        """
        Fits OLS on unstandardized variables (with intercept), then computes
        standardized betas post-hoc for slopes only: beta = b * sd(x)/sd(y)
        Stars based on two-tailed p-values from the OLS fit.
        """

        needed = [dv] + x_terms
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(x_terms) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # Check zero variance predictors on analytic sample; drop safely (do not error)
        dropped_zero_var = []
        keep_terms = []
        for c in x_terms:
            sd = d[c].std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                dropped_zero_var.append(c)
            else:
                keep_terms.append(c)

        if len(keep_terms) == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance after listwise deletion.")

        y = d[dv].astype(float)
        X = d[keep_terms].astype(float)
        X = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X).fit()

        # compute standardized betas for slopes
        y_sd = y.std(ddof=0)
        x_sds = d[keep_terms].std(ddof=0)

        rows = []
        # slopes in requested paper order, including those dropped (as NA) to keep alignment
        for term in x_terms:
            if term in keep_terms:
                b = model.params.get(term, np.nan)
                se = model.bse.get(term, np.nan)
                t = model.tvalues.get(term, np.nan)
                p = model.pvalues.get(term, np.nan)
                beta = standardized_beta(b, float(x_sds[term]), float(y_sd))
                rows.append(
                    {
                        "term": term,
                        "b_unstd": b,
                        "std_err": se,
                        "t": t,
                        "p_value": p,
                        "beta_std": beta,
                        "stars": star_from_p(p),
                        "dropped": "",
                    }
                )
            else:
                rows.append(
                    {
                        "term": term,
                        "b_unstd": np.nan,
                        "std_err": np.nan,
                        "t": np.nan,
                        "p_value": np.nan,
                        "beta_std": np.nan,
                        "stars": "",
                        "dropped": "zero_variance",
                    }
                )

        # intercept row (constant is unstandardized; beta not meaningful)
        b0 = model.params.get("const", np.nan)
        p0 = model.pvalues.get("const", np.nan)
        rows.append(
            {
                "term": "const",
                "b_unstd": b0,
                "std_err": model.bse.get("const", np.nan),
                "t": model.tvalues.get("const", np.nan),
                "p_value": p0,
                "beta_std": np.nan,
                "stars": star_from_p(p0),
                "dropped": "",
            }
        )

        full_table = pd.DataFrame(rows)

        paper_style = full_table.copy()
        # Table 2 reports standardized coefficients; show beta for slopes, intercept as unstd b
        paper_style["coef_table2"] = np.where(
            paper_style["term"].eq("const"),
            paper_style["b_unstd"],
            paper_style["beta_std"],
        )
        paper_style["coef_with_stars"] = paper_style["coef_table2"].map(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"
        ) + paper_style["stars"].fillna("")

        # Keep only relevant columns for "paper-like" display
        paper_style = paper_style[["term", "coef_with_stars", "coef_table2", "dropped"]]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),  # incl intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped_zero_var) if dropped_zero_var else "",
                }
            ]
        )

        return model, paper_style, full_table, fit, d.index

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to year 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Construct DVs
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
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_recode(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_recode(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_recode(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_recode(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_recode(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column: educ")
    educ = clean_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing income component: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(clean_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column: prestg80")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing SEX column: sex")
    df["female"] = binary_recode(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing AGE column: age")
    age = clean_missing(df["age"])
    # allow full adult range in GSS; keep 18-89 as previously
    df["age_years"] = age.where(age.between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing RACE column: race")
    race = clean_missing(df["race"]).where(clean_missing(df["race"]).isin([1, 2, 3]))

    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC as an approximate flag if available.
    # (The provided extract lacks a proper Hispanic identifier; this is the best implementable proxy here.)
    if "ethnic" in df.columns:
        eth = clean_missing(df["ethnic"])
        # Common GSS ETHNIC codes: 1=Mexican, 2=Puerto Rican, 3=Other Spanish.
        # Treat these as Hispanic; all other valid codes as non-Hispanic.
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[eth.isin([1, 2, 3])] = 1.0
        hisp.loc[eth.notna() & ~eth.isin([1, 2, 3])] = 0.0
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan

    # Religion indicators
    if "relig" not in df.columns:
        raise ValueError("Missing RELIG column: relig")
    relig = clean_missing(df["relig"])

    # No religion: RELIG==4 (none)
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7} if denom exists; else missing
    if "denom" in df.columns:
        denom = clean_missing(df["denom"])
        consprot = pd.Series(np.nan, index=df.index, dtype="float64")
        ok = relig.notna() & denom.notna()
        consprot.loc[ok] = ((relig.loc[ok] == 1) & (denom.loc[ok].isin([1, 6, 7]))).astype(float)
        df["cons_protestant"] = consprot
    else:
        df["cons_protestant"] = np.nan

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column: region")
    region = clean_missing(df["region"]).where(clean_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Fit models (Table 2)
    # -------------------------
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
            raise ValueError(f"Constructed predictor missing: {c}")

    mA, paperA, fullA, fitA, idxA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, idxB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # -------------------------
    # Diagnostics: verify variation in key dummies within each analytic sample
    # -------------------------
    def diag_freq(df, idx, cols):
        out = {}
        sub = df.loc[idx, cols].copy()
        for c in cols:
            vc = sub[c].value_counts(dropna=False).sort_index()
            out[c] = vc
        return out

    diag_cols = ["black", "hispanic", "other_race", "no_religion", "cons_protestant", "south", "female"]
    diagA = diag_freq(df, idxA, diag_cols)
    diagB = diag_freq(df, idxB, diag_cols)

    # -------------------------
    # Save outputs
    # -------------------------
    def write_table_txt(path, df_table, title=None):
        with open(path, "w", encoding="utf-8") as f:
            if title:
                f.write(title.strip() + "\n\n")
            f.write(df_table.to_string(index=False))
            f.write("\n")

    # Paper-style tables (standardized betas for slopes; intercept shown as unstd)
    write_table_txt("./output/Table2_ModelA_paper_style.txt", paperA, "Table 2 Model A (paper-style): standardized coefficients (slopes) + stars; intercept unstandardized")
    write_table_txt("./output/Table2_ModelB_paper_style.txt", paperB, "Table 2 Model B (paper-style): standardized coefficients (slopes) + stars; intercept unstandardized")

    # Full tables (unstandardized + derived standardized betas)
    write_table_txt("./output/Table2_ModelA_full_table.txt", fullA, "Model A full table: unstandardized b, SE, p, and derived standardized beta")
    write_table_txt("./output/Table2_ModelB_full_table.txt", fullB, "Model B full table: unstandardized b, SE, p, and derived standardized beta")

    # Fit summaries
    write_table_txt("./output/Table2_ModelA_fit.txt", fitA, "Model A fit statistics")
    write_table_txt("./output/Table2_ModelB_fit.txt", fitB, "Model B fit statistics")

    # Statsmodels summaries
    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(mA.summary().as_text())
        f.write("\n")
    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(mB.summary().as_text())
        f.write("\n")

    # Diagnostic frequencies
    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("Diagnostics: frequency tables in analytic samples (post listwise deletion)\n\n")
        f.write("Model A sample:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        for c, vc in diagA.items():
            f.write(f"{c}:\n{vc.to_string()}\n\n")
        f.write("\nModel B sample:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        for c, vc in diagB.items():
            f.write(f"{c}:\n{vc.to_string()}\n\n")

    results = {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }
    return results