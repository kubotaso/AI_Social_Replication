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

    def clean_gss_numeric(s):
        """
        Conservative NA-code cleaning for this extract.
        Many GSS extracts use high/special codes; we remove common sentinels.
        """
        x = to_num(s).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(s):
        """
        Music items: 1..5, where 4/5 = dislike, 1/2/3 = not-dislike.
        Anything else -> missing.
        """
        x = clean_gss_numeric(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_gss_numeric(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, item_cols, require_all=True):
        """
        Sum of dislike indicators across a set of items.
        Following the summary: DK treated as missing and cases excluded -> require_all=True.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in item_cols], axis=1)
        if require_all:
            return mat.sum(axis=1, min_count=len(item_cols))
        # fallback option (not used here)
        return mat.sum(axis=1, min_count=1)

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

    def ols_with_betas(df, ycol, xcols, model_name):
        """
        Fit OLS on unstandardized variables (with intercept), then compute standardized
        coefficients post-hoc on the *final analytic sample* as:
            beta_j = b_j * SD(X_j) / SD(Y)
        Intercept is unstandardized.
        """
        needed = [ycol] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # Drop zero-variance predictors in analytic sample (but do not error; keep code runnable)
        zero_var = []
        for c in xcols:
            v = d[c].astype(float)
            if v.nunique(dropna=True) <= 1:
                zero_var.append(c)
        x_use = [c for c in xcols if c not in zero_var]

        if len(x_use) == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance after listwise deletion (n={d.shape[0]}).")

        y = d[ycol].astype(float)
        X = d[x_use].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        m = sm.OLS(y, Xc).fit()

        # Standardized betas using analytic-sample SDs, slopes only
        sd_y = y.std(ddof=0)
        betas = {}
        for c in x_use:
            sd_x = d[c].astype(float).std(ddof=0)
            if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                betas[c] = np.nan
            else:
                betas[c] = m.params[c] * (sd_x / sd_y)

        # Assemble "paper-style" table (standardized betas + stars), include all requested terms
        ordered_terms = [
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
        label_map = {
            "racism_score": "Racism score",
            "education_years": "Education",
            "hh_income_per_capita": "Household income per capita",
            "occ_prestige": "Occupational prestige",
            "female": "Female",
            "age_years": "Age",
            "black": "Black",
            "hispanic": "Hispanic",
            "other_race": "Other race",
            "cons_protestant": "Conservative Protestant",
            "no_religion": "No religion",
            "south": "Southern",
            "const": "Constant",
        }

        rows = []
        for t in ordered_terms:
            if t in x_use:
                beta = betas.get(t, np.nan)
                p = m.pvalues.get(t, np.nan)
                rows.append([label_map[t], beta, stars_from_p(p), p])
            else:
                # keep row to preserve table structure; mark dropped
                rows.append([label_map[t], np.nan, "", np.nan])

        # Constant (unstandardized)
        const_p = m.pvalues.get("const", np.nan)
        rows.append([label_map["const"], m.params.get("const", np.nan), stars_from_p(const_p), const_p])

        paper_tab = pd.DataFrame(rows, columns=["term", "coef", "stars", "p_value"])
        paper_tab = paper_tab.set_index("term")

        full_tab = pd.DataFrame(
            {
                "b_unstd": m.params,
                "std_err": m.bse,
                "t": m.tvalues,
                "p_value": m.pvalues,
            }
        )
        # Add standardized beta column for slopes present in model; NaN for const
        beta_col = []
        for idx in full_tab.index:
            if idx == "const":
                beta_col.append(np.nan)
            else:
                beta_col.append(betas.get(idx, np.nan))
        full_tab["beta_std"] = beta_col

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m.nobs),
                    "k": int(m.df_model + 1),  # incl intercept
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
                }
            ]
        )

        # Save human-readable files
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(m.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            out = paper_tab.copy()
            # show coef with stars inline, keep p_value separate for transparency
            out["coef_with_stars"] = out["coef"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}") + out["stars"].astype(str)
            out2 = out[["coef_with_stars", "p_value"]]
            f.write(out2.to_string())

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full_tab.to_string(float_format=lambda x: f"{x: .6f}"))

        return m, paper_tab, full_tab, fit, d.index

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
    # Dependent variables
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all=True)

    # -----------------------
    # Racism score (0-5)
    # -----------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

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
        raise ValueError("Missing 'educ' column.")
    educ = clean_gss_numeric(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' column required for income per capita.")
    realinc = clean_gss_numeric(df["realinc"])
    hompop = clean_gss_numeric(df["hompop"]).where(lambda s: s > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column.")
    df["occ_prestige"] = clean_gss_numeric(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' column.")
    age = clean_gss_numeric(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies
    if "race" not in df.columns:
        raise ValueError("Missing 'race' column.")
    race = clean_gss_numeric(df["race"]).where(lambda s: s.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not present in provided variables -> cannot be constructed reliably.
    # Keep as all-missing so it is excluded by listwise deletion only if included.
    # To keep the code runnable, we include it but it will be dropped from analytic sample unless present.
    if "hispanic" in df.columns:
        # If user provided an explicit hispanic variable, use it if it looks binary already.
        h = clean_gss_numeric(df["hispanic"])
        df["hispanic"] = h.where(h.isin([0, 1]))
    else:
        df["hispanic"] = np.nan

    # Religion variables
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' column required for religion controls.")
    relig = clean_gss_numeric(df["relig"])
    denom = clean_gss_numeric(df["denom"])

    # Conservative Protestant proxy (as instructed)
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()))
    df["cons_protestant"] = consprot

    # No religion
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna())
    df["no_religion"] = norelig

    # South
    if "region" not in df.columns:
        raise ValueError("Missing 'region' column.")
    region = clean_gss_numeric(df["region"]).where(lambda s: s.isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Fit models (Table 2)
    # -----------------------
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

    results = {}

    # Model A
    mA, paperA, fullA, fitA, idxA = ols_with_betas(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    # Model B
    mB, paperB, fullB, fitB, idxB = ols_with_betas(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )

    results["ModelA_table_paper_style"] = paperA
    results["ModelB_table_paper_style"] = paperB
    results["ModelA_table_full"] = fullA
    results["ModelB_table_full"] = fullB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS with standardized coefficients (betas) computed post-hoc.\n")
        f.write("Betas computed as b * SD(X)/SD(Y) using each model's final analytic sample (after listwise deletion).\n")
        f.write("Stars computed from model p-values: * p<.05, ** p<.01, *** p<.001 (two-tailed).\n")
        f.write("\nNOTES:\n")
        f.write("- If 'hispanic' is not present in the input extract, it is set to missing and will reduce N via listwise deletion.\n")
        f.write("- If you have a proper Hispanic identifier, include it as a 0/1 column named 'hispanic' in the CSV.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results