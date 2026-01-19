def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative GSS-style missing handling.
        The provided extract already uses blanks for many missings, but we also
        treat common sentinel codes as missing.
        """
        x = to_num(x).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5; dislike if 4 or 5; like/neutral if 1-3.
        Anything outside 1-5 or sentinel-coded => missing.
        """
        x = clean_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Count of dislikes across items. Mirror "DK treated as missing and missing cases excluded"
        by requiring complete data across all items for that DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardize_series(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_unstd_and_betas(df_model, y_col, x_cols, model_name):
        """
        Fit OLS on unstandardized DV so intercept is interpretable (and comparable to paper format),
        then compute standardized betas for predictors using beta = b * sd(x) / sd(y).
        """
        d = df_model[[y_col] + x_cols].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = d[y_col].astype(float)
        X = d[x_cols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        m = sm.OLS(y, Xc).fit()

        # standardized betas from unstandardized regression
        sd_y = y.std(ddof=0)
        betas = {}
        for c in x_cols:
            sd_x = X[c].std(ddof=0)
            if (not np.isfinite(sd_x)) or sd_x == 0 or (not np.isfinite(sd_y)) or sd_y == 0:
                betas[c] = np.nan
            else:
                betas[c] = m.params[c] * (sd_x / sd_y)

        # constant is shown as unstandardized; standardized beta for constant not meaningful
        betas_const = np.nan

        tab = pd.DataFrame(
            {
                "b_unstd": m.params,
                "std_err": m.bse,
                "t": m.tvalues,
                "p_value": m.pvalues,
                "beta_std": pd.Series({"const": betas_const, **betas}),
            }
        )
        tab.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m.nobs),
                    "k": int(m.df_model + 1),  # includes intercept
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                }
            ]
        )

        return m, tab, fit

    def add_sig_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    # -------------------------
    # Load and filter to 1993
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Missing column: year")
    if "id" not in df.columns:
        raise ValueError("Missing column: id")

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
    # Racism score 0-5 (complete-case across 5 items)
    # -------------------------
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

    # -------------------------
    # Controls
    # -------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column for income per capita: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(18, 89))
    df["age_years"] = age

    # South
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Religion dummies
    if "relig" not in df.columns:
        raise ValueError("Missing column: relig")
    if "denom" not in df.columns:
        raise ValueError("Missing column: denom")

    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])

    # Conservative Protestant proxy per mapping
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()))
    df["cons_protestant"] = consprot

    # No religion
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna())
    df["no_religion"] = norelig

    # -------------------------
    # Race/ethnicity dummies (must be mutually exclusive: White non-Hispanic reference)
    # We do not have a true Hispanic flag in the provided variables. To satisfy the
    # required model structure without proxying ETHNIC, we create:
    #   - hispanic: all missing
    #   - black and other_race from RACE but set to missing if hispanic is missing (unknown)
    #
    # This preserves the correct *specification* and ensures Hispanic appears as a row.
    # If a valid Hispanic identifier is later added to the dataset, this code will use it.
    # -------------------------
    if "race" not in df.columns:
        raise ValueError("Missing column: race")

    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))

    # If a proper hispanic flag exists, use it; otherwise keep missing (do not proxy using ethnic)
    hispanic = None
    for cand in ["hispanic", "hispan", "hisp", "ethnic_hisp", "hispflag"]:
        if cand in df.columns:
            hispanic = binary_from_codes(df[cand], true_codes=[1], false_codes=[0, 2])
            break
    if hispanic is None:
        hispanic = pd.Series(np.nan, index=df.index, dtype="float64")
    df["hispanic"] = hispanic

    # Construct Black / Other race with exclusivity: if Hispanic==1, both should be 0;
    # if Hispanic==0, use RACE; if Hispanic missing, set missing (cannot classify reference group).
    df["black"] = np.nan
    df["other_race"] = np.nan

    known_hisp = df["hispanic"].isin([0.0, 1.0])
    idx_known = df.index[known_hisp]

    # Hispanic==1 => black=0, other=0 (ethnicity takes precedence)
    idx_h1 = df.index[df["hispanic"] == 1.0]
    df.loc[idx_h1, "black"] = 0.0
    df.loc[idx_h1, "other_race"] = 0.0

    # Hispanic==0 => use race
    idx_h0 = df.index[df["hispanic"] == 0.0]
    df.loc[idx_h0, "black"] = np.where(race.loc[idx_h0].isna(), np.nan, (race.loc[idx_h0] == 2).astype(float))
    df.loc[idx_h0, "other_race"] = np.where(race.loc[idx_h0].isna(), np.nan, (race.loc[idx_h0] == 3).astype(float))

    # -------------------------
    # Model specification (Table 2 RHS)
    # -------------------------
    x_cols = [
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
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # -------------------------
    # Fit two models (listwise deletion per model)
    # -------------------------
    results = {}

    def run_one(y_col, model_name):
        m, tab, fit = fit_unstd_and_betas(df, y_col, x_cols, model_name)

        # "Paper-style" display: standardized betas (predictors) + stars, plus unstd constant
        rows = []
        for term in ["racism_score", "education_years", "hh_income_per_capita", "occ_prestige",
                     "female", "age_years", "black", "hispanic", "other_race",
                     "cons_protestant", "no_religion", "south"]:
            beta = tab.loc[term, "beta_std"] if term in tab.index else np.nan
            p = tab.loc[term, "p_value"] if term in tab.index else np.nan
            rows.append({"term": term, "beta_std": beta, "sig": add_sig_stars(p)})

        const_b = tab.loc["const", "b_unstd"] if "const" in tab.index else np.nan
        const_p = tab.loc["const", "p_value"] if "const" in tab.index else np.nan
        paper_tbl = pd.DataFrame(rows).set_index("term")
        paper_tbl.loc["constant (b)"] = {"beta_std": const_b, "sig": add_sig_stars(const_p)}

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(m.summary().as_text())
            f.write("\n\nComputed standardized betas for predictors: beta = b * sd(x)/sd(y)\n")
            f.write("Stars computed from two-tailed p-values of the unstandardized OLS fit.\n")

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Paper-style table: standardized coefficients (betas) for predictors; constant shown as unstandardized b.\n")
            f.write(paper_tbl.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write("Full table: unstandardized OLS coefficients + SE/t/p, plus computed standardized betas (beta_std).\n")
            f.write(tab.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))

        return paper_tbl, tab, fit

    paperA, fullA, fitA = run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    paperB, fullB, fitB = run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS extract):\n")
        f.write("- Two OLS models with unstandardized DV (dislike counts) and standardized betas for predictors.\n")
        f.write("- Stars are computed from p-values of the unstandardized OLS fit (Table 2 itself does not print SEs).\n")
        f.write("- Race/ethnicity: requires Hispanic; if not present in the dataset, Hispanic is all-missing and will reduce N.\n\n")
        f.write("Model A: DV = count disliked among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B: DV = count disliked among remaining 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results["ModelA_table_paper_style"] = paperA
    results["ModelB_table_paper_style"] = paperB
    results["ModelA_table_full"] = fullA
    results["ModelB_table_full"] = fullB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    return results