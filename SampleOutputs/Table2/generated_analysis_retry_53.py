def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative NA handling for this extract:
        - Coerce to numeric
        - Treat common GSS-style sentinels as missing
        """
        s = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        s = s.mask(s.isin(sentinels))
        return s

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 (like very much .. dislike very much).
        Dislike = 4 or 5; Like/neutral = 1,2,3; else missing.
        """
        v = clean_na_codes(item).where(lambda z: z.between(1, 5))
        out = pd.Series(np.nan, index=v.index, dtype="float64")
        out.loc[v.isin([1, 2, 3])] = 0.0
        out.loc[v.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        v = clean_na_codes(series)
        out = pd.Series(np.nan, index=v.index, dtype="float64")
        out.loc[v.isin(false_codes)] = 0.0
        out.loc[v.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_complete=True):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_complete:
            # Complete-case across the items (strict interpretation of "DK treated as missing; missing cases excluded")
            return mat.sum(axis=1, min_count=len(items))
        # Alternative (not used): allow partials
        return mat.sum(axis=1, min_count=1)

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

    def ols_standardized_betas(df, y_col, x_cols, model_name):
        """
        Fit OLS on original scales, then compute standardized betas as:
            beta_j = b_j * SD(X_j) / SD(Y)
        using the final analytic sample (after listwise deletion).
        Return:
          - paper_style table: beta + stars (NO SE/t/p columns)
          - full table: b, se, t, p, beta (for internal/diagnostic use; not "Table 2 format")
          - fit table: N, R2, adj R2
          - model object
          - analytic sample index
        """
        needed = [y_col] + x_cols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        if d.shape[0] < (len(x_cols) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # zero-variance predictors in analytic sample -> cannot estimate standardized beta
        zero_var = []
        for c in x_cols:
            v = d[c].astype(float)
            if not np.isfinite(v).all():
                zero_var.append(c)
            else:
                if float(v.std(ddof=0)) == 0.0:
                    zero_var.append(c)
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Check coding/sample restrictions."
            )

        y = d[y_col].astype(float)
        X = d[x_cols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        sd_y = float(y.std(ddof=0))
        if not np.isfinite(sd_y) or sd_y == 0.0:
            raise ValueError(f"{model_name}: dependent variable has zero/invalid SD in analytic sample.")

        betas = {}
        for c in x_cols:
            sd_x = float(d[c].astype(float).std(ddof=0))
            betas[c] = float(model.params[c]) * (sd_x / sd_y)

        # Tables
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        # add beta_std for predictors; constant has no standardized beta
        full["beta_std"] = np.nan
        for c in x_cols:
            full.loc[c, "beta_std"] = betas[c]

        # paper-style: ONLY beta + stars (and constant reported separately as unstandardized)
        paper_rows = []
        for c in x_cols:
            p = float(model.pvalues[c]) if c in model.pvalues.index else np.nan
            paper_rows.append(
                {
                    "term": c,
                    "beta": betas[c],
                    "stars": star_from_p(p),
                }
            )
        # constant (unstandardized)
        p_const = float(model.pvalues["const"]) if "const" in model.pvalues.index else np.nan
        paper_rows.append(
            {
                "term": "const",
                "beta": float(model.params["const"]),
                "stars": star_from_p(p_const),
            }
        )
        paper = pd.DataFrame(paper_rows).set_index("term")

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(round(model.nobs)),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Table-2-style output is standardized betas + stars only; SE/t/p are not printed there.\n")

        def _fmt_df(df_to_print, path, float_fmt="{: .6f}"):
            with open(path, "w", encoding="utf-8") as f:
                if isinstance(df_to_print, pd.DataFrame):
                    f.write(df_to_print.to_string(float_format=lambda x: float_fmt.format(x)))
                else:
                    f.write(str(df_to_print))

        _fmt_df(paper, f"./output/{model_name}_paper_style.txt")
        _fmt_df(full, f"./output/{model_name}_full_diagnostics.txt")
        _fmt_df(fit, f"./output/{model_name}_fit.txt", float_fmt="{: .6f}")

        return model, paper, full, fit, d.index

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic checks
    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_complete=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_complete=True)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
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

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC/HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(lambda z: z > 0)
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
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (white reference): black, other_race
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_na_codes(df["race"]).where(lambda z: z.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy:
    # Not available in this extract. Create as 0 (NOT missing) so code runs and does not drop cases.
    # This cannot replicate the paper's Hispanic effect without a true Hispanic identifier.
    df["hispanic"] = 0.0

    # Conservative Protestant proxy (RELIG==1 and DENOM in {1,6,7})
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion (RELIG==4)
    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern (REGION==3)
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_na_codes(df["region"]).where(lambda z: z.isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Models (Table 2)
    # -----------------------------
    # Keep Table 2 order (plus constant added by fitting function)
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

    # Ensure columns exist
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # Label mapping for output
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

    mA, paperA, fullA, fitA, idxA = ols_standardized_betas(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, idxB = ols_standardized_betas(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )

    # Reindex paper tables into the paper's row order with labels
    paper_order = [
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
        "const",
    ]

    def relabel_and_order(paper):
        out = paper.reindex(paper_order).copy()
        out.index = [label_map.get(i, i) for i in out.index]
        # combine beta + stars into one column for human-readable table-2 style
        out["beta_with_stars"] = out.apply(
            lambda r: ("" if pd.isna(r["beta"]) else f"{r['beta']:.3f}") + str(r["stars"]),
            axis=1,
        )
        return out[["beta", "stars", "beta_with_stars"]]

    paperA2 = relabel_and_order(paperA)
    paperB2 = relabel_and_order(paperB)

    # Save final Table-2-style combined output
    with open("./output/Table2_like_output.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication-style output (computed from microdata): standardized betas + significance stars\n")
        f.write("Stars computed from OLS p-values (two-tailed): * p<.05, ** p<.01, *** p<.001\n")
        f.write("NOTE: This dataset extract lacks a true Hispanic identifier; 'Hispanic' is set to 0 for all cases here.\n")
        f.write("      This affects comparability for the Hispanic coefficient and potentially other estimates.\n\n")

        f.write("MODEL A DV: Dislike count among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(paperA2.to_string())
        f.write("\n\nFit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nMODEL B DV: Dislike count among the other 12 genres\n")
        f.write(paperB2.to_string())
        f.write("\n\nFit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    # Also save the two clean "paper-style" tables separately
    paperA2.to_string(open("./output/Table2_ModelA_paper_style.txt", "w", encoding="utf-8"))
    paperB2.to_string(open("./output/Table2_ModelB_paper_style.txt", "w", encoding="utf-8"))

    return {
        "ModelA_paper_style": paperA2,
        "ModelB_paper_style": paperB2,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "ModelA_full_diagnostics": fullA,
        "ModelB_full_diagnostics": fullB,
    }