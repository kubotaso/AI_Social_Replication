def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_missing(series):
        """
        Conservative missing handling for this extract:
        - Coerce non-numeric to NaN
        - Treat common GSS sentinel codes as NaN
        """
        x = to_num(series).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(series):
        """
        Music taste items are 1-5; dislike if 4 or 5.
        Missing if not in 1..5 after cleaning.
        """
        x = clean_missing(series)
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

    def build_count_listwise(df, item_cols):
        """
        Sum of dislike indicators across items, requiring complete responses
        on all component items (listwise for DV construction), matching the
        "DK treated as missing and missing cases excluded" rule.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in item_cols], axis=1)
        # require all items non-missing
        ok = mat.notna().all(axis=1)
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        out.loc[ok] = mat.loc[ok].sum(axis=1)
        return out

    def standardize(series, ddof=1):
        x = to_num(series)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        return (x - mu) / sd

    def ols_with_standardized_betas(df_model, dv, xcols, model_name):
        """
        Fit unstandardized OLS with intercept; compute standardized betas post-hoc:
            beta_j = b_j * SD(x_j) / SD(y)
        using analytic sample SDs (ddof=1).
        Return:
          - fitted model
          - "paper style" table: beta + stars (plus Constant as unstandardized)
          - "full" table: b, beta, se, t, p
          - fit dict
          - analytic sample index
        """
        needed = [dv] + xcols
        d = df_model[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        y = d[dv].astype(float)
        X = d[xcols].astype(float)

        # Guard against zero-variance predictors in analytic sample
        zero_var = [c for c in xcols if X[c].std(ddof=1) == 0 or not np.isfinite(X[c].std(ddof=1))]
        if zero_var:
            raise ValueError(f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. Check coding/sample restrictions.")

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas post-hoc (slopes only)
        y_sd = y.std(ddof=1)
        betas = {}
        for c in xcols:
            betas[c] = model.params[c] * (X[c].std(ddof=1) / y_sd)

        # Significance stars (two-tailed)
        def stars(p):
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        # Full table (computed)
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        # attach standardized betas to matching slope terms; constant is NaN
        full["beta_std"] = np.nan
        for c in xcols:
            full.loc[c, "beta_std"] = betas[c]

        # Paper-style table: standardized betas for slopes; Constant is unstandardized
        paper = pd.DataFrame(index=["Racism score", "Education", "Household income per capita", "Occupational prestige",
                                    "Female", "Age", "Black", "Hispanic", "Other race",
                                    "Conservative Protestant", "No religion", "Southern", "Constant"],
                             columns=["coef", "stars"], dtype=object)

        # Map internal names to paper labels
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
        }

        for c in xcols:
            lab = label_map.get(c, c)
            paper.loc[lab, "coef"] = float(betas[c])
            paper.loc[lab, "stars"] = stars(float(model.pvalues[c]))

        paper.loc["Constant", "coef"] = float(model.params["const"])
        paper.loc["Constant", "stars"] = stars(float(model.pvalues["const"]))

        fit = {
            "model": model_name,
            "n": int(model.nobs),
            "k": int(model.df_model + 1),  # includes intercept
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        }

        return model, paper, full, fit, d.index

    # -------------------------
    # Load data and filter year
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must include 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables (two counts)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_count_listwise(df, minority_items)
    df["dislike_other12_genres"] = build_count_listwise(df, other12_items)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])     # object to >half Black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])     # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])    # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])    # deny education chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])    # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    # require all five
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # RHS controls
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' (EDUC) column.")
    educ = clean_missing(df["educ"]).where(clean_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required income component: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(clean_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' (PRESTG80) column.")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' (SEX) column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' (AGE) column.")
    df["age_years"] = clean_missing(df["age"]).where(clean_missing(df["age"]).between(18, 89))

    # Race dummies from RACE (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing 'race' (RACE) column.")
    race = clean_missing(df["race"]).where(clean_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in this extract; include as all-zeros so model can run and term is present.
    # (This is a limitation of the provided data; it will not reproduce the paper exactly.)
    df["hispanic"] = 0.0

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion field: {c}")
    relig = clean_missing(df["relig"])
    denom = clean_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing 'region' (REGION) column.")
    region = clean_missing(df["region"]).where(clean_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -------------------------
    # Fit models (Table 2)
    # -------------------------
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

    def save_outputs(model, paper, full, fit, model_name):
        # Human-readable model summary (unstandardized OLS)
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            for k, v in fit.items():
                f.write(f"{k}: {v}\n")

        # Paper-style table: betas + stars only (and unstandardized Constant)
        paper_out = paper.copy()
        # format coefficient column to 3 decimals like typical tables
        def fmt_coef(v):
            if pd.isna(v):
                return ""
            try:
                return f"{float(v): .3f}"
            except Exception:
                return str(v)

        lines = []
        lines.append(f"{model_name}: Standardized coefficients (betas) with stars; Constant unstandardized")
        lines.append("Significance (two-tailed): * p<.05, ** p<.01, *** p<.001")
        lines.append("")
        header = f"{'Variable':<28} {'Coef':>8} {'':<3}"
        lines.append(header)
        lines.append("-" * len(header))
        for idx in paper_out.index:
            coef = fmt_coef(paper_out.loc[idx, "coef"])
            st = paper_out.loc[idx, "stars"]
            if st is None or (isinstance(st, float) and np.isnan(st)):
                st = ""
            lines.append(f"{idx:<28} {coef:>8} {st}")
        lines.append("")
        lines.append(f"N={fit['n']}  R2={fit['r2']:.3f}  AdjR2={fit['adj_r2']:.3f}")
        with open(f"./output/{model_name}_table_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Full computed table (not in paper; useful for debugging)
        full_out = full.copy()
        full_out.index.name = "term"
        full_out.to_string(
            open(f"./output/{model_name}_table_full_debug.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x: .6f}",
        )

    mA, paperA, fullA, fitA, idxA = ols_with_standardized_betas(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    save_outputs(mA, paperA, fullA, fitA, "Table2_ModelA_dislike_minority6")

    mB, paperB, fullB, fitB, idxB = ols_with_standardized_betas(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )
    save_outputs(mB, paperB, fullB, fitB, "Table2_ModelB_dislike_other12")

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication from provided GSS 1993 extract\n")
        f.write("Two OLS models; standardized betas computed post-hoc from unstandardized OLS slopes using analytic-sample SDs.\n")
        f.write("Important limitation: this extract contains no Hispanic identifier; 'hispanic' is set to 0 for all cases.\n")
        f.write("Tables saved:\n")
        f.write(" - Table2_ModelA_dislike_minority6_table_table2_style.txt\n")
        f.write(" - Table2_ModelB_dislike_other12_table_table2_style.txt\n")
        f.write("Debug tables with b/se/t/p saved as *_table_full_debug.txt (not in the published table).\n\n")
        f.write(f"Model A: N={fitA['n']} R2={fitA['r2']:.3f} AdjR2={fitA['adj_r2']:.3f}\n")
        f.write(f"Model B: N={fitB['n']} R2={fitB['r2']:.3f} AdjR2={fitB['adj_r2']:.3f}\n")

    results["ModelA_table2_style"] = paperA
    results["ModelB_table2_style"] = paperB
    results["ModelA_full_debug"] = fullA
    results["ModelB_full_debug"] = fullB
    results["ModelA_fit"] = pd.DataFrame([fitA])
    results["ModelB_fit"] = pd.DataFrame([fitB])

    return results