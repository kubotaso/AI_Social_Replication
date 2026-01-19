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
        Conservative missing-code handling for common GSS extracts.
        Treat typical sentinel codes as missing.
        """
        x = to_num(x).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(sentinel)))
        return x

    def likert_dislike_indicator(series):
        """
        Music taste items: 1-5; 4/5 => dislike. 1/2/3 => not dislike.
        Non-1..5 or NA-coded => missing.
        """
        x = clean_na_codes(series)
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

    def build_dislike_count(df, cols, require_all_items=True):
        inds = []
        for c in cols:
            inds.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(inds, axis=1)
        if require_all_items:
            return mat.sum(axis=1, min_count=len(cols))
        # fallback: allow partial but require at least half items answered
        return mat.sum(axis=1, min_count=max(1, int(np.ceil(len(cols) / 2))))

    def ols_with_betas_and_p(df_model, y_col, x_cols):
        """
        Fit unweighted OLS with intercept on original scales.
        Compute standardized betas using sample SDs from the estimation sample:
            beta_j = b_j * sd(x_j) / sd(y)
        Return:
          - model
          - table with beta + stars, plus constant (unstandardized) as in paper
          - fit stats
        """
        needed = [y_col] + x_cols
        d = df_model[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        # If any predictor is constant, drop it but record (do not crash)
        zero_var = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                zero_var.append(c)
        x_use = [c for c in x_cols if c not in zero_var]

        y = d[y_col].astype(float)
        X = d[x_use].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (exclude intercept)
        sd_y = float(y.std(ddof=1))
        betas = {}
        for c in x_use:
            sd_x = float(d[c].std(ddof=1))
            if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(model.params[c]) * sd_x / sd_y

        def stars(p):
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        # Build paper-style table (standardized betas + stars; constant unstandardized)
        rows = []
        for c in x_cols:
            if c in zero_var:
                rows.append((c, np.nan, ""))  # included in spec but not estimable in this sample
            else:
                rows.append((c, betas.get(c, np.nan), stars(float(model.pvalues.get(c, np.nan)))))

        const = float(model.params["const"])
        const_star = stars(float(model.pvalues["const"]))
        rows.append(("constant", const, const_star))

        tab = pd.DataFrame(rows, columns=["term", "coef", "stars"]).set_index("term")
        tab["coef_with_stars"] = tab["coef"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + tab["stars"]

        fit = {
            "n": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
        }
        return model, tab, fit, d

    # -----------------------
    # Load & filter
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Dependent variables (counts of dislikes)
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    # Require all component items observed (closest to "DK treated as missing and missing cases excluded")
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_items=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_items=True)

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

    rac_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = rac_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls
    # -----------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing educ column")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing realinc and/or hompop column")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race indicators (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing race column")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables. Keep as missing, but do NOT include in model.
    # (Including it would drop all rows. This is faithful to mapping instruction.)
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing relig and/or denom column")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()))
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -----------------------
    # Fit models (Table 2 style: standardized betas + stars; constant unstandardized)
    # Note: Hispanic not included due to missing mapping in provided data.
    # -----------------------
    x_cols = [
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

    # Fit Model A
    mA, tabA, fitA, dA = ols_with_betas_and_p(
        df, "dislike_minority_genres", x_cols
    )

    # Fit Model B
    mB, tabB, fitB, dB = ols_with_betas_and_p(
        df, "dislike_other12_genres", x_cols
    )

    # -----------------------
    # Save outputs (human-readable text)
    # -----------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # Model summaries
    write_text("./output/Table2_ModelA_summary.txt", mA.summary().as_text())
    write_text("./output/Table2_ModelB_summary.txt", mB.summary().as_text())

    # Paper-style tables (only betas + stars; and constant)
    write_text(
        "./output/Table2_ModelA_table.txt",
        "Table 2 Model A (Paper-style): standardized betas (computed from microdata) + stars; constant unstandardized\n"
        "Note: Hispanic dummy not included because no direct Hispanic identifier is present in provided data.\n\n"
        + tabA[["coef_with_stars"]].to_string()
        + "\n\nFit:\n"
        + pd.Series(fitA).to_string()
        + "\n",
    )
    write_text(
        "./output/Table2_ModelB_table.txt",
        "Table 2 Model B (Paper-style): standardized betas (computed from microdata) + stars; constant unstandardized\n"
        "Note: Hispanic dummy not included because no direct Hispanic identifier is present in provided data.\n\n"
        + tabB[["coef_with_stars"]].to_string()
        + "\n\nFit:\n"
        + pd.Series(fitB).to_string()
        + "\n",
    )

    # Overview
    overview = []
    overview.append("Table 2 replication attempt (1993 GSS) from provided extract\n")
    overview.append("Estimation: OLS with intercept; standardized betas computed as b * sd(x)/sd(y) on estimation sample.\n")
    overview.append("Stars based on p-values from OLS fit (* p<.05, ** p<.01, *** p<.001).\n")
    overview.append("Important: Table 2 in the paper reports only betas; SEs are not shown there.\n")
    overview.append("Important: Hispanic dummy is not estimable with the provided variables; therefore omitted.\n\n")
    overview.append("Model A DV: dislike_minority_genres (count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
    overview.append(pd.Series(fitA).to_string() + "\n\n")
    overview.append("Model B DV: dislike_other12_genres (count of dislikes among 12 remaining genres)\n")
    overview.append(pd.Series(fitB).to_string() + "\n")
    write_text("./output/Table2_overview.txt", "".join(overview))

    # Return results as dict of DataFrames
    fit_df = pd.DataFrame(
        [
            {"model": "ModelA_dislike_minority6", **fitA},
            {"model": "ModelB_dislike_other12", **fitB},
        ]
    ).set_index("model")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "Fit": fit_df,
    }