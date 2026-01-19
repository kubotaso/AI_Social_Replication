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

    def clean_missing(x):
        """
        Conservative missing handling for this extract:
        - coerce to numeric
        - treat common GSS sentinel codes as missing
        """
        x = to_num(x)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music items expected on 1-5:
          1/2/3 -> 0 (not disliked)
          4/5   -> 1 (disliked)
        Missing if not in 1..5 after cleaning.
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

    def build_dislike_count(df, items, require_all_items=True):
        """
        Sum dislike indicators across a set of music items.
        Paper note: DK treated as missing and cases excluded.
        -> We use complete-case across the items for that DV by default.
        """
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        if require_all_items:
            return mat.sum(axis=1, min_count=len(items))
        else:
            # Not used for Table 2 replication; kept for debugging.
            return mat.sum(axis=1, min_count=1)

    def standardize_beta_from_unstd(res, X, y):
        """
        Compute standardized coefficients (beta) from unstandardized OLS:
          beta_j = b_j * sd(x_j) / sd(y)
        Intercept beta is not meaningful -> NaN.
        """
        sd_y = np.nanstd(y, ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                betas[col] = np.nan
            else:
                sd_x = np.nanstd(X[col], ddof=0)
                betas[col] = res.params[col] * (sd_x / sd_y) if (sd_x > 0 and sd_y > 0) else np.nan
        return pd.Series(betas)

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

    def fit_table2_model(df, dv, x_order, model_name):
        """
        Faithful/simple implementation:
        - listwise deletion on DV + all predictors (including hispanic if present)
        - OLS with intercept
        - standardized betas computed from unstandardized model using SD ratios
        - stars from model p-values (two-tailed)
        - do NOT drop zero-variance predictors; if any become constant in the analytic sample,
          keep them but mark results as NaN and note it (statsmodels will drop if perfectly collinear);
          we handle by explicit variance check and removing only those constant columns to avoid runtime errors.
        """
        needed = [dv] + x_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Build X with intercept
        y = d[dv].astype(float)
        X = d[x_order].astype(float)

        # Remove any zero-variance predictors in THIS analytic sample (otherwise singular matrix)
        zero_var = [c for c in X.columns if np.nanstd(X[c].values, ddof=0) == 0]
        X_use = X.drop(columns=zero_var) if zero_var else X

        Xc = sm.add_constant(X_use, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        # Compute standardized betas (for the predictors actually used)
        betas = standardize_beta_from_unstd(res, Xc, y)

        # Build "paper-style" table in the paper's row order.
        # For predictors removed due to zero variance, show NaN and note in output.
        paper_rows = []
        for term in x_order:
            if term in Xc.columns:
                beta = float(betas.get(term, np.nan))
                p = float(res.pvalues.get(term, np.nan))
            else:
                beta = np.nan
                p = np.nan
            paper_rows.append(
                {
                    "term": term,
                    "beta": beta,
                    "stars": sig_stars(p),
                    "p_value": p,
                }
            )

        # Constant (unstandardized only; Table 2 prints a constant but reports standardized slopes)
        const_b = float(res.params.get("const", np.nan))
        const_p = float(res.pvalues.get("const", np.nan))
        paper_rows.append(
            {
                "term": "constant",
                "beta": np.nan,
                "stars": sig_stars(const_p),
                "p_value": const_p,
                "constant_b": const_b,
            }
        )

        paper = pd.DataFrame(paper_rows)

        # Full replication outputs (not in paper, but useful)
        full = pd.DataFrame(
            {
                "b_unstd": res.params,
                "std_err": res.bse,
                "t": res.tvalues,
                "p_value": res.pvalues,
                "beta": betas,
            }
        )
        full.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(res.nobs),
                    "r2": float(res.rsquared),
                    "adj_r2": float(res.rsquared_adj),
                    "zero_variance_dropped": ", ".join(zero_var) if zero_var else "",
                }
            ]
        )

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Table 2 prints standardized coefficients (betas) for slopes; betas here are computed from unstandardized OLS via SD ratios.\n")
            f.write("- Stars are based on this model's p-values (two-tailed): * p<.05, ** p<.01, *** p<.001.\n")
            if zero_var:
                f.write(f"- Dropped due to zero variance in analytic sample: {zero_var}\n")

        # Human-readable tables
        paper_out = paper.copy()
        # Format beta with stars; constant shown separately
        def fmt_beta(row):
            if row["term"] == "constant":
                b = row.get("constant_b", np.nan)
                if pd.isna(b):
                    return ""
                return f"{b:.3f}{row['stars']}"
            if pd.isna(row["beta"]):
                return ""
            return f"{row['beta']:.3f}{row['stars']}"

        paper_out["coef"] = paper_out.apply(fmt_beta, axis=1)
        paper_out = paper_out[["term", "coef", "p_value"]]

        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}: standardized betas (slopes) + stars; constant is unstandardized.\n\n")
            f.write(paper_out.to_string(index=False))

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda v: f"{v: .6f}"))

        return paper_out, full, fit

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year == 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # DVs (counts)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_items=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_items=True)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
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

    # -------------------------
    # Controls
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing required column: educ")
    educ = clean_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(lambda s: s > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing required column: prestg80")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing required column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing required column: age")
    age = clean_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: black and other_race from RACE (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing required column: race")
    race = clean_missing(df["race"]).where(lambda s: s.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not directly available in provided variable list.
    # To keep the model runnable and transparent, include it only if a column exists.
    if "hispanic" in df.columns:
        hisp = clean_missing(df["hispanic"])
        # assume already coded 0/1; otherwise map common 1/2 scheme if present
        if set(hisp.dropna().unique()).issubset({0, 1}):
            df["hispanic_dummy"] = hisp.astype(float)
        else:
            # try 1=yes 2=no
            df["hispanic_dummy"] = binary_from_codes(hisp, true_codes=[1], false_codes=[2])
    else:
        df["hispanic_dummy"] = np.nan  # will be listwise-dropped if included; we will not include if all-missing

    # Conservative Protestant proxy
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    relig = clean_missing(df["relig"])
    denom = clean_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.mask(relig.isna() | denom.isna())
    df["cons_protestant"] = consprot

    # No religion
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # Southern
    if "region" not in df.columns:
        raise ValueError("Missing required column: region")
    region = clean_missing(df["region"]).where(lambda s: s.isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -------------------------
    # Build RHS list in Table 2 order
    # -------------------------
    x_order_base = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic_dummy",   # included only if it has non-missing variation
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # If hispanic_dummy is all-missing, drop it (otherwise listwise deletion nukes the sample)
    if df["hispanic_dummy"].notna().sum() == 0:
        x_order = [c for c in x_order_base if c != "hispanic_dummy"]
    else:
        x_order = x_order_base

    # -------------------------
    # Fit both models
    # -------------------------
    paperA, fullA, fitA = fit_table2_model(
        df,
        dv="dislike_minority_genres",
        x_order=x_order,
        model_name="Table2_ModelA_dislike_minority6",
    )
    paperB, fullB, fitB = fit_table2_model(
        df,
        dv="dislike_other12_genres",
        x_order=x_order,
        model_name="Table2_ModelB_dislike_other12",
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson Table 2 replication attempt (GSS 1993)\n")
        f.write("Outputs:\n")
        f.write("- Paper-style tables: standardized betas (slopes) + stars; constant unstandardized.\n")
        f.write("- Full tables: unstandardized b, SE, t, p, and computed standardized beta.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nNotes:\n")
        f.write("- If hispanic is not available in the provided extract, it is omitted to avoid collapsing the sample.\n")
        f.write("- Music dislike DVs require complete responses on the relevant genre items.\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }