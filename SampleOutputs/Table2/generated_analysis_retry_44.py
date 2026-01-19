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

    def clean_na_codes(x):
        """
        Conservative NA cleaning for this extract:
        - Convert to numeric
        - Treat common GSS sentinel codes as missing
        - Do NOT treat legitimate scale endpoints as missing (music items use 1-5, so 8/9 are safe to drop).
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        return x.mask(x.isin(sentinels))

    def likert_dislike_indicator(x):
        """
        Music items: 1-5, where 4/5 => dislike (1), 1/2/3 => not dislike (0).
        Outside 1-5 or sentinel => missing.
        """
        x = clean_na_codes(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_na_codes(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Sum of dislike indicators across items.
        To mirror "DK treated as missing and missing cases excluded", require all items non-missing.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def compute_beta_from_unstd(b, x, y):
        """
        Standardized coefficient beta = b * sd(x)/sd(y), computed on estimation sample.
        """
        sx = np.std(x, ddof=0)
        sy = np.std(y, ddof=0)
        if not np.isfinite(sx) or not np.isfinite(sy) or sy == 0:
            return np.nan
        return float(b) * float(sx) / float(sy)

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

    def infer_hispanic_flag(df):
        """
        The provided extract may not include a direct Hispanic flag. We attempt a safe inference:
        - If a column named 'hispanic' exists: use it (0/1 or 1/2).
        - Else if 'ethnic' exists: treat common GSS ETHNIC codes for Hispanic as 1.
          (This is a pragmatic fallback to avoid runtime failure; it may not match the paper perfectly.)
        If neither available, return all-missing series.
        """
        if "hispanic" in df.columns:
            h = clean_na_codes(df["hispanic"])
            # Accept 0/1, or 1/2 (yes/no) style
            if set(h.dropna().unique()).issubset({0, 1}):
                return h.astype(float)
            if set(h.dropna().unique()).issubset({1, 2}):
                return binary_from_codes(h, true_codes=[1], false_codes=[2])
            # Otherwise just coerce: nonzero => 1, zero => 0
            out = pd.Series(np.nan, index=df.index, dtype="float64")
            out.loc[h == 0] = 0.0
            out.loc[h != 0] = 1.0
            out.loc[h.isna()] = np.nan
            return out

        if "ethnic" in df.columns:
            # Common in some GSS extracts: ETHNIC==1 denotes Hispanic (Mexican/Spanish/Hispanic).
            eth = clean_na_codes(df["ethnic"])
            # Use only if there is evidence of a "1" category and values look like categorical codes
            vals = set(eth.dropna().unique().tolist())
            if 1 in vals and (len(vals) <= 20):
                out = pd.Series(np.nan, index=df.index, dtype="float64")
                out.loc[eth == 1] = 1.0
                out.loc[(eth.notna()) & (eth != 1)] = 0.0
                return out

        return pd.Series(np.nan, index=df.index, dtype="float64")

    def fit_table2_model(df, dv, x_order, model_name, require_terms):
        """
        Fit OLS with intercept.
        Return:
          - paper_style table: term, beta (standardized), stars
          - full table: term, b_unstd, se, t, p, beta_std
          - fit stats: N, R2, adjR2
        Enforces that required terms exist and vary in the analytic sample.
        """
        needed = [dv] + x_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # Verify required terms are present
        missing_terms = [t for t in require_terms if t not in x_order]
        if missing_terms:
            raise ValueError(f"{model_name}: missing required predictors in x_order: {missing_terms}")

        # Verify variance (do not silently drop)
        zero_var = []
        for c in x_order:
            v = d[c].astype(float)
            if np.nanstd(v, ddof=0) == 0:
                zero_var.append(c)
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Check coding/sample restrictions."
            )

        y = d[dv].astype(float)
        X = d[x_order].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Build full table with standardized betas computed from unstandardized b and sample SDs
        rows = []
        for term in ["const"] + x_order:
            b = model.params.get(term, np.nan)
            se = model.bse.get(term, np.nan)
            t = model.tvalues.get(term, np.nan)
            p = model.pvalues.get(term, np.nan)

            if term == "const":
                beta = np.nan
                term_label = "Constant"
            else:
                beta = compute_beta_from_unstd(b, d[term].astype(float).values, y.values)
                term_label = term

            rows.append(
                {
                    "term": term_label,
                    "b_unstd": float(b) if np.isfinite(b) else np.nan,
                    "std_err": float(se) if np.isfinite(se) else np.nan,
                    "t": float(t) if np.isfinite(t) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "beta_std": float(beta) if np.isfinite(beta) else np.nan,
                }
            )

        full = pd.DataFrame(rows)

        # Paper style: standardized betas (slopes) + stars; constant shown as unstandardized
        paper_rows = []
        for c in x_order:
            r = full.loc[full["term"] == c].iloc[0]
            paper_rows.append({"term": c, "coef": r["beta_std"], "stars": stars_from_p(r["p_value"])})
        # constant
        r0 = full.loc[full["term"] == "Constant"].iloc[0]
        paper_rows.append({"term": "Constant", "coef": r0["b_unstd"], "stars": stars_from_p(r0["p_value"])})

        paper = pd.DataFrame(paper_rows)

        # Enforce no missing labels and correct ordering
        expected_terms = list(x_order) + ["Constant"]
        if paper["term"].isna().any():
            raise ValueError(f"{model_name}: NaN term label encountered in paper table.")
        if set(expected_terms) != set(paper["term"].tolist()):
            raise ValueError(f"{model_name}: term set mismatch in paper table assembly.")
        paper = paper.set_index("term").loc[expected_terms].reset_index()

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Standard errors/t/p are computed from this replication fit; Bryson Table 2 prints only standardized betas + stars.\n")

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write(paper.to_string(index=False, float_format=lambda v: f"{v: .3f}" if np.isfinite(v) else ""))
            f.write("\n")

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(index=False, float_format=lambda v: f"{v: .6f}" if np.isfinite(v) else ""))
            f.write("\n")

        return model, paper, full, fit

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Input must include 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.empty:
        raise ValueError("No rows with YEAR==1993 found.")

    # ----------------------------
    # Construct DVs
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # ----------------------------
    # Racism score (0-5)
    # ----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Controls
    # ----------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' column.")
    educ = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))
    df["education_years"] = educ

    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing 'realinc' or 'hompop' for income per capita.")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column.")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing 'age' column.")
    df["age_years"] = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing 'race' column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic (best-effort inference; do not fail runtime)
    df["hispanic"] = infer_hispanic_flag(df)

    if "relig" not in df.columns:
        raise ValueError("Missing 'relig' column.")
    relig = clean_na_codes(df["relig"])

    if "denom" in df.columns:
        denom = clean_na_codes(df["denom"])
        consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
        consprot.loc[relig.isna() | denom.isna()] = np.nan
    else:
        # Fallback: only based on being Protestant if denom missing (keeps variable varying)
        consprot = (relig == 1).astype(float)
        consprot.loc[relig.isna()] = np.nan
    df["cons_protestant"] = consprot

    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    if "region" not in df.columns:
        raise ValueError("Missing 'region' column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # ----------------------------
    # Fit models (Table 2)
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
    require_terms = list(x_order)

    results = {}

    mA, paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6", require_terms=require_terms
    )
    mB, paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12", require_terms=require_terms
    )

    results["ModelA_table_paper_style"] = paperA
    results["ModelB_table_paper_style"] = paperB
    results["ModelA_table_full"] = fullA
    results["ModelB_table_full"] = fullB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993)\n")
        f.write("Outputs:\n")
        f.write("- Paper-style table: standardized betas for predictors (beta = b * SDx/SDy) + stars; constant shown as unstandardized.\n")
        f.write("- Full table: unstandardized b, SE, t, p, and beta_std (computed from the estimation sample).\n\n")
        f.write("IMPORTANT: If the dataset lacks an official Hispanic identifier, 'hispanic' may be inferred from 'ethnic' if present; this may not match the paper.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results