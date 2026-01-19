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

    def clean_gss_missing(x):
        """
        Conservative missing-code cleaner for this extract:
        - Coerce to numeric
        - Treat typical GSS sentinel codes as missing
        - Do NOT treat 8/9 as missing globally because some real variables can take 8/9 validly.
          Here we apply variable-specific validation after this step.
        """
        x = to_num(x)
        sentinels = {97, 98, 99, 997, 998, 999, 9997, 9998, 9999}
        x = x.mask(x.isin(list(sentinels)))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: expected 1..5 where 4/5 = dislike.
        Anything outside 1..5 treated as missing.
        """
        x = clean_gss_missing(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def bin_from_codes(series, true_codes, false_codes, valid_codes=None):
        x = clean_gss_missing(series)
        if valid_codes is not None:
            x = x.where(x.isin(valid_codes))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, cols):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in cols], axis=1)
        # Bryson: DK treated as missing and missing cases excluded -> require complete items for the DV
        return mat.sum(axis=1, min_count=len(cols))

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def stars_from_p(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df_in, dv_col, x_order, model_name):
        """
        Fit OLS on unstandardized DV and X (with intercept),
        then compute standardized coefficients (betas) as:
            beta_j = b_j * sd(x_j) / sd(y)
        using estimation sample SDs (ddof=0), consistent with standard beta definition.
        """
        needed = [dv_col] + x_order
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_order)}).")

        y = to_num(d[dv_col])
        X = d[x_order].apply(to_num)

        # Drop any zero-variance predictors in THIS estimation sample (prevents NaN rows)
        drop_zero = []
        for c in list(X.columns):
            if not np.isfinite(X[c].std(ddof=0)) or X[c].std(ddof=0) == 0:
                drop_zero.append(c)
        if drop_zero:
            X = X.drop(columns=drop_zero)
            x_used = [c for c in x_order if c not in drop_zero]
        else:
            x_used = list(x_order)

        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance after listwise deletion; cannot fit.")

        Xc = sm.add_constant(X, has_constant="add")
        m = sm.OLS(y, Xc).fit()

        # Standardized betas for slopes (not for constant)
        sd_y = y.std(ddof=0)
        betas = {}
        for term in m.params.index:
            if term == "const":
                betas[term] = np.nan
            else:
                sd_x = X[term].std(ddof=0)
                betas[term] = float(m.params[term] * (sd_x / sd_y)) if (np.isfinite(sd_x) and sd_x != 0 and np.isfinite(sd_y) and sd_y != 0) else np.nan

        tab_full = pd.DataFrame(
            {
                "b_unstd": m.params,
                "std_err": m.bse,
                "t": m.tvalues,
                "p_value": m.pvalues,
                "beta_std": pd.Series(betas),
            }
        )
        tab_full.index.name = "term"

        # Paper-style table: standardized betas + stars (and constant unstandardized)
        paper_rows = []
        for v in ["racism_score", "education_years", "hh_income_per_capita", "occ_prestige",
                  "female", "age_years", "black", "hispanic", "other_race",
                  "cons_protestant", "no_religion", "south"]:
            if v in tab_full.index:
                beta = tab_full.loc[v, "beta_std"]
                p = tab_full.loc[v, "p_value"]
                paper_rows.append((v, beta, stars_from_p(p)))
            else:
                paper_rows.append((v, np.nan, ""))

        # Constant (unstandardized)
        if "const" in tab_full.index:
            paper_rows.append(("const", tab_full.loc["const", "b_unstd"], stars_from_p(tab_full.loc["const", "p_value"])))
        else:
            paper_rows.append(("const", np.nan, ""))

        tab_paper = pd.DataFrame(paper_rows, columns=["term", "coef", "sig"]).set_index("term")

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "n": int(m.nobs),
                "k_predictors": int(m.df_model),  # excludes constant
                "r2": float(m.rsquared),
                "adj_r2": float(m.rsquared_adj),
                "dropped_zero_variance_predictors": ", ".join(drop_zero) if drop_zero else ""
            }]
        )

        return m, tab_full, tab_paper, fit, d.index

    # -----------------------------
    # Load / filter (1993)
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs (Table 2)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items)

    # -----------------------------
    # Racism score (0-5 additive index)
    # -----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = bin_from_codes(df["rachaf"], true_codes=[1], false_codes=[2], valid_codes=[1, 2])      # object
    rac2 = bin_from_codes(df["busing"], true_codes=[2], false_codes=[1], valid_codes=[1, 2])      # oppose busing
    rac3 = bin_from_codes(df["racdif1"], true_codes=[2], false_codes=[1], valid_codes=[1, 2])     # deny discrimination
    rac4 = bin_from_codes(df["racdif3"], true_codes=[2], false_codes=[1], valid_codes=[1, 2])     # deny edu chance
    rac5 = bin_from_codes(df["racdif4"], true_codes=[1], false_codes=[2], valid_codes=[1, 2])     # endorse motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # RHS variables (Table 2)
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing required column: educ")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing required column: prestg80")
    prest = clean_gss_missing(df["prestg80"])
    df["occ_prestige"] = prest.where(prest.between(0, 100))

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing required column: sex")
    df["female"] = bin_from_codes(df["sex"], true_codes=[2], false_codes=[1], valid_codes=[1, 2])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing required column: age")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing required column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables; do NOT proxy.
    # To keep Table 2 RHS structure without inducing listwise deletion to n=0,
    # we create a neutral (all 0) placeholder AND EXCLUDE it from estimation if it has no variance.
    # However, to keep output aligned, we include it and let the model drop it only if needed.
    df["hispanic"] = np.nan  # unknown -> will be listwise-missing if included; we therefore handle below.

    # Conservative Protestant proxy: RELIG==1 (Protestant) & DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    ok = relig.isin([1, 2, 3, 4]) & denom.notna()
    consprot.loc[ok] = ((relig.loc[ok] == 1) & (denom.loc[ok].isin([1, 6, 7]))).astype(float)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = pd.Series(np.nan, index=df.index, dtype="float64")
    okr = relig.isin([1, 2, 3, 4])
    norelig.loc[okr] = (relig.loc[okr] == 4).astype(float)
    df["no_religion"] = norelig

    # South: REGION==3 (region codes expected 1..4)
    if "region" not in df.columns:
        raise ValueError("Missing required column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = pd.Series(np.nan, index=df.index, dtype="float64")
    south.loc[region.notna()] = (region.loc[region.notna()] == 3).astype(float)
    df["south"] = south

    # -----------------------------
    # Model spec (Table 2 order)
    # IMPORTANT: Hispanic not available -> we will run TWO versions:
    #   (1) "as-available" model: exclude hispanic to avoid n=0
    #   (2) "table-aligned" output: include hispanic row as NA in paper-style table.
    # This keeps code runnable and results auditable given the provided extract.
    # -----------------------------
    x_order_with_hisp = [
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
    x_order_no_hisp = [c for c in x_order_with_hisp if c != "hispanic"]

    # -----------------------------
    # Fit models (exclude Hispanic to prevent listwise deletion collapse)
    # -----------------------------
    results = {}

    def run_one(dv, model_name):
        m, tab_full, tab_paper, fit, used_idx = fit_table2_model(df, dv, x_order_no_hisp, model_name)

        # Reinsert hispanic row (as NA) into both tables for display alignment
        if "hispanic" not in tab_full.index:
            tab_full.loc["hispanic"] = [np.nan, np.nan, np.nan, np.nan, np.nan]
        tab_full = tab_full.loc[[t for t in (["const"] + x_order_with_hisp) if t in tab_full.index] + [t for t in tab_full.index if t not in (["const"] + x_order_with_hisp)]]
        # For paper table we already include all terms, but ensure ordering
        tab_paper = tab_paper.loc[[t for t in (x_order_with_hisp + ["const"]) if t in tab_paper.index]]

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(m.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Standardized coefficients (beta_std) computed from unstandardized slopes using SD(x)/SD(y).\n")
            f.write("- Hispanic indicator not available in provided extract; shown as NA and excluded from estimation.\n")
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(tab_full.to_string(float_format=lambda v: f"{v: .6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write(tab_paper.to_string(float_format=lambda v: f"{v: .6f}"))
            f.write("\n")

        return tab_full, tab_paper, fit

    tabA_full, tabA_paper, fitA = run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    tabB_full, tabB_paper, fitB = run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    results["ModelA_table_full"] = tabA_full
    results["ModelA_table_paper_style"] = tabA_paper
    results["ModelA_fit"] = fitA

    results["ModelB_table_full"] = tabB_full
    results["ModelB_table_paper_style"] = tabB_paper
    results["ModelB_fit"] = fitB

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993) - OLS with standardized coefficients (betas)\n")
        f.write("Dependent variables:\n")
        f.write("  Model A: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write("  Model B: count of dislikes among the remaining 12 genres\n\n")
        f.write("Important limitation:\n")
        f.write("  Hispanic dummy is not available in the provided variable list; it is excluded from estimation and shown as NA.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return results