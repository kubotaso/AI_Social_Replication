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

    def clean_gss_na(x):
        """
        Conservative NA cleaning for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        """
        s = to_num(x).copy()
        # Common GSS missing/sentinel values across items
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        s = s.mask(s.isin(sentinels))
        return s

    def likert_dislike_indicator(x):
        """
        Music taste items: expected 1..5, where 4/5 indicate dislike.
        Missing if outside 1..5 or sentinel/NA.
        """
        s = clean_gss_na(x)
        s = s.where(s.between(1, 5))
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin([1, 2, 3])] = 0.0
        out.loc[s.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        s = clean_gss_na(x)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin(false_codes)] = 0.0
        out.loc[s.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all=False):
        """
        Sum of item-level dislike indicators.
        If require_all=True, DV is missing unless all items observed.
        If require_all=False, DV is sum over observed items but still requires >=1 observed.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_all:
            return mat.sum(axis=1, min_count=len(items))
        else:
            # at least 1 observed item
            return mat.sum(axis=1, min_count=1)

    def zscore(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_ols_with_betas(df_in, dv, xcols, model_name):
        """
        Fit OLS on unstandardized DV with intercept, compute standardized betas post-hoc:
            beta_j = b_j * sd(x_j) / sd(y)
        This allows a nonzero intercept (as in the paper), while reporting standardized slopes.
        Listwise deletion on dv + all xcols.
        """
        needed = [dv] + xcols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Guard: must have enough data
        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = to_num(d[dv])
        X = d[xcols].apply(to_num)

        # Drop any zero-variance predictors (but keep a record)
        dropped = []
        keep = []
        for c in X.columns:
            v = X[c].astype(float)
            sd = v.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                dropped.append(c)
            else:
                keep.append(c)
        X = X[keep]

        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors are zero-variance after cleaning; cannot fit model.")

        Xc = sm.add_constant(X, has_constant="add")

        # Final finite check to avoid statsmodels MissingDataError
        ok = np.isfinite(y.to_numpy())
        ok &= np.isfinite(Xc.to_numpy()).all(axis=1)
        y = y.loc[d.index[ok]]
        Xc = Xc.loc[d.index[ok]]

        if y.shape[0] < (Xc.shape[1] + 1):
            raise ValueError(f"{model_name}: not enough cases after finite filtering (n={y.shape[0]}).")

        model = sm.OLS(y, Xc).fit()

        # Standardized betas (exclude intercept)
        y_sd = y.std(ddof=0)
        betas = {}
        if not np.isfinite(y_sd) or y_sd == 0:
            for c in X.columns:
                betas[c] = np.nan
        else:
            for c in X.columns:
                x_sd = X[c].std(ddof=0)
                b = model.params.get(c, np.nan)
                betas[c] = (b * x_sd / y_sd) if (np.isfinite(x_sd) and x_sd != 0 and np.isfinite(b)) else np.nan

        # Build output table: standardized beta + (computed) p-values/stars (not from paper)
        def stars(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        rows = []
        # Intercept row: keep unstandardized constant (beta undefined)
        rows.append(
            {
                "term": "Constant",
                "b_unstd": float(model.params.get("const", np.nan)),
                "beta_std": np.nan,
                "p_value": float(model.pvalues.get("const", np.nan)),
                "sig": stars(float(model.pvalues.get("const", np.nan))),
            }
        )
        for c in xcols:
            if c in X.columns:
                rows.append(
                    {
                        "term": c,
                        "b_unstd": float(model.params.get(c, np.nan)),
                        "beta_std": float(betas.get(c, np.nan)),
                        "p_value": float(model.pvalues.get(c, np.nan)),
                        "sig": stars(float(model.pvalues.get(c, np.nan))),
                    }
                )
            else:
                # Dropped for zero variance; keep row for faithful structure
                rows.append({"term": c, "b_unstd": np.nan, "beta_std": np.nan, "p_value": np.nan, "sig": ""})

        tab = pd.DataFrame(rows).set_index("term")

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_including_const": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Write outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            if dropped:
                f.write("\n\nDropped (zero variance): " + ", ".join(dropped) + "\n")

        # Human-readable table focused on Table-2-like content:
        # standardized betas + stars, plus constant (unstandardized)
        tab_show = tab.copy()
        tab_show["beta_std"] = tab_show["beta_std"].map(lambda v: f"{v: .3f}" if pd.notna(v) else "")
        tab_show["b_unstd"] = tab_show["b_unstd"].map(lambda v: f"{v: .3f}" if pd.notna(v) else "")
        tab_show["p_value"] = tab_show["p_value"].map(lambda v: f"{v: .4f}" if pd.notna(v) else "")
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write("Note: Standardized coefficients (beta_std) computed from microdata via beta = b * sd(x)/sd(y).\n")
            f.write("Stars are computed from two-tailed OLS p-values from microdata (not taken from the paper).\n\n")
            f.write(tab_show.to_string())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        return model, tab, fit, d.index

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year == 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    if df.shape[0] == 0:
        raise ValueError("No rows after filtering YEAR==1993.")

    # -----------------------------
    # Construct DVs (counts)
    # -----------------------------
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

    # Use require_all=False to avoid over-pruning the sample; paper notes DK treated as missing.
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all=False)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all=False)

    # -----------------------------
    # Construct racism score (0-5)
    # -----------------------------
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
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)  # require all 5 components

    # -----------------------------
    # RHS controls
    # -----------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    df["education_years"] = clean_gss_na(df["educ"]).where(clean_gss_na(df["educ"]).between(0, 20))

    # Income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(clean_gss_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = clean_gss_na(df["age"]).where(clean_gss_na(df["age"]).between(18, 89))

    # Race dummies (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_na(df["race"]).where(clean_gss_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables; keep as missing column and exclude from model
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_gss_na(df["relig"])
    denom = clean_gss_na(df["denom"])
    consprot = np.where(relig.isna() | denom.isna(), np.nan, ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float))
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_na(df["region"]).where(clean_gss_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Fit models (Table 2)
    # -----------------------------
    # Keep Table-2 RHS variables; include hispanic if present (it's all-missing here, so exclude to avoid n=0).
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # not available in this extract; including would drop all cases
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Ensure predictors exist
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    mA, tabA, fitA, idxA = fit_ols_with_betas(df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6")
    mB, tabB, fitB, idxB = fit_ols_with_betas(df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12")

    # Overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993 extract provided)\n")
        f.write("Models: OLS with intercept; standardized coefficients computed post-hoc as beta=b*sd(x)/sd(y).\n")
        f.write("Note: Hispanic indicator not included because no direct Hispanic field is present in the provided variables.\n")
        f.write("Note: Stars are computed from microdata p-values (two-tailed), not copied from the paper.\n\n")
        f.write("Model A DV: count of disliked among {Rap, Reggae, Blues, Jazz, Gospel, Latin}\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: count of disliked among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }