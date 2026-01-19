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
        Conservative missing-code handler for this extract.
        The provided file already has many missings as blank/NA; we additionally
        treat common GSS sentinel codes as missing.
        """
        x = to_num(x).copy()
        x = x.replace([np.inf, -np.inf], np.nan)
        # common sentinel codes in GSS-style extracts (applied conservatively)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5 scale; 4/5 indicates dislike.
        Missing if not in 1..5 after NA-code cleaning.
        """
        x = clean_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(item, true_codes, false_codes):
        x = clean_na_codes(item)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_series(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_dislike_count_allow_partial(df, items, min_answered=None):
        """
        Count of disliked genres across 'items', treating item-level DK/NA as missing.
        To avoid collapsing N too hard (and to be closer to typical index construction),
        we allow partial responses as long as at least `min_answered` items are observed.
        Default: require at least half observed.
        """
        if min_answered is None:
            min_answered = int(np.ceil(len(items) / 2))
        mats = []
        for c in items:
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)

        n_answered = mat.notna().sum(axis=1)
        count_disliked = mat.sum(axis=1, min_count=1)

        # keep only if enough items answered; else missing DV
        count_disliked = count_disliked.where(n_answered >= min_answered, np.nan)
        return count_disliked

    def compute_standardized_betas_and_pvalues(model, y, X_no_const):
        """
        Fit OLS on unstandardized y, X (with const). Then compute standardized betas as:
        beta_std_j = b_j * sd(x_j) / sd(y)
        Dummies are standardized the same mechanical way (as beta weights typically are).
        """
        params = model.params.copy()
        bse = model.bse.copy()
        tvals = model.tvalues.copy()
        pvals = model.pvalues.copy()

        y_sd = float(np.std(y, ddof=0))
        betas = {}
        for c in X_no_const.columns:
            x_sd = float(np.std(X_no_const[c], ddof=0))
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = params[c] * (x_sd / y_sd)

        # constant has no standardized beta
        betas["const"] = np.nan

        tab = pd.DataFrame(
            {
                "beta_std": pd.Series(betas),
                "b_unstd": params,
                "p_value": pvals,
            }
        )

        # Stars per conventional thresholds (replication-based, not "from the PDF")
        def star(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        tab["sig"] = tab["p_value"].map(star)
        tab["t"] = tvals
        tab["std_err"] = bse

        # Reorder: const last like typical tables
        order = [c for c in X_no_const.columns] + ["const"]
        tab = tab.reindex(order)
        return tab

    def fit_model(df, dv, xcols, model_name):
        # Listwise deletion on variables in the model only
        needed = [dv] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < len(xcols) + 5:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        y = to_num(d[dv]).astype(float)
        X = d[xcols].apply(to_num).astype(float)

        # Drop zero-variance predictors (should not happen for key dummies; if it does, log it)
        dropped = []
        keep = []
        for c in X.columns:
            sd = float(X[c].std(ddof=0))
            if not np.isfinite(sd) or sd == 0:
                dropped.append(c)
            else:
                keep.append(c)
        X = X[keep]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        tab = compute_standardized_betas_and_pvalues(model, y, X)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excluding intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            if dropped:
                f.write("\n\nDropped zero-variance predictors:\n")
                f.write(", ".join(dropped) + "\n")
            f.write("\n\nNOTE: Standard errors/p-values are computed from this replication run; "
                    "the published Table 2 reports standardized coefficients only.\n")

        # Save table in a paper-like view (standardized beta + stars), and an augmented view
        paper_like = tab[["beta_std", "sig"]].copy()
        paper_like.to_string(
            open(f"./output/{model_name}_table_paper_like.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x: .3f}"
        )

        tab.to_string(
            open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8"),
            float_format=lambda x: f"{x: .6f}"
        )

        return tab, fit

    # -----------------------
    # Load and filter year
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Build DVs
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    missing_music = [c for c in (minority_items + other12_items) if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing music item columns: {missing_music}")

    # Allow partial response to avoid collapsing N; require at least half answered
    df["dislike_minority_genres"] = build_dislike_count_allow_partial(df, minority_items, min_answered=3)
    df["dislike_other12_genres"] = build_dislike_count_allow_partial(df, other12_items, min_answered=6)

    # -----------------------
    # Racism score (0-5), require all 5 items (as specified)
    # -----------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    missing_rac = [c for c in racism_fields if c not in df.columns]
    if missing_rac:
        raise ValueError(f"Missing racism item columns: {missing_rac}")

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
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income per capita: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_na_codes(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies from RACE (White reference)
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: not available in provided variables; keep but DO NOT include in models
    # (as per mapping instruction, do not proxy using 'ethnic')
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column needed for religion coding: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    df["no_religion"] = norelig.where(~relig.isna(), np.nan)

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -----------------------
    # Models (Table 2): exclude hispanic because unavailable
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

    # sanity: ensure predictors exist
    missing_pred = [c for c in x_cols if c not in df.columns]
    if missing_pred:
        raise ValueError(f"Missing constructed predictors: {missing_pred}")

    tabA, fitA = fit_model(df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6")
    tabB, fitB = fit_model(df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12")

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993)\n")
        f.write("Models: OLS; standardized coefficients computed as b * sd(x)/sd(y)\n")
        f.write("Important: 'Hispanic' indicator is not available in the provided extract and is omitted.\n")
        f.write("Important: Standard errors/p-values are computed from this replication run; the published table omits SEs.\n\n")
        f.write("Model A DV: count of disliked among {Rap, Reggae, Blues, Jazz, Gospel, Latin} (partial-response allowed)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: count of disliked among the other 12 genres (partial-response allowed)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }