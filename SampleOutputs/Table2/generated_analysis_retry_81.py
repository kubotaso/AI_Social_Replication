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
        Conservative missing handling:
        - coerce to numeric
        - set common "not ascertained/refused/dk" style sentinels to NaN
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999, 0}
        x = x.mask(x.isin(list(sentinels)))
        return x

    def likert_dislike_indicator(series):
        """
        Music liking items: 1-5 where 4/5 = dislike.
        Treat values outside 1..5 as missing.
        """
        x = clean_gss_missing(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_gss_missing(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all_answered=False):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_all_answered:
            return mat.sum(axis=1, min_count=len(items))
        # Allow partial: sum across answered items (Bryson notes DK treated as missing; counts constructed from available)
        return mat.sum(axis=1, min_count=1)

    def standardized_betas_from_ols(res, X, y):
        """
        Standardized betas: b_j * sd(X_j) / sd(y), for j excluding intercept.
        Uses sample SD (ddof=1).
        """
        sd_y = y.std(ddof=1)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            sd_x = X[col].std(ddof=1)
            betas[col] = res.params[col] * (sd_x / sd_y) if (sd_x > 0 and sd_y > 0) else np.nan
        return pd.Series(betas)

    def add_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_one(df, dv, xcols, model_name, pretty):
        d = df[[dv] + xcols].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if d.shape[0] < (len(xcols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)
        X = d[xcols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        # If singular, explicitly drop exact-zero-variance columns (rare but can happen after filtering)
        zero_var = [c for c in Xc.columns if c != "const" and Xc[c].std(ddof=0) == 0]
        if zero_var:
            Xc = Xc.drop(columns=zero_var)
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_ols(res, Xc, y)
        # Align betas to requested xcols order; missing ones (dropped) become NaN
        beta_std = pd.Series({c: betas.get(c, np.nan) for c in xcols}, name="beta_std")

        # Build table like Table 2: standardized betas + stars; plus constant (unstandardized)
        pvals = res.pvalues
        out_rows = []
        for c in xcols:
            b = beta_std[c]
            p = pvals.get(c, np.nan)
            out_rows.append(
                {
                    "term": pretty.get(c, c),
                    "beta_std": b,
                    "stars": add_stars(p),
                }
            )
        # Constant row: unstandardized intercept
        out_rows.append(
            {
                "term": "Constant",
                "beta_std": res.params.get("const", np.nan),
                "stars": add_stars(pvals.get("const", np.nan)),
            }
        )
        table = pd.DataFrame(out_rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(res.nobs),
                    "r2": float(res.rsquared),
                    "adj_r2": float(res.rsquared_adj),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
            f.write("\n\nStandardized coefficients (beta): computed as b * SD(X)/SD(Y) on the estimation sample.\n")
            if zero_var:
                f.write(f"\nDropped zero-variance predictors: {', '.join(zero_var)}\n")
            f.write("\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(table.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

        return res, table, fit, d.index

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct DVs (counts of dislikes)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Bryson treats DK as missing; we avoid forcing all 18 answered to reduce N collapse.
    df["dislike_6_minority_associated"] = build_dislike_count(df, minority_items, require_all_answered=False)
    df["dislike_12_remaining"] = build_dislike_count(df, other12_items, require_all_answered=False)

    # -----------------------------
    # Racism score (0-5 additive)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])    # object to majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])    # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])   # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])   # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])   # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    # Require all 5 to match "sum of five items" and avoid changing meaning
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_gss_missing(df["educ"])
    df["education"] = educ.where(educ.between(0, 20))

    # Income per capita: realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_pc"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female: SEX 1 male, 2 female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_gss_missing(df["age"])
    df["age"] = age.where(age.between(18, 89))

    # Race dummies: RACE 1 white, 2 black, 3 other
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if available in this extract; treat 1 as Hispanic/Latino, 0 otherwise
    # (If ETHNIC is not coded this way in your extract, this will surface in diagnostics/output.)
    if "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        # Common in some GSS extracts: 1 = Hispanic; other positive codes = non-Hispanic categories.
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))
    else:
        # If absent, keep missing; model will fail loudly due to listwise deletion if included.
        df["hispanic"] = np.nan

    # Religion: Conservative Protestant proxy + No religion
    if "relig" not in df.columns:
        raise ValueError("Missing relig column.")
    relig = clean_gss_missing(df["relig"])

    if "denom" in df.columns:
        denom = clean_gss_missing(df["denom"])
        consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
        consprot = consprot.mask(relig.isna() | denom.isna())
    else:
        # fallback: cannot classify by denom; set missing to avoid misclassification
        consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    df["cons_protestant"] = consprot

    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -----------------------------
    # Model specification (Table 2 RHS)
    # -----------------------------
    xcols = [
        "racism_score",
        "education",
        "hh_income_pc",
        "occ_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    pretty = {
        "racism_score": "Racism score",
        "education": "Education",
        "hh_income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
    }

    # -----------------------------
    # Diagnostics to avoid silent n=0 failures
    # -----------------------------
    diag_vars = ["dislike_6_minority_associated", "dislike_12_remaining"] + xcols
    miss = []
    for c in diag_vars:
        if c not in df.columns:
            continue
        miss.append(
            {
                "var": c,
                "n": int(df.shape[0]),
                "non_missing": int(df[c].notna().sum()),
                "missing": int(df[c].isna().sum()),
                "mean": float(df[c].mean(skipna=True)) if df[c].notna().any() else np.nan,
            }
        )
    miss_df = pd.DataFrame(miss).sort_values(["missing"], ascending=False)
    miss_df.to_string(open("./output/Table2_missingness.txt", "w", encoding="utf-8"), index=False, float_format=lambda v: f"{v: .3f}")

    # -----------------------------
    # Fit models
    # -----------------------------
    modelA, tableA, fitA, idxA = fit_one(
        df=df,
        dv="dislike_6_minority_associated",
        xcols=xcols,
        model_name="Table2_Model1_dislike_minority_associated6",
        pretty=pretty,
    )
    modelB, tableB, fitB, idxB = fit_one(
        df=df,
        dv="dislike_12_remaining",
        xcols=xcols,
        model_name="Table2_Model2_dislike_other12",
        pretty=pretty,
    )

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (GSS 1993): OLS with standardized coefficients (beta = b*SD(X)/SD(Y)).\n")
        f.write("Two models with different DV dislike counts.\n\n")
        f.write("Model 1 DV: Dislike count of Rap, Reggae, Blues, Jazz, Gospel, Latin.\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(tableA.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
        f.write("\n\n")
        f.write("Model 2 DV: Dislike count of the 12 remaining genres.\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(tableB.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
        f.write("\n")

    return {
        "Model1_table": tableA,
        "Model2_table": tableB,
        "Model1_fit": fitA,
        "Model2_fit": fitB,
        "missingness": miss_df,
    }