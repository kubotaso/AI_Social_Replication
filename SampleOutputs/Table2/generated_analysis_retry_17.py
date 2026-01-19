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

    def clean_missing_like_gss(x):
        """
        Conservative missing-code handling for this extract.
        - Treat common GSS sentinel codes as missing.
        - Leave substantive values intact.
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5.
        Dislike = 4 or 5. Like/neutral = 1,2,3.
        Non 1-5 -> missing.
        """
        x = clean_missing_like_gss(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_missing_like_gss(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(x):
        x = to_num(x)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        return (x - mu) / sd

    def build_count_allow_partial(df, items, min_nonmissing=1):
        """
        Count of disliked genres across a set of items.
        - Each item: 1 if dislike (4/5), 0 if 1/2/3, missing otherwise.
        - DV is the sum across available items, requiring at least min_nonmissing observed items.
        This is intentionally less strict than requiring complete responses on all items to avoid
        collapsing N (which was a major issue in the earlier attempt).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        nn = mat.notna().sum(axis=1)
        y = mat.sum(axis=1, min_count=min_nonmissing)
        y = y.mask(nn < min_nonmissing)
        return y

    def stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def beta_from_unstd(model, y, X_no_const):
        """
        Standardized beta = b_j * sd(x_j) / sd(y), using the estimation sample.
        Intercept is not standardized (left as unstandardized).
        """
        sd_y = np.std(y, ddof=0)
        betas = {}
        for col in X_no_const.columns:
            sd_x = np.std(X_no_const[col], ddof=0)
            b = model.params.get(col, np.nan)
            if sd_y == 0 or sd_x == 0 or not np.isfinite(sd_y) or not np.isfinite(sd_x):
                betas[col] = np.nan
            else:
                betas[col] = b * (sd_x / sd_y)
        return betas

    def fit_table2_style(df, dv, xcols, model_name, pretty_names, include_hispanic_if_available=True):
        """
        OLS on original metric; report standardized betas + stars, and unstandardized constant.
        Listwise deletion across dv and xcols.
        """
        use_cols = [dv] + xcols
        d = df[use_cols].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Ensure we have enough cases
        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)})")

        y = to_num(d[dv]).astype(float)
        X = d[xcols].apply(to_num).astype(float)

        # Drop any constant predictors (rare, but prevents singularities)
        nunique = X.nunique(dropna=True)
        constant_cols = [c for c in X.columns if nunique.get(c, 0) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            xcols_eff = [c for c in xcols if c not in constant_cols]
        else:
            xcols_eff = list(xcols)

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas computed post hoc (matches common meaning of "standardized coefficients")
        betas = beta_from_unstd(model, y.values, X)

        # Build Table 2-style output: standardized betas for predictors, constant unstandardized
        rows = []

        for c in xcols:
            if c in constant_cols:
                beta = np.nan
                p = np.nan
                st = ""
            elif c in model.params.index:
                beta = betas.get(c, np.nan)
                p = model.pvalues.get(c, np.nan)
                st = stars_from_p(p)
            else:
                beta = np.nan
                p = np.nan
                st = ""
            rows.append(
                {
                    "term": pretty_names.get(c, c),
                    "std_beta": beta,
                    "sig": st,
                }
            )

        # Constant row (unstandardized)
        rows.append(
            {
                "term": "Constant",
                "std_beta": model.params.get("const", np.nan),  # store constant here per paper-style
                "sig": stars_from_p(model.pvalues.get("const", np.nan)),
            }
        )

        out = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_constant_predictors": ", ".join(constant_cols) if constant_cols else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Table 2 in the paper reports standardized coefficients only; SEs are not printed there.\n")
            f.write("This file includes full OLS output from the replication estimation.\n")

        with open(f"./output/{model_name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv}\n\n")
            f.write(out.to_string(index=False, float_format=lambda v: "   " if pd.isna(v) else f"{v: .3f}"))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False, float_format=lambda v: f"{v: .3f}" if isinstance(v, float) else str(v)))
            f.write("\n")

        return out, fit, model

    def missingness_report(df, cols, path):
        rep = []
        for c in cols:
            s = df[c]
            rep.append(
                {
                    "var": c,
                    "nonmissing": int(s.notna().sum()),
                    "missing": int(s.isna().sum()),
                    "unique_nonmissing": int(s.nunique(dropna=True)),
                }
            )
        rep = pd.DataFrame(rep).sort_values(["missing", "var"], ascending=[False, True])
        rep.to_csv(path, index=False)
        return rep

    # -----------------------------
    # Load and filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

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

    # Allow partial responses to prevent severe N collapse (paper: DK treated as missing; missing cases excluded;
    # our earlier approach required all items and cut N ~ in half).
    df["dislike_minority_genres"] = build_count_allow_partial(df, minority_items, min_nonmissing=1)
    df["dislike_other12_genres"] = build_count_allow_partial(df, other12_items, min_nonmissing=1)

    # -----------------------------
    # Racism score (0-5 additive, listwise across components)
    # -----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)  # require all 5 present

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ")
    educ = clean_missing_like_gss(df["educ"]).where(clean_missing_like_gss(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing realinc and/or hompop")
    realinc = clean_missing_like_gss(df["realinc"])
    hompop = clean_missing_like_gss(df["hompop"]).where(clean_missing_like_gss(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80")
    df["occ_prestige"] = clean_missing_like_gss(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age")
    df["age_years"] = clean_missing_like_gss(df["age"]).where(clean_missing_like_gss(df["age"]).between(18, 89))

    # Race dummies (white ref)
    if "race" not in df.columns:
        raise ValueError("Missing race")
    race = clean_missing_like_gss(df["race"]).where(clean_missing_like_gss(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.notna(), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.notna(), (race == 3).astype(float), np.nan)

    # Hispanic: not available in provided variables (per instruction); keep as missing and exclude from model
    # (We still create the column for transparency.)
    df["hispanic"] = np.nan

    # Conservative Protestant proxy
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing relig and/or denom")
    relig = clean_missing_like_gss(df["relig"])
    denom = clean_missing_like_gss(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern
    if "region" not in df.columns:
        raise ValueError("Missing region")
    region = clean_missing_like_gss(df["region"]).where(clean_missing_like_gss(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.notna(), (region == 3).astype(float), np.nan)

    # -----------------------------
    # Model spec (Table 2 RHS)
    # NOTE: Hispanic cannot be included because it is not available here.
    # We keep the rest and explicitly report that omission.
    # -----------------------------
    xcols = [
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

    pretty = {
        "racism_score": "Racism score",
        "education_years": "Education (years)",
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

    # Missingness reports (helps debug N shortfalls)
    all_model_cols_A = ["dislike_minority_genres"] + xcols
    all_model_cols_B = ["dislike_other12_genres"] + xcols
    missingness_report(df, all_model_cols_A, "./output/Table2_missingness_modelA.csv")
    missingness_report(df, all_model_cols_B, "./output/Table2_missingness_modelB.csv")

    # Fit models
    tabA, fitA, modelA = fit_table2_style(
        df=df,
        dv="dislike_minority_genres",
        xcols=xcols,
        model_name="Table2_ModelA_dislike_minority6",
        pretty_names=pretty,
    )
    tabB, fitB, modelB = fit_table2_style(
        df=df,
        dv="dislike_other12_genres",
        xcols=xcols,
        model_name="Table2_ModelB_dislike_other12",
        pretty_names=pretty,
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Replication output for Bryson Table 2 (GSS 1993).\n")
        f.write("Main table files are *_table2_style.txt: standardized betas + stars; constant is unstandardized.\n")
        f.write("\nIMPORTANT LIMITATION:\n")
        f.write("- Hispanic indicator is not present in the provided extract, so it cannot be included.\n")
        f.write("- N will differ from the paper if the extract differs and/or if missing-data patterns differ.\n\n")
        f.write("Model A: DV = count of disliked among Rap, Reggae, Blues, Jazz, Gospel, Latin.\n")
        f.write(fitA.to_string(index=False, float_format=lambda v: f"{v: .3f}" if isinstance(v, float) else str(v)))
        f.write("\n\nModel B: DV = count of disliked among the other 12 genres.\n")
        f.write(fitB.to_string(index=False, float_format=lambda v: f"{v: .3f}" if isinstance(v, float) else str(v)))
        f.write("\n")

    return {
        "ModelA_table2_style": tabA,
        "ModelB_table2_style": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }