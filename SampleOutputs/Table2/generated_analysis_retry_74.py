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

    def clean_na_codes(x):
        """
        Conservative NA handling for common GSS-style sentinels.
        We DO NOT blanket-drop small integers (like 8/9) for all variables because
        those can be valid codes for some items. Here we only remove obvious large
        sentinels and allow per-variable domain checks later.
        """
        x = to_num(x).copy()
        x = x.mask(x.isin([98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 scale; 4/5 = dislike; 1/2/3 = not dislike.
        Anything outside 1..5 treated as missing.
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

    def build_count_complete(df, items):
        """
        Paper: DK treated as missing; cases with missing excluded.
        Implement DV as complete-case across ALL component items.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardize_betas_from_unstandardized_fit(model, y, X):
        """
        Compute standardized betas from an OLS fit on raw variables:
            beta_j = b_j * sd(x_j) / sd(y)
        Intercept is not standardized (kept separately as unstandardized constant).
        """
        y_sd = float(np.std(y, ddof=0))
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = float(np.std(X[col], ddof=0))
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = float(model.params[col]) * (x_sd / y_sd)
        return betas

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

    def missingness_report(df, cols):
        rep = []
        for c in cols:
            rep.append(
                {
                    "var": c,
                    "n_nonmissing": int(df[c].notna().sum()),
                    "n_missing": int(df[c].isna().sum()),
                    "mean": float(df[c].mean(skipna=True)) if df[c].notna().any() else np.nan,
                    "sd": float(df[c].std(skipna=True, ddof=0)) if df[c].notna().any() else np.nan,
                }
            )
        return pd.DataFrame(rep)

    def fit_table2_model(df, dv, xcols, model_name, pretty_names):
        """
        Listwise deletion on DV + RHS only, fit unstandardized OLS, then compute
        standardized coefficients and stars from the same fit (two-tailed).
        Output: standardized betas + stars, and intercept (unstd) + stars.
        """
        needed = [dv] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Build X, y
        y = to_num(d[dv]).astype(float)
        X = pd.DataFrame({c: to_num(d[c]).astype(float) for c in xcols}, index=d.index)

        # Drop any constant predictors (avoid singular matrix)
        keep = []
        dropped_const = []
        for c in X.columns:
            sd = float(np.std(X[c], ddof=0))
            if np.isfinite(sd) and sd > 0:
                keep.append(c)
            else:
                dropped_const.append(c)
        X = X[keep]

        Xc = sm.add_constant(X, has_constant="add")

        # Fit
        model = sm.OLS(y, Xc).fit()

        # Standardized betas
        betas = standardize_betas_from_unstandardized_fit(model, y.values, Xc)

        # Assemble output in the paper's row order, including rows that are missing in data
        rows = []
        for raw_name in xcols:
            pname = pretty_names.get(raw_name, raw_name)
            if raw_name in betas:
                beta = betas[raw_name]
                p = float(model.pvalues.get(raw_name, np.nan))
            else:
                beta = np.nan
                p = np.nan
            rows.append(
                {
                    "term": pname,
                    "beta_std": beta,
                    "stars": stars_from_p(p),
                }
            )

        # Constant row (unstandardized)
        p_const = float(model.pvalues.get("const", np.nan))
        rows.append(
            {
                "term": "Constant",
                "beta_std": float(model.params.get("const", np.nan)),  # stored as constant value (unstd)
                "stars": stars_from_p(p_const),
            }
        )

        out = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_constant_predictors": ", ".join(dropped_const) if dropped_const else "",
                }
            ]
        )

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTES:\n")
            f.write("- Standardized coefficients computed as b * SD(X)/SD(Y) from unstandardized OLS.\n")
            f.write("- Stars computed from two-tailed OLS p-values: * p<.05, ** p<.01, *** p<.001.\n")
            if dropped_const:
                f.write(f"- Dropped constant predictors (no variance in estimation sample): {', '.join(dropped_const)}\n")

        # Human-readable table (paper style: beta + stars only)
        tab_disp = out.copy()
        tab_disp["beta_disp"] = tab_disp["beta_std"].map(lambda v: "" if not np.isfinite(v) else f"{v: .3f}") + tab_disp["stars"]
        tab_disp = tab_disp[["term", "beta_disp"]]
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(tab_disp.to_string(index=False))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        return out, fit, model

    # -------------------------
    # Load & filter year
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # DVs: dislike counts
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete(df, other12_items)

    # -------------------------
    # Racism score (0-5): complete-case across 5 items
    # -------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racmat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racmat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing educ")
    educ = clean_na_codes(df["educ"])
    df["education"] = educ.where(educ.between(0, 20))

    # Household income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing realinc and/or hompop")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(lambda s: s > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80")
    df["occupational_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age")
    age = clean_na_codes(df["age"])
    # Keep broad plausible adult range; do not over-restrict
    df["age"] = age.where(age.between(18, 89))

    # Race dummies: Black and Other race from RACE
    if "race" not in df.columns:
        raise ValueError("Missing race")
    race = clean_na_codes(df["race"]).where(lambda s: s.isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: try to construct from 'ethnic' if present (best-effort), else missing.
    # The earlier note cautioned against ETHNIC as a perfect proxy, but Table 2 requires
    # a Hispanic indicator; we implement a transparent, documented heuristic.
    #
    # Heuristic: treat ETHNIC codes that clearly correspond to Hispanic origins as 1.
    # If the coding in this extract differs, this will be imperfect but avoids dropping the row.
    hisp = pd.Series(np.nan, index=df.index, dtype="float64")
    if "ethnic" in df.columns:
        e = clean_na_codes(df["ethnic"])
        # Common GSS ETHNIC (older coding): 20-29 often Hispanic groups (Mexican, Puerto Rican, etc.)
        # Keep narrowly focused to reduce misclassification.
        hisp_val = e.where(e.between(20, 29))
        hisp = pd.Series(0.0, index=df.index, dtype="float64")
        hisp.loc[hisp_val.notna()] = 1.0
        hisp.loc[e.isna()] = np.nan
    df["hispanic"] = hisp

    # Religion: No religion from RELIG==4 (None)
    if "relig" not in df.columns:
        raise ValueError("Missing relig")
    relig = clean_na_codes(df["relig"])
    df["no_religion"] = binary_from_codes(relig, true_codes=[4], false_codes=[1, 2, 3, 5, 6, 7, 8, 9])

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "denom" not in df.columns:
        raise ValueError("Missing denom")
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["conservative_protestant"] = consprot

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region")
    region = clean_na_codes(df["region"]).where(lambda s: s.isin([1, 2, 3, 4]))
    df["southern"] = (region == 3).astype(float)
    df.loc[region.isna(), "southern"] = np.nan

    # -------------------------
    # Prepare model column order (paper order)
    # -------------------------
    xcols = [
        "racism_score",
        "education",
        "hh_income_per_capita",
        "occupational_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "conservative_protestant",
        "no_religion",
        "southern",
    ]

    pretty = {
        "racism_score": "Racism score",
        "education": "Education",
        "hh_income_per_capita": "Household income per capita",
        "occupational_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "conservative_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    # -------------------------
    # Diagnostics: missingness (pre-model)
    # -------------------------
    diag_cols = [
        "dislike_minority_genres",
        "dislike_other12_genres",
    ] + xcols
    diag = missingness_report(df, [c for c in diag_cols if c in df.columns])
    diag_path = "./output/Table2_missingness_1993.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("Missingness/summary diagnostics (1993 only; before listwise deletion)\n\n")
        f.write(diag.to_string(index=False))
        f.write("\n")

    # -------------------------
    # Fit models
    # -------------------------
    tabA, fitA, modelA = fit_table2_model(
        df=df,
        dv="dislike_minority_genres",
        xcols=xcols,
        model_name="Table2_ModelA_dislike_minority6",
        pretty_names=pretty,
    )
    tabB, fitB, modelB = fit_table2_model(
        df=df,
        dv="dislike_other12_genres",
        xcols=xcols,
        model_name="Table2_ModelB_dislike_other12",
        pretty_names=pretty,
    )

    # Combined overview
    overview_path = "./output/Table2_overview.txt"
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): standardized OLS coefficients (beta)\n")
        f.write("Standardized betas computed from unstandardized OLS: beta = b * SD(X)/SD(Y)\n")
        f.write("Stars computed from two-tailed OLS p-values (* p<.05, ** p<.01, *** p<.001).\n")
        f.write("\nModel A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres (12 remaining genres)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "Diagnostics_missingness": diag,
    }