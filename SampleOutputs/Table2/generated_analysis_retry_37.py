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

    def clean_na(series):
        """
        Conservative NA-code cleaning for this extract:
        - treat common GSS sentinels (8/9, 98/99, 998/999, 9998/9999) as missing
        - leave other values intact; per-variable validity ranges applied elsewhere
        """
        x = to_num(series).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(series):
        """
        Music taste items: valid values 1-5, where 4/5 = dislike.
        Anything outside 1-5 (including NA codes) -> missing.
        """
        x = clean_na(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_sample(s, w=None):
        """
        Weighted or unweighted z-score computed on non-missing values.
        For standardized betas, we will z-score within the analytic sample.
        """
        s = to_num(s)
        if w is None:
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True, ddof=0)
        else:
            w = to_num(w)
            ok = s.notna() & w.notna() & (w > 0)
            if ok.sum() == 0:
                return s * np.nan
            sw = w.loc[ok].astype(float)
            xs = s.loc[ok].astype(float)
            mu = np.sum(sw * xs) / np.sum(sw)
            var = np.sum(sw * (xs - mu) ** 2) / np.sum(sw)
            sd = np.sqrt(var)
        if not np.isfinite(sd) or sd == 0:
            return s * np.nan
        return (s - mu) / sd

    def build_count_completecase(df, items):
        """
        Count dislikes across items.
        Paper notes DK treated as missing and missing cases excluded.
        Implement as: require all component items non-missing for the DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

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

    def fit_table2_model(df, dv, x_order, model_name, w_col=None):
        """
        Fit OLS (optionally WLS if w_col provided), compute standardized betas (SPSS-style):
            beta_j = b_j * sd(x_j)/sd(y)
        using SDs from the analytic sample (and weights if provided).
        Output:
          - paper_style: beta + stars (computed from this model's p-values)
          - full: b, se, t, p, beta
          - fit: n, r2, adj_r2
        """
        needed = [dv] + x_order + ([w_col] if w_col else [])
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Ensure all predictors vary (do not silently drop; fail fast with diagnostics)
        zero_var = []
        for c in x_order:
            if d[c].nunique(dropna=True) <= 1:
                zero_var.append(c)
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Fix coding/sample so all listed Table 2 predictors vary."
            )

        y = to_num(d[dv]).astype(float)
        X = d[x_order].apply(to_num).astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        weights = None
        if w_col:
            weights = to_num(d[w_col]).astype(float)
            weights = weights.where(weights > 0)
            if weights.isna().any():
                # Drop rows with invalid weights
                ok = weights.notna()
                y = y.loc[ok]
                Xc = Xc.loc[ok]
                X = X.loc[ok]
                weights = weights.loc[ok]

        if w_col:
            res = sm.WLS(y, Xc, weights=weights).fit()
        else:
            res = sm.OLS(y, Xc).fit()

        # Compute standardized betas for slopes only (constant unstandardized)
        if w_col:
            y_sd = zscore_sample(y, weights).std(ddof=0)  # will be 1 by construction? no: we used zscore->sd=1; but we didn't.
            # better compute sd directly:
            y_z = zscore_sample(y, weights)
            y_sd = 1.0 if y_z.notna().any() else np.nan
            # For beta formula, we need sd(y), not 1; compute directly weighted:
            oky = y.notna() & weights.notna() & (weights > 0)
            sw = weights.loc[oky].values
            yy = y.loc[oky].values
            y_mu = np.sum(sw * yy) / np.sum(sw)
            y_var = np.sum(sw * (yy - y_mu) ** 2) / np.sum(sw)
            sd_y = float(np.sqrt(y_var)) if y_var >= 0 else np.nan
        else:
            sd_y = float(y.std(ddof=0))

        betas = {}
        for c in x_order:
            b = float(res.params.get(c, np.nan))
            if w_col:
                okx = X[c].notna() & weights.notna() & (weights > 0)
                sw = weights.loc[okx].values
                xx = X.loc[okx, c].values
                x_mu = np.sum(sw * xx) / np.sum(sw)
                x_var = np.sum(sw * (xx - x_mu) ** 2) / np.sum(sw)
                sd_x = float(np.sqrt(x_var)) if x_var >= 0 else np.nan
            else:
                sd_x = float(X[c].std(ddof=0))
            if not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (sd_x / sd_y)

        # Assemble full table
        full = pd.DataFrame(
            {
                "b_unstd": res.params,
                "std_err": res.bse,
                "t": res.tvalues,
                "p_value": res.pvalues,
            }
        )
        full.index.name = "term"
        # Add beta for terms in x_order; constant beta left blank
        full["beta_std"] = np.nan
        for c in x_order:
            if c in full.index:
                full.loc[c, "beta_std"] = betas.get(c, np.nan)

        # Paper-style table: standardized betas (slopes) + unstandardized constant
        rows = []
        for c in x_order:
            p = float(full.loc[c, "p_value"])
            rows.append(
                {
                    "term": c,
                    "coef": float(full.loc[c, "beta_std"]),
                    "stars": stars_from_p(p),
                }
            )
        # Constant: unstandardized
        p_const = float(full.loc["const", "p_value"]) if "const" in full.index else np.nan
        rows.append(
            {
                "term": "Constant",
                "coef": float(full.loc["const", "b_unstd"]) if "const" in full.index else np.nan,
                "stars": stars_from_p(p_const),
            }
        )
        paper_style = pd.DataFrame(rows).set_index("term")

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(res.nobs),
                    "k_predictors": int(res.df_model),  # excludes intercept
                    "r2": float(res.rsquared),
                    "adj_r2": float(res.rsquared_adj),
                    "weights_used": bool(w_col),
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
            f.write("\n\nNOTE: Table 2 in Bryson (1996) reports standardized coefficients only; SE/t/p here are from this re-estimation.\n")

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Standardized coefficients (beta) for slopes; constant is unstandardized.\n")
            f.write("Stars computed from this model's two-tailed p-values: * p<.05, ** p<.01, *** p<.001.\n\n")
            f.write(paper_style.to_string(float_format=lambda x: f"{x: .3f}"))

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))

        return paper_style, full, fit

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # Construct dependent variables
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # ----------------------------
    # Racism score (0-5 additive)
    # ----------------------------
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
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Controls
    # ----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    edu = clean_na(df["educ"]).where(clean_na(df["educ"]).between(0, 20))
    df["education_years"] = edu

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_na(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age"] = clean_na(df["age"]).where(clean_na(df["age"]).between(18, 89))

    # Race indicators from RACE (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not present as a clean dedicated field in provided list.
    # However, the dataset includes `ethnic`; to avoid runtime failure and allow estimation,
    # create a transparent proxy using ETHNIC if it looks like the standard GSS Hispanic code.
    # If ETHNIC coding is not Hispanic/Non-Hispanic, this should be revisited.
    if "ethnic" in df.columns:
        eth = clean_na(df["ethnic"])
        # Common GSS coding: 1 = Hispanic, 2 = Not Hispanic; treat others as missing.
        df["hispanic"] = binary_from_codes(eth, true_codes=[1], false_codes=[2])
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4 (none); keep as 0/1 with missing preserved
    df["no_religion"] = binary_from_codes(relig, true_codes=[4], false_codes=[1, 2, 3, 5])

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Fit models (Table 2 spec)
    # ----------------------------
    x_order = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
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
    for c in x_order:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # Save basic diagnostics for the key problematic field (no_religion)
    diag_cols = ["dislike_minority_genres", "dislike_other12_genres"] + x_order
    diag = df[diag_cols].copy()
    diag_path = "./output/Table2_diagnostics.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("Diagnostics (1993 only): non-missing counts and value counts for key dummies\n\n")
        f.write("Non-missing counts:\n")
        f.write(diag.notna().sum().to_string())
        f.write("\n\nValue counts (including NaN) for: no_religion, south, black, hispanic, other_race\n\n")
        for c in ["no_religion", "south", "black", "hispanic", "other_race"]:
            f.write(f"{c}:\n")
            f.write(diag[c].value_counts(dropna=False).to_string())
            f.write("\n\n")

    paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6", w_col=None
    )
    paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12", w_col=None
    )

    # Combined overview
    overview_path = "./output/Table2_overview.txt"
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("GSS 1993 Table 2 re-estimation.\n")
        f.write("Outputs:\n")
        f.write("- paper_style tables: standardized betas for slopes + unstandardized constant + stars from this model.\n")
        f.write("- full tables: unstandardized b, SE, t, p, and computed standardized betas.\n")
        f.write("\nModel A DV: dislike_minority_genres = count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin (0-6).\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislike_other12_genres = count of dislikes among remaining 12 genres (0-12).\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }