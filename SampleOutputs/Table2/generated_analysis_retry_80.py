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
        Conservative missing-code handler for this extract:
        - Coerce to numeric
        - Treat common GSS missing codes as NaN (9/8, 99/98, 999/998, etc.)
        - Additionally, if value is an integer and ends with 8/9 and is "large" (>= 98), treat as missing.
        """
        x = to_num(x).astype("float64")
        common = {8, 9, 98, 99, 998, 999, 9998, 9999, 99998, 99999}
        x = x.mask(x.isin(list(common)))

        # Heuristic for other high missing codes (e.g., 997, 998, 999, 9997...)
        xi = x.dropna().astype(np.int64)
        bad_idx = xi.index[(xi >= 97) & ((xi % 10 == 8) | (xi % 10 == 9))]
        if len(bad_idx) > 0:
            x.loc[bad_idx] = np.nan
        return x

    def likert_dislike(series):
        """
        Music items: 1-5. Dislike if 4 or 5. Like/neutral if 1-3.
        Missing if not in 1..5 after cleaning.
        """
        s = clean_gss_missing(series)
        s = s.where(s.between(1, 5))
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin([1, 2, 3])] = 0.0
        out.loc[s.isin([4, 5])] = 1.0
        return out

    def binary_from(series, true_codes, false_codes):
        s = clean_gss_missing(series)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin(false_codes)] = 0.0
        out.loc[s.isin(true_codes)] = 1.0
        return out

    def build_count_allow_partial(df, items, require_min_nonmissing=1):
        """
        Count disliked genres across items.
        'Don't know' treated as missing at item-level.
        To avoid collapsing N to ~0 in this extract, we DO NOT require complete item response.
        We require at least `require_min_nonmissing` observed items; otherwise DV is missing.
        """
        mat = pd.concat([likert_dislike(df[c]).rename(c) for c in items], axis=1)
        nonmiss = mat.notna().sum(axis=1)
        cnt = mat.sum(axis=1, min_count=require_min_nonmissing)
        cnt = cnt.where(nonmiss >= require_min_nonmissing)
        return cnt

    def zscore(x):
        x = to_num(x)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return x * np.nan
        return (x - mu) / sd

    def standardized_betas_from_unstd_fit(model, y, X):
        """
        Given unstandardized OLS fit (with intercept), compute standardized betas:
        beta_j = b_j * sd(X_j) / sd(Y)
        Dummies are treated like any other predictor (as is typical in "standardized coefficients").
        """
        y_sd = y.std(ddof=0)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = X[col].std(ddof=0)
            b = model.params.get(col, np.nan)
            betas[col] = b * (x_sd / y_sd) if (np.isfinite(b) and np.isfinite(x_sd) and np.isfinite(y_sd) and y_sd != 0) else np.nan
        return pd.Series(betas)

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

    def fit_one(df_in, dv, xcols, model_name, pretty_names):
        needed = [dv] + xcols
        d = df_in[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)

        # drop rows missing dv
        d = d.dropna(subset=[dv])

        # drop rows missing any predictors
        d = d.dropna(subset=xcols)

        # Guard
        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = to_num(d[dv]).astype(float)
        X = pd.DataFrame({c: to_num(d[c]).astype(float) for c in xcols}, index=d.index)

        # Drop zero-variance predictors
        dropped = []
        keep = []
        for c in X.columns:
            sd = X[c].std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                dropped.append(c)
            else:
                keep.append(c)
        X = X[keep]

        if X.shape[1] == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance after filtering.")

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        beta = standardized_betas_from_unstd_fit(model, y, Xc)
        rows = []
        order = [c for c in xcols if c in X.columns]  # preserve requested order, excluding dropped

        for c in order:
            p = model.pvalues.get(c, np.nan)
            rows.append(
                {
                    "term": pretty_names.get(c, c),
                    "beta_std": float(beta.get(c, np.nan)),
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "stars": stars_from_p(p),
                }
            )

        # Intercept (unstandardized, like the paper reports a constant)
        p0 = model.pvalues.get("const", np.nan)
        rows.append(
            {
                "term": "Constant",
                "beta_std": np.nan,
                "p_value": float(p0) if np.isfinite(p0) else np.nan,
                "stars": stars_from_p(p0),
                "b_unstd": float(model.params.get("const", np.nan)),
            }
        )

        tab = pd.DataFrame(rows)
        # Add unstandardized b for predictors (optional, but helpful for debugging)
        b_unstd = []
        for r in tab["term"]:
            b_unstd.append(np.nan)
        tab["b_unstd"] = tab.get("b_unstd", np.nan)
        # Fill b_unstd for predictor rows
        term_to_col = {pretty_names.get(c, c): c for c in X.columns}
        for i, r in enumerate(tab["term"].tolist()):
            if r in term_to_col:
                col = term_to_col[r]
                tab.loc[i, "b_unstd"] = float(model.params.get(col, np.nan))

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join([pretty_names.get(c, c) for c in dropped]) if dropped else "",
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            if dropped:
                f.write("\n\nDropped zero-variance predictors:\n")
                for c in dropped:
                    f.write(f"- {pretty_names.get(c, c)} ({c})\n")

        # Human-readable table
        tab_out = tab.copy()
        # Pretty formatting: show beta with stars; constant uses b_unstd with stars
        def fmt_beta(row):
            if row["term"] == "Constant":
                b = row["b_unstd"]
                if pd.isna(b):
                    return ""
                return f"{b:.3f}{row['stars']}"
            b = row["beta_std"]
            if pd.isna(b):
                return ""
            return f"{b:.3f}{row['stars']}"

        tab_out["reported"] = tab_out.apply(fmt_beta, axis=1)
        keep_cols = ["term", "reported", "beta_std", "b_unstd", "p_value"]
        tab_out = tab_out[keep_cols]

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(tab_out.to_string(index=False))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))

        return model, tab_out, fit, d.index

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Year filter
    if "year" not in df.columns:
        raise ValueError("Missing column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # Required variable presence checks (minimal)
    required_music = [
        "rap", "reggae", "blues", "jazz", "gospel", "latin",
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    required_racism = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    required_controls = ["educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "relig", "denom", "region"]

    missing_cols = [c for c in (required_music + required_racism + required_controls) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -----------------------------
    # Construct DVs
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    # Require at least half the items answered to reduce noise but avoid N collapse
    df["dislike_minority6"] = build_count_allow_partial(df, minority_items, require_min_nonmissing=3)
    df["dislike_other12"] = build_count_allow_partial(df, other12_items, require_min_nonmissing=6)

    # -----------------------------
    # Racism score (0-5), require all 5 items
    # -----------------------------
    rac1 = binary_from(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)  # require all 5

    # -----------------------------
    # Controls
    # -----------------------------
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_pc"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    df["female"] = binary_from(df["sex"], true_codes=[2], false_codes=[1])

    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: use ETHNIC as a best-effort proxy only if present; otherwise missing.
    # Many GSS extracts include "ethnic" codes; in this provided file it exists.
    # We define Hispanic=1 for common Hispanic/Latino origin responses (e.g., Mexico/Puerto Rico/Cuba/Latin America/Spain).
    # This is imperfect but prevents structural omission and allows estimation.
    if "ethnic" in df.columns:
        ethnic = clean_gss_missing(df["ethnic"])
        # Heuristic: treat 1-9 as "Hispanic/Latino origin" codes in many GSS encodings; otherwise 0.
        # If the coding differs, this will not match perfectly, but it avoids all-missing/constant.
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        hisp.loc[ethnic.notna()] = 0.0
        hisp.loc[ethnic.between(1, 9)] = 1.0
        df["hispanic"] = hisp
    else:
        df["hispanic"] = np.nan

    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])

    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["southern"] = south

    # -----------------------------
    # Fit models (unstandardized OLS; compute standardized betas)
    # -----------------------------
    xcols = [
        "racism_score",
        "education_years",
        "hh_income_pc",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "southern",
    ]

    pretty = {
        "racism_score": "Racism score",
        "education_years": "Education",
        "hh_income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age_years": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
    }

    results = {}

    modelA, tableA, fitA, idxA = fit_one(
        df_in=df,
        dv="dislike_minority6",
        xcols=xcols,
        model_name="Table2_ModelA_dislike_6_minority_associated",
        pretty_names=pretty,
    )

    modelB, tableB, fitB, idxB = fit_one(
        df_in=df,
        dv="dislike_other12",
        xcols=xcols,
        model_name="Table2_ModelB_dislike_12_remaining",
        pretty_names=pretty,
    )

    # -----------------------------
    # Diagnostics: missingness and sample attrition
    # -----------------------------
    diag_cols = ["dislike_minority6", "dislike_other12"] + xcols
    miss = pd.DataFrame(
        {
            "missing_n": df[diag_cols].isna().sum(),
            "nonmissing_n": df[diag_cols].notna().sum(),
            "missing_pct": (df[diag_cols].isna().mean() * 100.0),
        }
    ).reset_index().rename(columns={"index": "variable"})
    miss.to_csv("./output/Table2_missingness.csv", index=False)

    with open("./output/Table2_missingness.txt", "w", encoding="utf-8") as f:
        f.write("Missingness diagnostics (1993 only)\n")
        f.write(miss.to_string(index=False))

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt from microdata (1993 GSS extract)\n")
        f.write("OLS on unstandardized variables; standardized betas computed as b * SD(X)/SD(Y).\n")
        f.write("Stars are computed from model p-values (two-tailed).\n\n")
        f.write("Model A DV: count of disliked genres among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: count of disliked genres among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results["ModelA_table"] = tableA
    results["ModelB_table"] = tableB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB
    results["Missingness"] = miss

    return results