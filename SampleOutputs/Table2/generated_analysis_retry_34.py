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

    def clean_na(series):
        """
        Conservative missing-code cleaner for this extract:
        - coerces to numeric
        - sets common GSS sentinels to NaN
        """
        x = to_num(series)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        return x.mask(x.isin(sentinels))

    def likert_dislike_indicator(series):
        """
        Music taste items are 1-5; 4/5 = dislike.
        Missing if outside 1-5 after cleaning.
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

    def build_dislike_count(df, items):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        # Require all items observed (DK treated as missing; exclude cases with any missing)
        return mat.sum(axis=1, min_count=len(items))

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

    def weighted_mean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return np.sum(w[m] * x[m]) / np.sum(w[m])

    def weighted_std(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        mu = np.sum(w[m] * x[m]) / np.sum(w[m])
        var = np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])
        return np.sqrt(var)

    def fit_table2_model(df, dv, x_order, model_name, w_col=None):
        """
        Fit OLS/WLS with:
        - unstandardized coefficients (b) and p-values
        - standardized betas computed as b * sd(x)/sd(y) using (weighted) SDs on analytic sample
        Output:
        - paper_style: beta with stars, ordered like Table 2 (+ constant last)
        - full: b, se, t, p, beta
        - fit: n, r2, adj_r2, k_predictors
        """
        needed = [dv] + x_order
        if w_col is not None:
            needed = needed + [w_col]
        d = df[needed].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Ensure predictors all vary (do not silently drop)
        zero_var = []
        for c in x_order:
            v = d[c]
            if v.nunique(dropna=True) <= 1:
                zero_var.append(c)
        if zero_var:
            # Save diagnostic and proceed by dropping them (so code runs), but record clearly.
            diag_path = f"./output/{model_name}_diagnostic_zero_variance.txt"
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write(f"Zero-variance predictors in analytic sample for {model_name}:\n")
                for c in zero_var:
                    f.write(f"- {c}\n")
                f.write("\nThese were dropped to allow estimation.\n")
            x_used = [c for c in x_order if c not in zero_var]
        else:
            x_used = list(x_order)

        if d.shape[0] < (len(x_used) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_used)}).")

        y = d[dv].astype(float)
        X = d[x_used].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        if w_col is not None:
            w = d[w_col].astype(float)
            model = sm.WLS(y, Xc, weights=w).fit()
        else:
            w = None
            model = sm.OLS(y, Xc).fit()

        # Standardized betas (do NOT standardize the constant)
        if w is None:
            y_sd = float(np.nanstd(y.values, ddof=0))
            x_sd = {c: float(np.nanstd(X[c].values, ddof=0)) for c in X.columns}
        else:
            y_sd = float(weighted_std(y.values, w.values))
            x_sd = {c: float(weighted_std(X[c].values, w.values)) for c in X.columns}

        betas = {}
        for term in model.params.index:
            if term == "const":
                betas[term] = np.nan
            else:
                sd_x = x_sd.get(term, np.nan)
                if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(y_sd) or y_sd == 0:
                    betas[term] = np.nan
                else:
                    betas[term] = float(model.params[term] * (sd_x / y_sd))

        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": pd.Series(betas),
            }
        )
        full.index.name = "term"

        # Paper-style: standardized betas + stars; include constant unstandardized (like paper)
        # Keep the requested x_order display order even if some were dropped; show NaN for dropped.
        rows = []
        for c in x_order:
            if c in full.index:
                b = full.loc[c, "beta_std"]
                p = full.loc[c, "p_value"]
                rows.append((c, b, stars(p)))
            else:
                rows.append((c, np.nan, ""))
        # Constant
        rows.append(("const", float(full.loc["const", "b_unstd"]), stars(float(full.loc["const", "p_value"]))))

        paper_style = pd.DataFrame(rows, columns=["term", "coef", "stars"]).set_index("term")
        paper_style["coef_star"] = paper_style["coef"].map(lambda v: "" if not np.isfinite(v) else f"{v:.3f}") + paper_style["stars"]

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "weights_used": bool(w_col is not None),
                    "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        with open(f"./output/{model_name}_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Standardized OLS coefficients (beta) with stars; constant shown unstandardized.\n")
            f.write(paper_style[["coef", "stars", "coef_star"]].to_string())
            f.write("\n")

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write("Unstandardized coefficients with SE/t/p and computed standardized betas.\n")
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        return model, paper_style, full, fit

    # -----------------------
    # Load and filter to 1993
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Dependent variables
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items)

    # -----------------------
    # Racism score (0-5)
    # -----------------------
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

    # -----------------------
    # Controls (Table 2 RHS)
    # -----------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing 'educ' column.")
    educ = clean_na(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing 'realinc' or 'hompop' column.")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing 'prestg80' column.")
    df["occ_prestige"] = clean_na(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing 'sex' column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing 'age' column.")
    age = clean_na(df["age"])
    # Keep broad valid range; do not impose 18+ (paper uses adults, but keep as recorded to avoid dropping)
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing 'race' column.")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic indicator:
    # Not directly available per mapping instruction. We create it as all-missing and allow it to be dropped by listwise deletion
    # only if included; to keep the model estimable, we include it only if it has variation.
    df["hispanic"] = np.nan

    # Conservative Protestant (proxy): RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing 'relig' or 'denom' column.")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing 'region' column.")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------
    # Model specification (Table 2 order)
    # Include hispanic only if it exists with variation (else keep out to avoid killing sample)
    # -----------------------
    x_order_base = [
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

    # If hispanic is entirely missing, drop it from estimation order (but keep a note)
    if df["hispanic"].notna().sum() == 0:
        x_order = [c for c in x_order_base if c != "hispanic"]
        hisp_note = "Hispanic indicator not available in provided extract; omitted from estimation."
    else:
        x_order = x_order_base
        hisp_note = ""

    # -----------------------
    # Fit models
    # -----------------------
    mA, paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6", w_col=None
    )
    mB, paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12", w_col=None
    )

    # -----------------------
    # Overview file
    # -----------------------
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): Standardized OLS coefficients (computed from microdata).\n")
        f.write("DVs:\n")
        f.write("- Model A: count of dislikes among {Rap, Reggae, Blues, Jazz, Gospel, Latin}\n")
        f.write("- Model B: count of dislikes among {Bigband, Blugrass, Country, Musicals, Classicl, Folk, Moodeasy, Newage, Opera, Conrock, Oldies, Hvymetal}\n\n")
        if hisp_note:
            f.write("NOTE: " + hisp_note + "\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel A (paper-style coefficients):\n")
        f.write(paperA[["coef", "stars", "coef_star"]].to_string())
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nModel B (paper-style coefficients):\n")
        f.write(paperB[["coef", "stars", "coef_star"]].to_string())
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }