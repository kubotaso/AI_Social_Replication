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

    def clean_gss_missing(x):
        """
        Conservative GSS missing-code cleaner for this extract.
        - Many items use 8/9 (DK/NA), 98/99, 0, and sometimes 998/999.
        - We only blank out common sentinel codes (including 0) to avoid treating them as valid.
        """
        x = to_num(x).copy()
        sentinels = {0, 8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items expected 1-5.
        Dislike = 1 if 4/5; Like/neutral = 0 if 1/2/3; missing otherwise.
        """
        x = clean_gss_missing(item)
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

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def count_dislikes(df, item_cols, require_all_answered=True):
        inds = []
        for c in item_cols:
            inds.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(inds, axis=1)
        if require_all_answered:
            # Mirror "DK treated as missing and cases with missing excluded" (strict).
            return mat.sum(axis=1, min_count=len(item_cols))
        # (Not used; kept for completeness)
        return mat.sum(axis=1, min_count=1)

    def compute_standardized_betas_from_ols(y_raw, X_raw, add_const=True):
        """
        Fit OLS on raw variables, then compute standardized betas:
            beta_std_j = b_j * sd(x_j) / sd(y)
        Intercept not standardized; returned separately as raw intercept.
        """
        # statsmodels needs numeric arrays
        y = to_num(y_raw)
        X = X_raw.apply(to_num)

        if add_const:
            Xc = sm.add_constant(X, has_constant="add")
        else:
            Xc = X

        model = sm.OLS(y, Xc).fit()

        # Standard deviations over estimation sample
        sd_y = y.std(ddof=0)
        betas = {}
        for col in X.columns:
            sd_x = X[col].std(ddof=0)
            b = model.params.get(col, np.nan)
            if not np.isfinite(sd_y) or sd_y == 0 or (not np.isfinite(sd_x)) or sd_x == 0:
                betas[col] = np.nan
            else:
                betas[col] = b * (sd_x / sd_y)

        intercept = model.params.get("const", np.nan) if add_const else np.nan

        return model, betas, intercept

    def star_from_p(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def format_coef(beta, p):
        if pd.isna(beta):
            return ""
        return f"{beta:.3f}{star_from_p(p)}"

    # -----------------------
    # Load and filter
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

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

    # Strict complete-item DV construction (matches "DK treated as missing and cases excluded")
    df["dv_minority6"] = count_dislikes(df, minority_items, require_all_answered=True)
    df["dv_other12"] = count_dislikes(df, other12_items, require_all_answered=True)

    # -----------------------
    # Racism score (0-5)
    # -----------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
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
    # Controls
    # -----------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_gss_missing(df["educ"])
    df["education"] = educ.where(educ.between(0, 20))

    # Income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_pc"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_gss_missing(df["age"])
    df["age"] = age.where(age.between(18, 89))

    # Race dummies (White ref): black, other_race
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator
    # Not present in this extract. To avoid runtime errors and still keep spec "faithful",
    # we include a placeholder if no usable field exists; model will proceed without it.
    df["hispanic"] = np.nan
    hispanic_available = False
    # If a direct Hispanic flag exists in some versions of the file, use it
    for cand in ["hispanic", "hispan", "hispanp", "hispanicid", "hisp"]:
        if cand in df.columns and cand != "hispanic":
            hx = clean_gss_missing(df[cand])
            # Try a common coding: 1=yes, 2=no
            tmp = binary_from_codes(hx, true_codes=[1], false_codes=[2])
            if tmp.notna().any():
                df["hispanic"] = tmp
                hispanic_available = True
                break

    # Conservative Protestant proxy using RELIG and DENOM as provided
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index, dtype="float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index, dtype="float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern (REGION==3)
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index, dtype="float64")
    south.loc[region.isna()] = np.nan
    df["southern"] = south

    # -----------------------
    # Model fitting (listwise deletion per model)
    # -----------------------
    paper_order = [
        ("racism_score", "Racism score"),
        ("education", "Education"),
        ("hh_income_pc", "Household income per capita"),
        ("occ_prestige", "Occupational prestige"),
        ("female", "Female"),
        ("age", "Age"),
        ("black", "Black"),
        ("hispanic", "Hispanic"),
        ("other_race", "Other race"),
        ("cons_protestant", "Conservative Protestant"),
        ("no_religion", "No religion"),
        ("southern", "Southern"),
    ]

    def fit_and_tabulate(dv_col, model_label):
        rhs_cols = [c for c, _ in paper_order]

        # If Hispanic is entirely missing, drop it from estimation to avoid n=0.
        local_paper_order = paper_order.copy()
        local_rhs = rhs_cols.copy()
        if (not hispanic_available) and df["hispanic"].isna().all():
            local_paper_order = [(c, lab) for (c, lab) in local_paper_order if c != "hispanic"]
            local_rhs = [c for c in local_rhs if c != "hispanic"]

        needed = [dv_col] + local_rhs
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(local_rhs) + 2):
            # Write a diagnostic file and return an empty-but-labeled table
            diag_path = f"./output/{model_label}_diagnostic.txt"
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write(f"{model_label}\n")
                f.write(f"DV: {dv_col}\n")
                f.write(f"RHS used ({len(local_rhs)}): {local_rhs}\n")
                f.write(f"Complete-case N: {d.shape[0]}\n\n")
                f.write("Non-missing counts (before listwise deletion):\n")
                for col in needed:
                    f.write(f"{col}: {int(df[col].notna().sum())}\n")
            # Return empty structure
            out = pd.DataFrame(
                {
                    "Variable": [lab for _, lab in local_paper_order] + ["Constant"],
                    "Standardized beta (computed)": [""] * (len(local_paper_order) + 1),
                }
            )
            fit = pd.DataFrame([{"model": model_label, "n": int(d.shape[0]), "r2": np.nan, "adj_r2": np.nan}])
            return None, out, fit

        y = d[dv_col]
        X = d[local_rhs]

        # Standardize continuous & binary predictors by z-scoring (common beta computation)
        # Keep regression on raw DV so intercept is in raw units (like the paper's table).
        Xz = X.apply(zscore)

        # If any predictor becomes all-NaN due to zero variance, drop it and note
        dropped = [c for c in Xz.columns if Xz[c].isna().all()]
        Xz = Xz.drop(columns=dropped)

        # Fit and compute standardized betas in two ways:
        # - model uses z-scored predictors (so coef ~= beta if y is raw? No, then coef is in y units per SD(x))
        # - standardized beta computed post-hoc from raw model (more comparable across y scaling)
        # We'll do the post-hoc standardized beta from raw X and y, which is standard.
        model, betas, intercept = compute_standardized_betas_from_ols(y, X, add_const=True)

        # Build formatted table in paper order (excluding dropped-unavailable predictors)
        rows = []
        for col, label in local_paper_order:
            if col in dropped:
                beta = np.nan
                p = np.nan
            elif col in betas:
                beta = betas[col]
                p = model.pvalues.get(col, np.nan)
            else:
                beta = np.nan
                p = np.nan
            rows.append((label, format_coef(beta, p)))

        # Constant
        p_const = model.pvalues.get("const", np.nan)
        rows.append(("Constant", f"{intercept:.3f}{star_from_p(p_const)}" if np.isfinite(intercept) else ""))

        table = pd.DataFrame(rows, columns=["Variable", "Standardized beta (computed)"])
        fit = pd.DataFrame(
            [
                {
                    "model": model_label,
                    "n": int(model.nobs),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_after_standardization": ", ".join(dropped) if dropped else "",
                    "note": "Hispanic excluded (not available in data extract)" if (not hispanic_available and "hispanic" not in local_rhs) else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_label}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            if dropped:
                f.write("\n\nDropped predictors due to zero variance (all-NaN zscore): " + ", ".join(dropped) + "\n")
            if (not hispanic_available) and ("hispanic" not in local_rhs):
                f.write("\nNote: Hispanic dummy not estimated because no direct Hispanic identifier was available in this extract.\n")

        with open(f"./output/{model_label}_table.txt", "w", encoding="utf-8") as f:
            f.write(table.to_string(index=False))

        with open(f"./output/{model_label}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))

        return model, table, fit

    modelA, tableA, fitA = fit_and_tabulate(
        dv_col="dv_minority6",
        model_label="Table2_ModelA_Dislike_Minority_Associated6",
    )
    modelB, tableB, fitB = fit_and_tabulate(
        dv_col="dv_other12",
        model_label="Table2_ModelB_Dislike_Other12_Remaining",
    )

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Replication attempt from microdata (computed from provided 1993 GSS extract)\n")
        f.write("OLS on raw DV counts; standardized betas computed post-estimation; stars from computed p-values.\n")
        f.write("Genre dislike counts treat DK/refused/etc. as missing; DVs require all component items answered.\n")
        f.write("\nModel A DV: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: count of dislikes among the other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nTables:\n")
        f.write("\nModel A coefficients:\n")
        f.write(tableA.to_string(index=False))
        f.write("\n\nModel B coefficients:\n")
        f.write(tableB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }