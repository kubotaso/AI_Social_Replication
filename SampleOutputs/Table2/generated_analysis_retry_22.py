def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na_codes(x):
        """
        Conservative NA cleaning for typical GSS-style sentinel codes.
        We only drop a small set of common sentinels; then apply variable-specific validity checks.
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(item):
        """
        Music items are 1-5. Dislike is 4/5, like-neutral is 1/2/3.
        Anything else treated as missing.
        """
        x = clean_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_recode(series, true_codes, false_codes):
        x = clean_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count(df, items, require_all_answered=True):
        mats = []
        for c in items:
            if c not in df.columns:
                raise ValueError(f"Missing required music item column: {c}")
            mats.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(mats, axis=1)
        if require_all_answered:
            return mat.sum(axis=1, min_count=len(items))
        return mat.sum(axis=1, min_count=1)

    def standardized_betas_from_ols(model, y, X):
        """
        Compute standardized betas from an unstandardized OLS fit:
            beta_j = b_j * sd(x_j) / sd(y)
        """
        y_sd = float(np.nanstd(y, ddof=0))
        if not np.isfinite(y_sd) or y_sd == 0:
            raise ValueError("DV has zero/invalid SD; cannot compute standardized betas.")
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = float(np.nanstd(X[col], ddof=0))
            if not np.isfinite(x_sd) or x_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = float(model.params[col] * (x_sd / y_sd))
        return pd.Series(betas)

    def sig_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df_in, dv_col, x_order, model_name):
        # listwise deletion on variables in the model (DV + all predictors)
        use_cols = [dv_col] + x_order
        d = df_in[use_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={d.shape[0]}).")

        y = d[dv_col].astype(float)
        X = d[x_order].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # standardized betas (predictors only) + intercept unstandardized
        betas = standardized_betas_from_ols(model, y=y.values, X=Xc)
        out = pd.DataFrame(index=["const"] + x_order)
        out.index.name = "term"
        out["b_unstd"] = model.params.reindex(out.index)
        out["p_value"] = model.pvalues.reindex(out.index)

        out["beta_std"] = np.nan
        for c in x_order:
            out.loc[c, "beta_std"] = betas.get(c, np.nan)

        out["sig"] = out["p_value"].apply(sig_stars)

        # "paper-style" view: standardized betas + stars, and constant as unstandardized
        paper_style = pd.DataFrame(index=out.index)
        paper_style.index.name = "term"
        paper_style["coef"] = np.nan
        paper_style.loc["const", "coef"] = out.loc["const", "b_unstd"]
        paper_style.loc[x_order, "coef"] = out.loc[x_order, "beta_std"]
        paper_style["sig"] = out["sig"]

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "dv": dv_col,
                "n": int(model.nobs),
                "k_predictors": int(model.df_model),  # excludes intercept
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
            }]
        )

        # write outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE:\n")
            f.write("- Table 2 in the paper reports standardized coefficients for predictors and an unstandardized constant.\n")
            f.write("- beta_std here is computed from the unstandardized OLS as b * sd(x)/sd(y).\n")
            f.write("- sig stars are from two-tailed OLS p-values from this re-estimation (not from the PDF).\n")

        def _fmt_num(x):
            if pd.isna(x):
                return ""
            return f"{x: .3f}"

        # save paper-style table
        paper_lines = []
        paper_lines.append(f"{model_name}\n")
        paper_lines.append(f"DV: {dv_col}\n")
        paper_lines.append(f"N={int(model.nobs)}  R2={model.rsquared:.3f}  AdjR2={model.rsquared_adj:.3f}\n\n")
        paper_lines.append("term\tcoef\t(sig)\n")
        for term in paper_style.index:
            coef = paper_style.loc[term, "coef"]
            sig = paper_style.loc[term, "sig"]
            paper_lines.append(f"{term}\t{_fmt_num(coef)}\t{sig}\n")
        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.writelines(paper_lines)

        # save full technical table (includes p-values and unstd b)
        out_to_save = out.copy()
        out_to_save["b_unstd"] = out_to_save["b_unstd"].astype(float)
        out_to_save["beta_std"] = out_to_save["beta_std"].astype(float)
        out_to_save["p_value"] = out_to_save["p_value"].astype(float)
        out_to_save.to_string(
            open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8"),
            float_format=lambda v: f"{v: .6f}",
        )

        return model, paper_style, out_to_save, fit, d.index

    # -------------------------
    # Load + normalize columns
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # DVs: dislike counts
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    df["dislike_minority_genres"] = build_count(df, minority_items, require_all_answered=True)
    df["dislike_other12_genres"] = build_count(df, other12_items, require_all_answered=True)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_recode(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_recode(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_recode(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_recode(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_recode(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education (years 0-20)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column for income pc: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female (sex: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_recode(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(18, 89))
    df["age_years"] = age

    # Race dummies
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in this extract -> cannot construct faithfully.
    # To keep the model runnable and explicit, we include a placeholder but do NOT include it in the model
    # unless a usable column exists.
    hispanic_col = None
    for cand in ["hispanic", "hispan", "hisp", "ethnic_hispanic"]:
        if cand in df.columns:
            hispanic_col = cand
            break
    if hispanic_col is not None:
        # If provided, treat as 1/0 already or recode common patterns.
        h = clean_na_codes(df[hispanic_col])
        if set(h.dropna().unique()).issubset({0, 1}):
            df["hispanic"] = h.astype(float)
        else:
            # attempt: 1=yes 2=no
            df["hispanic"] = binary_recode(h, true_codes=[1], false_codes=[2])
    else:
        df["hispanic"] = np.nan  # explicit missing

    # Conservative Protestant proxy (as specified)
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot[(relig.isna()) | (denom.isna())] = np.nan
    df["cons_protestant"] = consprot

    # No religion (relig==4)
    norelig = (relig == 4).astype(float)
    norelig[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # South (region==3)
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Model specification (Table 2 order)
    # -------------------------
    x_order = [
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

    # If Hispanic is entirely missing, we cannot include it; drop it but log clearly.
    drop_notes = []
    if df["hispanic"].isna().all():
        x_order = [c for c in x_order if c != "hispanic"]
        drop_notes.append("Dropped Hispanic: no Hispanic identifier available in this dataset extract.")

    # If any dummy is constant after listwise deletion per model, statsmodels will handle it,
    # but we prefer to keep the paper spec; we do NOT auto-drop unless it's all-missing (above).
    # "no_religion dropped" previously happened due to standardizing; we do not standardize predictors for fitting now.

    # -------------------------
    # Fit models
    # -------------------------
    modelA, paperA, fullA, fitA, idxA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    modelB, paperB, fullB, fitB, idxB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS extract)\n")
        f.write("Models: OLS on unstandardized DV counts; standardized betas computed as b * sd(x)/sd(y).\n")
        f.write("Significance stars from two-tailed OLS p-values from this re-estimation.\n")
        if drop_notes:
            f.write("\nNOTES / LIMITATIONS:\n")
            for n in drop_notes:
                f.write(f"- {n}\n")
        f.write("\n\nModel A: Dislike of minority-associated genres (6)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B: Dislike of remaining genres (12)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table_paper_style": paperA,
        "ModelB_table_paper_style": paperB,
        "ModelA_table_full": fullA,
        "ModelB_table_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }