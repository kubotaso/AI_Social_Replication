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
        Conservative NA-code handling for this extract.
        Treat common GSS sentinels as missing. Do NOT over-aggressively drop valid values.
        """
        x = to_num(x).copy()
        na_vals = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(na_vals)))
        return x

    def likert_dislike_indicator(item):
        """
        1-5: 4/5 = dislike; 1/2/3 = not-dislike. Missing otherwise.
        """
        x = clean_na_codes(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na_codes(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore_series(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def build_count_allow_partial(df, items, min_answered):
        """
        Build count of dislikes, allowing partial item nonresponse.
        Count is computed if at least `min_answered` items are observed.
        Then rescale to the full-item count via: count * (K / answered),
        to reduce sample loss while keeping the dependent variable on the count scale.
        """
        K = len(items)
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        answered = mat.notna().sum(axis=1)
        disliked = mat.sum(axis=1, min_count=1)

        # require enough items answered
        disliked = disliked.where(answered >= min_answered)

        # rescale to full count range
        scaled = disliked * (K / answered.replace(0, np.nan))
        return scaled

    def standardized_betas_and_stars(model):
        """
        model is an OLS fit on unstandardized y and X with intercept.
        Compute standardized betas as b * sd(x)/sd(y), excluding intercept.
        Stars are based on model p-values (two-tailed): * <.05, ** <.01, *** <.001
        """
        params = model.params.copy()
        pvals = model.pvalues.copy()

        # collect SDs used for standardization on estimation sample
        y = model.model.endog
        X = model.model.exog
        exog_names = list(model.model.exog_names)

        y_sd = np.std(y, ddof=0)
        betas = {}
        for j, name in enumerate(exog_names):
            if name == "const":
                continue
            x_sd = np.std(X[:, j], ddof=0)
            betas[name] = params[name] * (x_sd / y_sd) if y_sd != 0 else np.nan

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

        rows = []
        for name in exog_names:
            if name == "const":
                rows.append(
                    {
                        "term": "Constant",
                        "beta_std": np.nan,
                        "b_unstd": float(params[name]),
                        "stars": "",  # do not star constant in "paper-match" table
                        "p_value_replication": float(pvals[name]) if np.isfinite(pvals[name]) else np.nan,
                    }
                )
            else:
                rows.append(
                    {
                        "term": name,
                        "beta_std": float(betas.get(name, np.nan)),
                        "b_unstd": float(params[name]),
                        "stars": star(pvals[name]),
                        "p_value_replication": float(pvals[name]) if np.isfinite(pvals[name]) else np.nan,
                    }
                )
        out = pd.DataFrame(rows).set_index("term")
        return out

    def fit_model(df, y_col, x_cols, model_label):
        """
        OLS on unstandardized variables, then compute standardized betas post-hoc.
        Listwise deletion on y and x only.
        """
        d = df[[y_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if d.shape[0] < (len(x_cols) + 5):
            raise ValueError(f"{model_label}: insufficient complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = to_num(d[y_col])
        X = d[x_cols].apply(to_num)

        # Drop zero-variance predictors (after listwise deletion) but record them
        dropped = []
        keep = []
        for c in X.columns:
            sd = X[c].std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                dropped.append(c)
            else:
                keep.append(c)
        X = X[keep]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Build Table 2-style output: standardized betas + stars; keep b for debugging/replication
        coef_table = standardized_betas_and_stars(model)

        fit = pd.DataFrame(
            [
                {
                    "model": model_label,
                    "dv": y_col,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance_predictors": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_label}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE:\n")
            f.write("- Table 2 in the paper reports standardized coefficients only and does not publish SEs.\n")
            f.write("- p-values here are computed from the replication OLS and are used only to add stars.\n")
            if dropped:
                f.write(f"- Dropped zero-variance predictors after listwise deletion: {', '.join(dropped)}\n")

        with open(f"./output/{model_label}_table.txt", "w", encoding="utf-8") as f:
            f.write("Replication output (computed from microdata)\n")
            f.write("beta_std = standardized coefficient (computed post-hoc as b * sd(x)/sd(y))\n")
            f.write("stars based on replication p-values: * p<.05, ** p<.01, *** p<.001\n\n")
            f.write(coef_table.to_string(float_format=lambda v: f"{v: .6f}"))
            f.write("\n\n")
            f.write(fit.to_string(index=False))

        return coef_table, fit, d.index

    # -------------------------
    # Load and filter year
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables (counts of disliked genres)
    # Allow partial responses to reduce excessive N loss.
    # -------------------------
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

    # Require most items answered (tunable): 5/6 and 10/12
    df["dislike_minority_genres"] = build_count_allow_partial(df, minority_items, min_answered=5)
    df["dislike_other12_genres"] = build_count_allow_partial(df, other12_items, min_answered=10)

    # -------------------------
    # Racism scale (0-5 additive index; complete on all 5 items)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)  # require all 5 items

    # -------------------------
    # RHS predictors (coded per mapping)
    # -------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_na_codes(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Income per capita
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
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

    # Race dummies (White reference): black and other_race from RACE
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.notna(), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.notna(), (race == 3).astype(float), np.nan)

    # Hispanic dummy not available in provided variables -> omit from model (cannot proxy)
    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(relig.notna() & denom.notna())
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(relig.notna())
    df["no_religion"] = norelig

    # Southern
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.notna(), (region == 3).astype(float), np.nan)

    # -------------------------
    # Models (Table 2 RHS; excluding hispanic due to missing field)
    # Keep names stable and table rows labeled.
    # -------------------------
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

    # Ensure predictors exist
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # Fit both models
    tabA, fitA, idxA = fit_model(df, "dislike_minority_genres", x_cols, "Table2_ModelA_Minority6")
    tabB, fitB, idxB = fit_model(df, "dislike_other12_genres", x_cols, "Table2_ModelB_Other12")

    # Missingness audit to help diagnose N mismatches vs paper
    audit_cols = ["dislike_minority_genres", "dislike_other12_genres"] + x_cols
    missing_audit = pd.DataFrame(
        {
            "missing_n": df[audit_cols].isna().sum(),
            "nonmissing_n": df[audit_cols].notna().sum(),
        }
    ).sort_values(["missing_n", "nonmissing_n"], ascending=[False, True])

    with open("./output/Table2_missingness_audit.txt", "w", encoding="utf-8") as f:
        f.write("Missingness audit (1993 only). Counts of missing/non-missing per analysis variable.\n\n")
        f.write(missing_audit.to_string())

    # Produce "paper-match style" tables: standardized betas + stars only (no SEs)
    def paper_style(tab, title):
        # Keep only standardized betas and stars for predictors; keep constant separately (unstandardized)
        out = tab.copy()
        # Rename rows to human-friendly labels
        rename_map = {
            "racism_score": "Racism score",
            "education_years": "Education (years)",
            "hh_income_per_capita": "Household income per capita",
            "occ_prestige": "Occupational prestige",
            "female": "Female",
            "age_years": "Age",
            "black": "Black",
            "other_race": "Other race",
            "cons_protestant": "Conservative Protestant",
            "no_religion": "No religion",
            "south": "Southern",
            "Constant": "Constant",
        }
        out.index = [rename_map.get(i, i) for i in out.index]

        # Split: predictors vs constant
        predictors = out.loc[[i for i in out.index if i != "Constant"], ["beta_std", "stars"]].copy()
        const = out.loc[["Constant"], ["b_unstd"]].copy()

        # Order like Table 2 (minus Hispanic, which is unavailable)
        desired_order = [
            "Racism score",
            "Education (years)",
            "Household income per capita",
            "Occupational prestige",
            "Female",
            "Age",
            "Black",
            # "Hispanic" not available
            "Other race",
            "Conservative Protestant",
            "No religion",
            "Southern",
        ]
        predictors = predictors.reindex([r for r in desired_order if r in predictors.index])

        # Format a single column like "0.130**"
        formatted = predictors.copy()
        formatted["beta_with_stars"] = formatted["beta_std"].map(lambda v: f"{v: .3f}" if pd.notna(v) else "") + formatted["stars"].fillna("")
        formatted = formatted[["beta_with_stars"]]

        # Attach constant at bottom (unstandardized)
        formatted.loc["Constant (unstandardized)"] = [f"{float(const.iloc[0,0]): .3f}" if const.shape[0] == 1 else ""]

        # Save
        with open(f"./output/{title}_paper_style.txt", "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write("Standardized OLS coefficients (beta) + significance stars computed from replication p-values.\n")
            f.write("Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed). Constant shown unstandardized.\n")
            f.write("NOTE: Hispanic dummy is omitted because no direct Hispanic identifier exists in the provided extract.\n\n")
            f.write(formatted.to_string())

        return formatted

    paperA = paper_style(tabA, "Table2_ModelA_Minority6")
    paperB = paper_style(tabB, "Table2_ModelB_Other12")

    # Combined overview
    overview = pd.concat([fitA, fitB], ignore_index=True)
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication from microdata (1993 only)\n")
        f.write("Two OLS models; standardized betas reported in paper-style files.\n")
        f.write("Hispanic dummy omitted (field not present in provided variables).\n\n")
        f.write(overview.to_string(index=False))
        f.write("\n\nSee also: Table2_missingness_audit.txt\n")

    return {
        "ModelA_replication_table": tabA,
        "ModelB_replication_table": tabB,
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "missingness_audit": missing_audit,
    }