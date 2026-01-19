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
        Conservative missing-code handling for this extract:
        - coerce non-numeric to NaN
        - set common GSS sentinel codes to NaN
        Note: we avoid aggressive rules that might drop valid values.
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5; define dislike=1 if in {4,5}, like/neutral=0 if in {1,2,3}.
        Anything else (including DK/NA codes) => missing.
        """
        x = clean_gss_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items, require_all_answered=True):
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_all_answered:
            # listwise across items (DK treated as missing -> case excluded)
            return mat.sum(axis=1, min_count=len(items))
        # alternative (not used): partial sums with varying denominators
        return mat.sum(axis=1, min_count=1)

    def zscore(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_unstandardized_then_beta(df_model, y_col, x_cols, model_name):
        """
        Fit OLS on raw variables with intercept, then compute standardized betas as:
            beta_j = b_j * SD(x_j) / SD(y)
        (Intercept remains unstandardized.)
        """
        d = df_model[[y_col] + x_cols].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = to_num(d[y_col])
        X = pd.DataFrame({c: to_num(d[c]) for c in x_cols}, index=d.index)

        # Drop any constant predictors (perfect collinearity / no variation)
        dropped = []
        keep = []
        for c in X.columns:
            sd = X[c].std(skipna=True, ddof=0)
            if not np.isfinite(sd) or sd == 0:
                dropped.append(c)
            else:
                keep.append(c)
        X = X[keep]

        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (exclude intercept)
        y_sd = y.std(ddof=0)
        beta = {}
        for c in keep:
            beta[c] = model.params[c] * (X[c].std(ddof=0) / y_sd)

        beta = pd.Series(beta, name="beta_std").reindex(x_cols)  # include dropped as NaN
        pvals = model.pvalues.reindex(["const"] + keep)

        # Stars from our computed p-values (replication); Table 2 doesn't report SEs/p-values.
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

        table = pd.DataFrame(index=["Racism score", "Education", "Household income per capita",
                                    "Occupational prestige", "Female", "Age", "Black", "Hispanic",
                                    "Other race", "Conservative Protestant", "No religion", "Southern",
                                    "Constant"])

        name_map = {
            "racism_score": "Racism score",
            "education_years": "Education",
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

        # Fill standardized betas for predictors
        for c in x_cols:
            row = name_map.get(c, c)
            table.loc[row, "beta_std"] = float(beta.loc[c]) if c in beta.index and pd.notna(beta.loc[c]) else np.nan

        # Fill unstandardized constant
        table.loc["Constant", "constant_b"] = float(model.params.get("const", np.nan))

        # Stars (from replication p-values; intercept stars based on intercept p-value)
        # For standardized betas we use the p-value from the unstandardized coefficient test.
        star_col = []
        for idx in table.index:
            if idx == "Constant":
                p = float(pvals.get("const", np.nan))
            else:
                # map back to variable name
                inv = {v: k for k, v in name_map.items()}
                v = inv.get(idx, None)
                if v is None:
                    p = np.nan
                else:
                    p = float(model.pvalues.get(v, np.nan))
            star_col.append(stars(p))
        table["stars"] = star_col

        # Display column: beta with stars for predictors, constant separately
        disp = []
        for idx, r in table.iterrows():
            if idx == "Constant":
                if pd.isna(r["constant_b"]):
                    disp.append("")
                else:
                    disp.append(f"{r['constant_b']:.3f}{r['stars']}")
            else:
                if pd.isna(r["beta_std"]):
                    disp.append("")
                else:
                    disp.append(f"{r['beta_std']:.3f}{r['stars']}")
        table["display"] = disp

        fit = {
            "model": model_name,
            "n": int(model.nobs),
            "k_predictors": int(model.df_model),  # excludes intercept
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        }

        notes = {
            "dropped_constant_predictors": dropped,
            "used_predictors": keep,
        }

        return model, table, fit, notes, d.index

    # -----------------------------
    # Load and filter
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs (Table 2)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # "DK treated as missing and cases excluded" -> require all items observed for the DV
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all_answered=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all_answered=True)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to half-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    if "educ" not in df.columns:
        raise ValueError("Missing educ (EDUC).")
    educ = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required income component: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 (PRESTG80).")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing sex (SEX).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing age (AGE).")
    df["age_years"] = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))

    if "race" not in df.columns:
        raise ValueError("Missing race (RACE).")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = (race == 2).astype(float)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = (race == 3).astype(float)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: not available in the provided variables. To avoid collapsing N, omit it from estimation.
    # (We still create the column for completeness of outputs.)
    df["hispanic"] = np.nan

    # Religion: conservative Protestant + no religion
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion fields: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    df["cons_protestant"] = (((relig == 1) & (denom.isin([1, 6, 7]))).astype(float))
    df.loc[relig.isna() | denom.isna(), "cons_protestant"] = np.nan

    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    if "region" not in df.columns:
        raise ValueError("Missing region (REGION).")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -----------------------------
    # Model specs (Table 2 RHS, minus Hispanic due to missing identifier)
    # -----------------------------
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # omitted (no valid mapping in provided data)
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # -----------------------------
    # Fit models and save outputs
    # -----------------------------
    results = {}

    def save_table_text(path, table, title, fit, notes):
        with open(path, "w", encoding="utf-8") as f:
            f.write(title + "\n")
            f.write("=" * len(title) + "\n\n")
            f.write("NOTE: This is a replication output computed from the provided microdata.\n")
            f.write("Table 2 in Bryson (1996) reports standardized coefficients (betas) and stars, not SEs.\n")
            f.write("Stars here are based on two-tailed p-values from the OLS fit in this replication.\n\n")
            f.write("Fit:\n")
            for k, v in fit.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nDropped/constant predictors (if any):\n")
            f.write(f"  {notes.get('dropped_constant_predictors', [])}\n")
            f.write("\nUsed predictors:\n")
            f.write(f"  {notes.get('used_predictors', [])}\n\n")
            f.write("Coefficient table (Table-2 style):\n")
            f.write("  - Predictors shown as standardized beta (beta_std)\n")
            f.write("  - Constant shown as unstandardized intercept (constant_b)\n\n")

            out = table[["display"]].copy()
            out.index.name = "term"
            f.write(out.to_string())

            f.write("\n\nRaw numeric values:\n")
            f.write(table[["beta_std", "constant_b", "stars"]].to_string())

    def save_model_summary(path, model):
        with open(path, "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

    # Model A
    modelA, tableA, fitA, notesA, idxA = fit_unstandardized_then_beta(
        df, "dislike_minority_genres", x_cols, "Table2_ModelA_dislike_minority6"
    )
    save_model_summary("./output/Table2_ModelA_summary.txt", modelA)
    save_table_text("./output/Table2_ModelA_table.txt", tableA,
                    "Table 2 Model A: Dislike of minority-associated genres (6-item count)", fitA, notesA)

    # Model B
    modelB, tableB, fitB, notesB, idxB = fit_unstandardized_then_beta(
        df, "dislike_other12_genres", x_cols, "Table2_ModelB_dislike_other12"
    )
    save_model_summary("./output/Table2_ModelB_summary.txt", modelB)
    save_table_text("./output/Table2_ModelB_table.txt", tableB,
                    "Table 2 Model B: Dislike of other genres (12-item count)", fitB, notesB)

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication from provided GSS 1993 extract\n")
        f.write("============================================================\n\n")
        f.write("Implementation notes:\n")
        f.write("- Year restricted to 1993.\n")
        f.write("- DVs are counts of 'dislike' responses (4/5 on 1-5 scale).\n")
        f.write("- Item-level DK/NA treated as missing; DV requires all items answered (listwise across items).\n")
        f.write("- Standardized betas computed from unstandardized OLS as b * SD(X)/SD(Y).\n")
        f.write("- Stars are computed from this replication's OLS p-values.\n")
        f.write("- Hispanic dummy is not estimated because no Hispanic identifier is present in the provided variables.\n\n")
        f.write("Model A fit:\n")
        for k, v in fitA.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nModel B fit:\n")
        for k, v in fitB.items():
            f.write(f"  {k}: {v}\n")

    results["ModelA_table"] = tableA
    results["ModelB_table"] = tableB
    results["ModelA_fit"] = pd.DataFrame([fitA])
    results["ModelB_fit"] = pd.DataFrame([fitB])

    return results