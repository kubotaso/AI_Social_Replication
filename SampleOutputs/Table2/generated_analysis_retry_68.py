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

    def clean_na_codes(x):
        """
        Conservative missing-code handling for this extracted file:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        """
        x = to_num(x).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(series):
        """
        Music items: 1-5 scale; 4/5 = dislike; 1/2/3 = not dislike.
        Anything else (incl. DK/NA) => missing.
        """
        x = clean_na_codes(series)
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

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def build_dislike_count(df, item_cols, require_all=True):
        inds = []
        for c in item_cols:
            inds.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(inds, axis=1)
        if require_all:
            return mat.sum(axis=1, min_count=len(item_cols))
        # if not require_all, allow partial sums (not used here)
        return mat.sum(axis=1, min_count=1)

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

    def fit_table2_model(df, dv, x_order, model_name):
        """
        Fit unstandardized OLS for intercept/R2/N, compute standardized betas for slopes:
            beta_j = b_j * sd(x_j) / sd(y)
        Stars are computed from the p-values of the unstandardized OLS (same inference).
        Intercept is reported unstandardized (not standardized).
        """
        needed = [dv] + x_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        y = d[dv].astype(float)
        X = d[x_order].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        m = sm.OLS(y, Xc).fit()

        sd_y = y.std(ddof=0)
        betas = {}
        for c in x_order:
            sd_x = X[c].std(ddof=0)
            if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                betas[c] = np.nan
            else:
                betas[c] = m.params[c] * (sd_x / sd_y)

        # Build "paper-style" table: standardized betas (slopes) + stars
        rows = []
        for c in x_order:
            p = m.pvalues.get(c, np.nan)
            bstd = betas.get(c, np.nan)
            rows.append(
                {
                    "variable": c,
                    "std_beta": bstd,
                    "stars": add_stars(p),
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                }
            )

        # Constant: unstandardized
        rows.append(
            {
                "variable": "const",
                "std_beta": np.nan,
                "stars": add_stars(m.pvalues.get("const", np.nan)),
                "p_value": float(m.pvalues.get("const", np.nan)) if np.isfinite(m.pvalues.get("const", np.nan)) else np.nan,
            }
        )
        tab = pd.DataFrame(rows)

        # Add intercept separately for clarity
        intercept = float(m.params.get("const", np.nan))

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m.nobs),
                    "k": int(m.df_model + 1),
                    "r2": float(m.rsquared),
                    "adj_r2": float(m.rsquared_adj),
                    "intercept_unstd": intercept,
                }
            ]
        )

        return m, tab, fit, d.index

    # -----------------------------
    # Load data and filter year
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Dependent variables (exact sets)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Require complete responses on all items for each DV (DK treated as missing; cases excluded)
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all=True)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
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

    # -----------------------------
    # Controls
    # -----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing educ column (EDUC).")
    educ = clean_na_codes(df["educ"]).where(clean_na_codes(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_na_codes(df["realinc"])
    hompop = clean_na_codes(df["hompop"]).where(clean_na_codes(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column (PRESTG80).")
    df["occ_prestige"] = clean_na_codes(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing sex column (SEX).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column (AGE).")
    df["age_years"] = clean_na_codes(df["age"]).where(clean_na_codes(df["age"]).between(18, 89))

    # Race dummies from RACE (white reference)
    if "race" not in df.columns:
        raise ValueError("Missing race column (RACE).")
    race = clean_na_codes(df["race"]).where(clean_na_codes(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables -> cannot construct faithfully
    # Keep as missing so it is excluded from listwise deletion only if not in model.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy from RELIG and DENOM (as specified in mapping instruction)
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_na_codes(df["relig"])
    denom = clean_na_codes(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    df["no_religion"] = (relig == 4).astype(float)
    df.loc[relig.isna(), "no_religion"] = np.nan

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column (REGION).")
    region = clean_na_codes(df["region"]).where(clean_na_codes(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = (region == 3).astype(float)
    df.loc[region.isna(), "south"] = np.nan

    # -----------------------------
    # Model specs
    # -----------------------------
    # IMPORTANT: Hispanic dummy is not present in this extract; exclude it (otherwise n=0 listwise).
    x_order = [
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

    # Sanity: ensure predictors exist and have at least some non-missing
    for c in x_order:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    # -----------------------------
    # Fit both models
    # -----------------------------
    mA, tabA, fitA, idxA = fit_table2_model(df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6")
    mB, tabB, fitB, idxB = fit_table2_model(df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12")

    # -----------------------------
    # Human-readable outputs
    # -----------------------------
    def write_outputs(model, tab, fit, name):
        # Statsmodels summary (unstandardized OLS)
        with open(f"./output/{name}_ols_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        # Table 2 style: standardized betas + stars, plus intercept separately
        pretty = tab.copy()
        # Add a printable coefficient-with-stars column
        pretty["std_beta_str"] = pretty["std_beta"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + pretty["stars"].fillna("")
        # Intercept row: show unstandardized intercept in the string column
        intercept_val = float(fit["intercept_unstd"].iloc[0])
        pretty.loc[pretty["variable"] == "const", "std_beta_str"] = f"{intercept_val:.3f}" + pretty.loc[pretty["variable"] == "const", "stars"].fillna("").iloc[0]

        out_cols = ["variable", "std_beta_str"]
        with open(f"./output/{name}_table2_style.txt", "w", encoding="utf-8") as f:
            f.write("Standardized OLS coefficients (slopes) with significance stars computed from OLS p-values.\n")
            f.write("Note: intercept is reported unstandardized; standardized intercept is not defined.\n\n")
            f.write(pretty[out_cols].to_string(index=False))
            f.write("\n\nFit:\n")
            f.write(fit.drop(columns=["intercept_unstd"]).to_string(index=False))
            f.write("\n")

        # Full computed table (auditable)
        full = tab.merge(fit[["model", "intercept_unstd"]], how="left", left_on=tab.index.map(lambda _: 0), right_index=True)
        # simpler: just save tab + fit separately as text
        tab.to_csv(f"./output/{name}_computed_betas.csv", index=False)
        fit.to_csv(f"./output/{name}_fit.csv", index=False)

    write_outputs(mA, tabA, fitA, "Table2_ModelA_dislike_minority6")
    write_outputs(mB, tabB, fitB, "Table2_ModelB_dislike_other12")

    # Diagnostics: variable variation in each model sample
    def diag(df_used, name):
        lines = []
        lines.append(f"Diagnostics for {name}\n")
        lines.append(f"N={df_used.shape[0]}\n")
        for c in x_order:
            s = df_used[c]
            lines.append(f"{c}: nonmissing={int(s.notna().sum())}, mean={float(s.mean()):.4f}, sd={float(s.std(ddof=0)):.4f}, min={float(s.min()):.4f}, max={float(s.max()):.4f}\n")
        return "".join(lines)

    dA = df.loc[idxA, ["dislike_minority_genres"] + x_order].dropna()
    dB = df.loc[idxB, ["dislike_other12_genres"] + x_order].dropna()
    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write(diag(dA, "ModelA (minority-associated 6 genres)"))
        f.write("\n")
        f.write(diag(dB, "ModelB (other 12 genres)"))

    # Return as dict of DataFrames (auditable)
    return {
        "ModelA_table2_style": tabA[["variable", "std_beta", "stars", "p_value"]],
        "ModelB_table2_style": tabB[["variable", "std_beta", "stars", "p_value"]],
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }