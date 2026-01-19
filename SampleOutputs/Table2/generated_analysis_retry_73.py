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
        Conservative missing-code handling for this extract.
        Many GSS items use high values as NA (e.g., 8/9, 98/99, 998/999, 9998/9999).
        """
        x = to_num(x).copy()
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5 scale. Dislike/dislike very much are 4/5.
        Missing if outside 1-5 or NA-coded.
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

    def zscore(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def build_dislike_count(df, items, allow_partial=False, min_non_missing=None):
        """
        Count dislikes across items.

        Bryson treats DK as missing. The exact threshold rule for partial item nonresponse is
        not fully specified in the prompt. To avoid collapsing N, default to allow_partial
        with a minimum required number of answered items (if provided), otherwise require >=1.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)

        if not allow_partial:
            # complete-case across items
            return mat.sum(axis=1, min_count=len(items))

        # allow partial: require at least min_non_missing observed items
        if min_non_missing is None:
            min_non_missing = 1
        nonmiss = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, min_count=1)
        count = count.mask(nonmiss < min_non_missing, np.nan)
        return count

    def standardize_and_fit(df_model, y_col, x_cols, add_const=True):
        """
        Fit OLS on standardized y and standardized x.
        Returns fitted model and a dataframe of standardized betas (no SE/t/p exposed in final table).
        """
        d = df_model[[y_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
        y = zscore(d[y_col])
        Xz = pd.DataFrame({c: zscore(d[c]) for c in x_cols}, index=d.index)

        # Drop columns that became all-NaN or zero variance after standardization
        keep = []
        for c in Xz.columns:
            if Xz[c].notna().all():
                if Xz[c].std(ddof=0) > 0:
                    keep.append(c)
        Xz = Xz[keep]

        if add_const:
            X = sm.add_constant(Xz, has_constant="add")
        else:
            X = Xz

        # also drop any rows with NaN introduced by zscore (shouldn't, but safe)
        ok = y.notna() & np.isfinite(y)
        y = y.loc[ok]
        X = X.loc[ok]

        model = sm.OLS(y, X).fit()
        return model, d.index, keep

    def format_stars_from_p(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def save_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

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
    # Dependent variables
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # To avoid over-deleting (a major prior issue), allow partial item response but require
    # most items answered. This is a reasonable approximation when the paper says DK treated
    # as missing and cases with missing excluded (often meaning excluded at model stage,
    # not necessarily requiring all 6/12 items answered).
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, allow_partial=True, min_non_missing=5)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, allow_partial=True, min_non_missing=10)

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
    # require all 5 components observed
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(lambda s: s > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_gss_missing(df["age"])
    df["age"] = age.where(age.between(18, 89))

    # Race dummies (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_gss_missing(df["race"]).where(lambda s: s.isin([1, 2, 3]))
    df["black"] = np.where(race.notna(), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.notna(), (race == 3).astype(float), np.nan)

    # Hispanic: not present in provided variables; cannot be constructed faithfully.
    # Keep as missing and do NOT include in model (otherwise listwise deletion collapses N).
    df["hispanic"] = np.nan

    # Religion / denomination
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])

    # Conservative Protestant proxy: Protestant & (Baptist, Other denom, or Non-denom)
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot[(relig.isna()) | (denom.isna())] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG == 4
    norelig = (relig == 4).astype(float)
    norelig[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # South: REGION == 3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_gss_missing(df["region"]).where(lambda s: s.isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Model specs (Table 2)
    # -----------------------------
    # Note: Hispanic cannot be included from provided data.
    x_cols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age",
        "black",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    pretty_names = {
        "racism_score": "Racism score",
        "education_years": "Education",
        "hh_income_per_capita": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
        "const": "Constant",
    }

    # -----------------------------
    # Fit models and build "paper-style" output (betas + stars only)
    # -----------------------------
    results = {}

    def run_one(dv_col, model_key):
        model, used_idx, kept_x = standardize_and_fit(df, dv_col, x_cols, add_const=True)

        # Standardized betas are the coefficients from the standardized regression
        params = model.params.copy()
        pvals = model.pvalues.copy()

        # Build table in the intended order (excluding missing/unavailable vars)
        ordered_terms = ["const"] + kept_x  # model contains const + kept_x
        rows = []
        for term in ["racism_score", "education_years", "hh_income_per_capita", "occ_prestige",
                     "female", "age", "black", "other_race", "cons_protestant", "no_religion", "south"]:
            if term in params.index:
                beta = float(params.loc[term])
                star = format_stars_from_p(float(pvals.loc[term]))
                rows.append({"variable": pretty_names.get(term, term), "std_beta": beta, "stars": star, "std_beta_star": f"{beta:.3f}{star}"})
            else:
                # keep explicit row if dropped to make debugging clear
                rows.append({"variable": pretty_names.get(term, term), "std_beta": np.nan, "stars": "", "std_beta_star": ""})

        # Constant is not a standardized beta; report unstandardized intercept from an unstandardized model
        # to mirror typical presentation (and avoid meaningless "standardized intercept").
        # Compute unstandardized OLS (same sample) for intercept only.
        d_raw = df[[dv_col] + kept_x].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
        Xraw = sm.add_constant(d_raw[kept_x], has_constant="add")
        yraw = d_raw[dv_col]
        model_raw = sm.OLS(yraw, Xraw).fit()
        const_val = float(model_raw.params.get("const", np.nan))
        const_p = float(model_raw.pvalues.get("const", np.nan))
        const_star = format_stars_from_p(const_p) if np.isfinite(const_p) else ""
        const_row = {"variable": pretty_names["const"], "std_beta": np.nan, "stars": const_star, "std_beta_star": f"{const_val:.3f}{const_star}"}

        table = pd.DataFrame(rows)
        table = pd.concat([table, pd.DataFrame([const_row])], ignore_index=True)

        fit = pd.DataFrame([{
            "model": model_key,
            "n": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "notes": "Std betas from OLS on z-scored variables; stars from model p-values; intercept from unstandardized OLS on same sample."
        }])

        # Save human-readable files
        save_text(f"./output/{model_key}_summary.txt", model.summary().as_text())
        save_text(f"./output/{model_key}_table.txt",
                  table.to_string(index=False, justify="left",
                                  formatters={"std_beta": lambda v: "" if pd.isna(v) else f"{v:.6f}"}))
        save_text(f"./output/{model_key}_fit.txt", fit.to_string(index=False))

        return table, fit, model, model_raw

    tabA, fitA, modelA_std, modelA_raw = run_one("dislike_minority_genres", "Table2_ModelA_dislike_minority6")
    tabB, fitB, modelB_std, modelB_raw = run_one("dislike_other12_genres", "Table2_ModelB_dislike_other12")

    overview = []
    overview.append("1993 GSS Table 2-style replication (computed from provided microdata)\n")
    overview.append("Outputs: standardized betas with significance stars; no SE/t/p columns are reported in the tables.\n")
    overview.append("Note: Hispanic indicator is not available in the provided extract, so it cannot be included.\n")
    overview.append("\nModel A DV: Count of disliked minority-associated genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
    overview.append(fitA.to_string(index=False))
    overview.append("\n\nModel B DV: Count of disliked other 12 genres\n")
    overview.append(fitB.to_string(index=False))
    overview.append("\n")
    save_text("./output/Table2_overview.txt", "".join(overview))

    results["ModelA_table"] = tabA
    results["ModelB_table"] = tabB
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Extra diagnostics to help debug N collapse and dropped columns (saved, not returned)
    diag = []
    diag.append("Diagnostics (missingness rates on constructed/used vars; 1993 only)\n\n")
    diag_vars = [
        "dislike_minority_genres", "dislike_other12_genres", "racism_score",
        "education_years", "hh_income_per_capita", "occ_prestige", "female", "age",
        "black", "other_race", "cons_protestant", "no_religion", "south",
    ]
    for v in diag_vars:
        if v in df.columns:
            miss = df[v].isna().mean()
            diag.append(f"{v:24s} missing_rate={miss:.3f}\n")
    save_text("./output/Table2_diagnostics.txt", "".join(diag))

    return results