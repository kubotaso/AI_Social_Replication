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

    def clean_gss_missing(x):
        """
        Conservative missing-code handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: valid 1..5. Dislike if 4 or 5.
        Missing if outside 1..5 or sentinel.
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

    def build_dislike_count_allow_partial(df, items, min_nonmissing=None):
        """
        Sum of binary dislike indicators with item-level missing.
        IMPORTANT: To avoid collapsing N (a major prior issue), we do NOT require all items present.
        Instead:
          - require at least min_nonmissing items answered (default: all items if not specified).
        """
        if min_nonmissing is None:
            min_nonmissing = len(items)

        cols = []
        for c in items:
            cols.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(cols, axis=1)

        nonmiss = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, skipna=True)
        count = count.where(nonmiss >= min_nonmissing)
        return count

    def zscore_from_sample(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def significance_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_style(df, dv, x_order, model_name):
        """
        Fit OLS on unstandardized DV/predictors (with intercept).
        Report standardized coefficients (betas) for slopes via beta = b * sd(x)/sd(y),
        plus stars from this model's p-values.
        Intercept reported as unstandardized constant.
        """
        needed = [dv] + x_order
        d = df[needed].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_order) + 10):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        y = to_num(d[dv])
        X = d[x_order].apply(to_num)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas for slopes
        sd_y = y.std(ddof=0)
        betas = {}
        for col in x_order:
            sd_x = X[col].std(ddof=0)
            b = model.params.get(col, np.nan)
            if not np.isfinite(sd_x) or sd_x == 0 or not np.isfinite(sd_y) or sd_y == 0:
                betas[col] = np.nan
            else:
                betas[col] = b * (sd_x / sd_y)

        # Build paper-style table (betas + stars; intercept unstandardized)
        rows = []
        for col in x_order:
            p = float(model.pvalues.get(col, np.nan))
            coef = float(betas[col]) if np.isfinite(betas[col]) else np.nan
            rows.append(
                {
                    "variable": col,
                    "std_coef": coef,
                    "stars": significance_stars(p) if np.isfinite(p) else "",
                    "std_coef_star": (f"{coef:.3f}" + significance_stars(p)) if np.isfinite(coef) and np.isfinite(p) else "",
                }
            )
        # constant (unstandardized)
        const = float(model.params.get("const", np.nan))
        pconst = float(model.pvalues.get("const", np.nan))
        rows.append(
            {
                "variable": "constant",
                "std_coef": np.nan,
                "stars": significance_stars(pconst) if np.isfinite(pconst) else "",
                "std_coef_star": (f"{const:.3f}" + (significance_stars(pconst) if np.isfinite(pconst) else "")) if np.isfinite(const) else "",
            }
        )

        paper_tab = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv,
                    "n": int(model.nobs),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Save human-readable outputs
        summary_path = f"./output/{model_name}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNOTE: Table-style output reports standardized coefficients (betas) for slopes and the unstandardized constant.\n")

        table_path = f"./output/{model_name}_Table2_style.txt"
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(f"{model_name} (Table-2 style)\n")
            f.write("Standardized coefficients (betas) for slopes; unstandardized intercept.\n")
            f.write(paper_tab[["variable", "std_coef_star"]].to_string(index=False))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        # Also save a diagnostic table with both b and beta (useful for debugging)
        diag = pd.DataFrame(
            {
                "b_unstd": model.params,
                "p_value": model.pvalues,
            }
        )
        beta_series = pd.Series({k: v for k, v in betas.items()}, name="beta_std")
        diag = diag.join(beta_series, how="left")
        diag_path = f"./output/{model_name}_diagnostics.txt"
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(f"{model_name} diagnostics\n")
            f.write(diag.to_string())
            f.write("\n")

        return model, paper_tab, fit, d.index

    # ----------------------------
    # Load data, normalize columns
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # Dependent variables
    # ----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # IMPORTANT: prior versions requiring all items created huge N drops.
    # Use a lenient, explicit rule: require at least half the items answered for each DV.
    df["dislike_minority_genres"] = build_dislike_count_allow_partial(df, minority_items, min_nonmissing=3)
    df["dislike_other12_genres"] = build_dislike_count_allow_partial(df, other12_items, min_nonmissing=6)

    # ----------------------------
    # Racism score (0-5)
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
    # Require all 5 components to compute scale (paper treats DK as missing)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # ----------------------------
    # Controls
    # ----------------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
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
    age = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))
    df["age_years"] = age

    # Race (black, other). Hispanic not in provided extract -> best-effort proxy using ETHNIC.
    # NOTE: This is a compromise to avoid zero-variance/dropped Hispanic. It may not match Bryson exactly.
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    if "ethnic" in df.columns:
        # In this extract ETHNIC appears numeric; typical GSS ancestry codes include:
        # 20 = Mexican, 21 = Puerto Rican, 22 = Cuban, 23 = other Spanish/Hispanic.
        # Use this widely-used coding where available; otherwise missing.
        eth = clean_gss_missing(df["ethnic"])
        df["hispanic"] = np.where(eth.isna(), np.nan, eth.isin([20, 21, 22, 23]).astype(float))
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant (approximation from RELIG/DENOM)
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()))
    df["cons_protestant"] = consprot

    # No religion
    df["no_religion"] = ((relig == 4).astype(float)).where(~relig.isna())

    # Southern
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = ((region == 3).astype(float)).where(~region.isna())

    # ----------------------------
    # Model spec (Table 2 order)
    # ----------------------------
    var_labels = {
        "racism_score": "Racism score",
        "education_years": "Education (years)",
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

    # Drop rows with zero-variance predictors can silently happen if a variable is all-NA after listwise deletion.
    # We will diagnose BEFORE fitting.
    def diagnostic_counts(df_in, dv, xcols, name):
        d = df_in[[dv] + xcols].copy().replace([np.inf, -np.inf], np.nan)
        # Not listwise yet; show availability
        info = []
        for c in [dv] + xcols:
            s = d[c]
            info.append(
                {
                    "variable": c,
                    "n_nonmissing": int(s.notna().sum()),
                    "n_unique_nonmissing": int(s.dropna().nunique()),
                    "mean": float(to_num(s).mean()) if s.notna().any() else np.nan,
                }
            )
        out = pd.DataFrame(info)
        out_path = f"./output/{name}_variable_diagnostics.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out.to_string(index=False))
            f.write("\n")
        return out

    diagnostic_counts(df, "dislike_minority_genres", x_order, "Table2_ModelA")
    diagnostic_counts(df, "dislike_other12_genres", x_order, "Table2_ModelB")

    # Fit models
    mA, tabA, fitA, idxA = fit_table2_style(df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6")
    mB, tabB, fitB, idxB = fit_table2_style(df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12")

    # Replace variable codes with paper-like labels for output DataFrames
    def relabel_table(tab):
        tab = tab.copy()
        tab["variable"] = tab["variable"].map(lambda v: var_labels.get(v, v))
        return tab

    tabA_l = relabel_table(tabA)
    tabB_l = relabel_table(tabB)

    # Write a combined overview file
    overview_path = "./output/Table2_overview.txt"
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993)\n")
        f.write("Outputs are computed from the provided microdata; no numbers are copied from the paper.\n")
        f.write("Table-style coefficients are standardized betas for slopes; intercept is unstandardized.\n\n")
        f.write("MODEL A DV: Dislike count among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(tabA_l[["variable", "std_coef_star"]].to_string(index=False))
        f.write("\n\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nMODEL B DV: Dislike count among the 12 remaining genres\n")
        f.write(tabB_l[["variable", "std_coef_star"]].to_string(index=False))
        f.write("\n\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    results = {
        "ModelA_table": tabA_l,
        "ModelB_table": tabB_l,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }
    return results