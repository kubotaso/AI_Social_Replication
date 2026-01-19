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

    def is_gss_missing(x):
        """
        Conservative missing-value handling for this extract:
        - Keep only clearly-valid codes for each variable via explicit allowlists/ranges.
        - Additionally treat a few common sentinel codes as missing.
        """
        if pd.isna(x):
            return True
        # common GSS-style sentinels
        if x in (8, 9, 98, 99, 998, 999, 9998, 9999):
            return True
        return False

    def clean_music_item(series):
        """Music taste items are 1-5. Anything else -> missing."""
        x = to_num(series)
        x = x.mask(x.apply(lambda v: is_gss_missing(v)))
        x = x.where(x.between(1, 5))
        return x

    def dislike_indicator(series):
        """1 if dislike/dislike very much (4,5); 0 if (1,2,3); missing otherwise."""
        x = clean_music_item(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_allow(series, true_vals, false_vals):
        """Binary recode with explicit true/false codes; others -> missing."""
        x = to_num(series)
        x = x.mask(x.apply(lambda v: is_gss_missing(v)))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_vals)] = 0.0
        out.loc[x.isin(true_vals)] = 1.0
        return out

    def zscore(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def build_count_complete_case(df, items):
        """
        Count of dislikes across items.
        Per instructions/paper summary: treat DK/etc as missing; exclude missing cases.
        Implement DV as missing unless ALL component items are observed.
        """
        cols = []
        for c in items:
            cols.append(dislike_indicator(df[c]).rename(c))
        mat = pd.concat(cols, axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardized_ols_with_unstd_intercept(df_model, dv, xcols, model_name):
        """
        Fit OLS where standardized coefficients are computed for predictors only.
        Intercept is reported unstandardized from the unstandardized model (as in paper tables).
        - Standardized beta for x_j: b_j * sd(x_j)/sd(y)
        """
        needed = [dv] + xcols
        d = df_model[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < len(xcols) + 2:
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)
        X = d[xcols].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # standardized betas for predictors (not intercept)
        y_sd = y.std(ddof=0)
        betas = {}
        for c in xcols:
            x_sd = X[c].std(ddof=0)
            if not np.isfinite(x_sd) or x_sd <= 0 or not np.isfinite(y_sd) or y_sd <= 0:
                betas[c] = np.nan
            else:
                betas[c] = model.params[c] * (x_sd / y_sd)

        beta_s = pd.Series(betas, name="coef_beta")

        # build table in paper-like order with labels
        tab = pd.DataFrame({
            "coef_beta": beta_s,
            "t": model.tvalues.reindex(xcols),
            "p_value": model.pvalues.reindex(xcols),
        })

        def star(p):
            if pd.isna(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        tab["stars"] = tab["p_value"].apply(star)
        tab["coef_beta_star"] = tab["coef_beta"].map(lambda v: np.nan if pd.isna(v) else float(v)).astype(float)

        # add intercept separately (unstandardized)
        intercept = float(model.params["const"]) if "const" in model.params.index else np.nan
        intercept_p = float(model.pvalues["const"]) if "const" in model.pvalues.index else np.nan
        intercept_row = pd.DataFrame(
            {
                "coef_beta": [np.nan],
                "t": [model.tvalues.get("const", np.nan)],
                "p_value": [intercept_p],
                "stars": [star(intercept_p)],
                "coef_beta_star": [np.nan],
            },
            index=["Constant (unstandardized)"],
        )

        tab = pd.concat([tab, intercept_row], axis=0)

        fit = pd.DataFrame([{
            "model": model_name,
            "n": int(model.nobs),
            "k_predictors": int(model.df_model),  # excludes intercept
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "intercept_unstd": intercept,
        }])

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nStandardized coefficients (betas) computed as b * sd(x)/sd(y); intercept shown unstandardized.\n")
            f.write("\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        # human-readable coefficient table
        tab_to_write = tab.copy()
        tab_to_write.index.name = "term"
        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_to_write.to_string(float_format=lambda x: f"{x: .6f}"))

        return model, tab, fit, d.index

    # -------------------------
    # Load data + restrict to 1993
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["year", "id"]
    for c in required:
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
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_allow(df["rachaf"], true_vals=[1], false_vals=[2])   # object to school > half black
    rac2 = binary_from_allow(df["busing"], true_vals=[2], false_vals=[1])   # oppose busing
    rac3 = binary_from_allow(df["racdif1"], true_vals=[2], false_vals=[1])  # deny discrimination
    rac4 = binary_from_allow(df["racdif3"], true_vals=[2], false_vals=[1])  # deny educational chance
    rac5 = binary_from_allow(df["racdif4"], true_vals=[1], false_vals=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = to_num(df["educ"]).mask(to_num(df["educ"]).apply(lambda v: is_gss_missing(v)))
    df["education"] = educ.where(educ.between(0, 20))

    # HH income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = to_num(df["realinc"]).mask(to_num(df["realinc"]).apply(lambda v: is_gss_missing(v)))
    hompop = to_num(df["hompop"]).mask(to_num(df["hompop"]).apply(lambda v: is_gss_missing(v)))
    hompop = hompop.where(hompop > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column.")
    prest = to_num(df["prestg80"]).mask(to_num(df["prestg80"]).apply(lambda v: is_gss_missing(v)))
    df["occupational_prestige"] = prest

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing sex column.")
    df["female"] = binary_from_allow(df["sex"], true_vals=[2], false_vals=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column.")
    age = to_num(df["age"]).mask(to_num(df["age"]).apply(lambda v: is_gss_missing(v)))
    df["age"] = age.where(age.between(18, 89))

    # Race indicators from RACE: (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = to_num(df["race"]).mask(to_num(df["race"]).apply(lambda v: is_gss_missing(v)))
    race = race.where(race.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator:
    # The provided extract has no dedicated Hispanic variable; however, it *does* include ETHNIC.
    # We implement a conservative proxy: treat ETHNIC==1 as "Hispanic" if that code appears,
    # otherwise leave missing and drop only those cases via listwise deletion.
    # This avoids conditioning the sample on race/ethnicity while allowing the model to run when the proxy exists.
    if "ethnic" in df.columns:
        eth = to_num(df["ethnic"]).mask(to_num(df["ethnic"]).apply(lambda v: is_gss_missing(v)))
        if (eth == 1).any():
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 1).astype(float))
        else:
            df["hispanic"] = np.nan
    else:
        df["hispanic"] = np.nan

    # Religion variables
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column.")
    relig = to_num(df["relig"]).mask(to_num(df["relig"]).apply(lambda v: is_gss_missing(v)))
    denom = to_num(df["denom"]).mask(to_num(df["denom"]).apply(lambda v: is_gss_missing(v)))

    # No religion: RELIG == 4 (none)
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy per provided mapping: RELIG==1 and DENOM in {1,6,7}
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["conservative_protestant"] = consprot

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = to_num(df["region"]).mask(to_num(df["region"]).apply(lambda v: is_gss_missing(v)))
    region = region.where(region.isin([1, 2, 3, 4]))
    df["southern"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Diagnostics (requested by feedback): do not filter on race/religion; just tabulate
    # -------------------------
    diag_lines = []
    diag_lines.append(f"Rows after YEAR==1993: {df.shape[0]}")
    if "hispanic" in df.columns:
        diag_lines.append("\nHispanic proxy value counts (including NaN):")
        diag_lines.append(df["hispanic"].value_counts(dropna=False).to_string())
    diag_lines.append("\nNo religion value counts (including NaN):")
    diag_lines.append(df["no_religion"].value_counts(dropna=False).to_string())
    diag_lines.append("\nRace value counts (raw race codes, including NaN):")
    diag_lines.append(df["race"].value_counts(dropna=False).to_string())

    with open("./output/diagnostics_counts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(diag_lines) + "\n")

    # -------------------------
    # Fit models (same RHS)
    # -------------------------
    xcols = [
        "racism_score",
        "education",
        "hh_income_per_capita",
        "occupational_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "conservative_protestant",
        "no_religion",
        "southern",
    ]
    for c in xcols:
        if c not in df.columns:
            raise ValueError(f"Missing constructed predictor: {c}")

    results = {}

    modelA, tabA, fitA, idxA = standardized_ols_with_unstd_intercept(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    modelB, tabB, fitB, idxB = standardized_ols_with_unstd_intercept(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )

    # Paper-like ordering + nicer labels for returned tables
    label_map = {
        "racism_score": "Racism score",
        "education": "Education",
        "hh_income_per_capita": "Household income per capita",
        "occupational_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "conservative_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "Constant (unstandardized)": "Constant",
    }

    def relabel(tab):
        out = tab.copy()
        out = out.reindex(xcols + ["Constant (unstandardized)"])
        out.index = [label_map.get(i, i) for i in out.index]
        return out

    tabA_l = relabel(tabA)
    tabB_l = relabel(tabB)

    results["ModelA_table"] = tabA_l
    results["ModelB_table"] = tabB_l
    results["ModelA_fit"] = fitA
    results["ModelB_fit"] = fitB

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS with standardized coefficients (betas)\n")
        f.write("Betas computed as b * sd(x)/sd(y) from unstandardized OLS; intercept reported unstandardized.\n")
        f.write("Missing values: only explicitly valid codes used; DK/refused/sentinels treated as missing; listwise deletion per model.\n")
        f.write("Note: 'Hispanic' depends on whether ETHNIC==1 exists in this extract; see diagnostics_counts.txt.\n\n")
        f.write("Model A DV: count of disliked minority-associated genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(tabA_l.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n\n")
        f.write("Model B DV: count of disliked other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(tabB_l.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    return results