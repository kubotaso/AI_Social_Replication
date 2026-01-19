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
        Conservative missing handling:
        - Convert to numeric
        - Treat common GSS sentinels as missing
        - Leave ordinary values intact
        """
        x = to_num(x)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        return x.mask(x.isin(sentinels))

    def likert_dislike_indicator(x):
        """
        Music taste: 1-5.
        Dislike = 4/5 => 1
        Like/neutral = 1/2/3 => 0
        Missing if not in 1..5 or NA-coded.
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

    def build_count_complete(df, items):
        """
        Count of disliked genres.
        To mirror "DK treated as missing and missing cases excluded", require complete data
        across the items used in that DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def standardize_beta_from_unstd_fit(fit, y, X):
        """
        Compute standardized beta coefficients from an unstandardized OLS fit:
        beta_j = b_j * sd(X_j) / sd(Y), using sample SD (ddof=1).
        Intercept is not standardized; keep unstandardized intercept separately.
        """
        y_sd = y.std(ddof=1)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = X[col].std(ddof=1)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = fit.params[col] * (x_sd / y_sd)
        return pd.Series(betas)

    def star_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_one(df, dv, xcols, model_name, pretty_names=None):
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        if d.shape[0] < (len(xcols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(xcols)}).")

        y = d[dv].astype(float)

        # Build X and drop any zero-variance predictors (would cause singularity)
        X = d[xcols].astype(float)
        zero_var = [c for c in X.columns if X[c].std(ddof=1) == 0 or not np.isfinite(X[c].std(ddof=1))]
        if zero_var:
            X = X.drop(columns=zero_var)

        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xc).fit()

        # Standardized betas from unstandardized fit (preferred for matching "standardized OLS coefficients")
        beta = standardize_beta_from_unstd_fit(fit, y, Xc)

        # Assemble table in requested order (if pretty_names supplied)
        rows = []
        for term in ["racism_score", "education", "hh_income_pc", "occ_prestige", "female", "age",
                     "black", "hispanic", "other_race", "cons_protestant", "no_religion", "south"]:
            if term in Xc.columns and term in beta.index:
                p = fit.pvalues.get(term, np.nan)
                b = beta.loc[term]
                rows.append((term, b, p, star_from_p(p)))
            else:
                rows.append((term, np.nan, np.nan, ""))

        tab = pd.DataFrame(rows, columns=["term", "beta_std", "p_value", "stars"]).set_index("term")

        # Add intercept (unstandardized) as in paper
        const_b = fit.params.get("const", np.nan)
        const_p = fit.pvalues.get("const", np.nan)
        const_row = pd.DataFrame(
            {"beta_std": [np.nan], "p_value": [const_p], "stars": [star_from_p(const_p)], "constant_b": [const_b]},
            index=["constant"]
        )

        # Provide constant_b column for all rows (NaN except constant)
        tab["constant_b"] = np.nan
        out = pd.concat([tab, const_row], axis=0)

        # Optional pretty labels
        if pretty_names is not None:
            out.insert(0, "label", [pretty_names.get(i, i) for i in out.index])

        fitstats = pd.DataFrame([{
            "model": model_name,
            "n": int(fit.nobs),
            "k_predictors": int(len(Xc.columns) - 1),
            "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
        }])

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Standardized betas computed post-estimation: beta = b * SD(X) / SD(Y) on estimation sample.\n")
            f.write("- Constant reported as unstandardized intercept from the same unstandardized OLS.\n")
            f.write(f"- Dropped zero-variance predictors (if any): {', '.join(zero_var) if zero_var else 'none'}\n")

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(out.to_string(float_format=lambda x: f"{x: .6f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fitstats.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

        return fit, out, fitstats, d.index

    # -----------------------------
    # Load / normalize columns
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must contain 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct DVs (complete-case across each DV's items)
    # -----------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dv_minority6_dislike"] = build_count_complete(df, minority_items)
    df["dv_other12_dislike"] = build_count_complete(df, other12_items)

    # -----------------------------
    # Racism scale (0-5; require all five items)
    # -----------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls / predictors
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column.")
    educ = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))
    df["education"] = educ

    # Income per capita: REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing realinc and/or hompop columns.")
    realinc = clean_gss_missing(df["realinc"]).where(clean_gss_missing(df["realinc"]) >= 0)
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
    df["age"] = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))

    # Race dummies: black, other_race; and attempt a hispanic proxy if available.
    if "race" not in df.columns:
        raise ValueError("Missing race column.")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: best-effort from available 'ethnic' if it is a GSS Hispanic-origin code.
    # If not decodable, leave as missing (will reduce N); we therefore use a cautious rule:
    # treat small, common Hispanic codes as Hispanic; otherwise non-Hispanic.
    # This is necessarily approximate given the provided extract.
    hisp = pd.Series(np.nan, index=df.index, dtype="float64")
    if "hispanic" in df.columns:
        # if dataset already contains a hispanic indicator, use it (expects 0/1 or 1/2 etc.)
        hx = clean_gss_missing(df["hispanic"])
        # map 1/0 or 1/2
        if hx.dropna().isin([0, 1]).all():
            hisp = hx.astype(float)
        else:
            # try 1=yes, 2=no
            hisp = binary_from_codes(hx, true_codes=[1], false_codes=[2])
    elif "ethnic" in df.columns:
        ex = clean_gss_missing(df["ethnic"])
        # Commonly, GSS ETHNIC (in some extracts) is: 1=Mexican, 2=Puerto Rican, 3=Other Spanish, 4=Non-Hispanic.
        # Implement that mapping when observed; otherwise leave as missing.
        vals = set(ex.dropna().unique().tolist())
        if vals.issubset({1, 2, 3, 4}):
            hisp = pd.Series(np.nan, index=df.index, dtype="float64")
            hisp.loc[ex.isin([1, 2, 3])] = 1.0
            hisp.loc[ex.isin([4])] = 0.0
        else:
            # If ethnic codes are not interpretable, set to missing (cannot safely construct)
            hisp = pd.Series(np.nan, index=df.index, dtype="float64")
    df["hispanic"] = hisp

    # Religion: Conservative Protestant + No religion
    if "relig" not in df.columns:
        raise ValueError("Missing relig column.")
    relig = clean_gss_missing(df["relig"])

    # Conservative Protestant proxy: RELIG==1 (Protestant) and DENOM in {1,6,7}
    consprot = pd.Series(np.nan, index=df.index, dtype="float64")
    if "denom" in df.columns:
        denom = clean_gss_missing(df["denom"])
        consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
        consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column.")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Models (Table 2): two DVs, same RHS
    # -----------------------------
    xcols = [
        "racism_score",
        "education",
        "hh_income_pc",
        "occ_prestige",
        "female",
        "age",
        "black",
        "hispanic",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    for c in xcols + ["dv_minority6_dislike", "dv_other12_dislike"]:
        if c not in df.columns:
            raise ValueError(f"Missing constructed column: {c}")

    pretty = {
        "racism_score": "Racism score",
        "education": "Education",
        "hh_income_pc": "Household income per capita",
        "occ_prestige": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
        "constant": "Constant (unstandardized intercept)",
    }

    # Fit both
    modelA, tableA, fitA, idxA = fit_one(
        df=df,
        dv="dv_minority6_dislike",
        xcols=xcols,
        model_name="Table2_ModelA_dislike_6_minority_associated",
        pretty_names=pretty,
    )
    modelB, tableB, fitB, idxB = fit_one(
        df=df,
        dv="dv_other12_dislike",
        xcols=xcols,
        model_name="Table2_ModelB_dislike_12_remaining",
        pretty_names=pretty,
    )

    # Missingness audit (to diagnose N collapse)
    audit_cols = ["dv_minority6_dislike", "dv_other12_dislike"] + xcols
    miss = pd.DataFrame({
        "missing_n": df[audit_cols].isna().sum(),
        "nonmissing_n": df[audit_cols].notna().sum(),
        "missing_pct": (df[audit_cols].isna().mean() * 100.0),
    }).sort_values("missing_n", ascending=False)

    with open("./output/Table2_missingness_audit.txt", "w", encoding="utf-8") as f:
        f.write("Missingness audit (1993 only)\n")
        f.write(miss.to_string(float_format=lambda x: f"{x: .3f}"))
        f.write("\n\nComplete-case N by model:\n")
        f.write(f"Model A complete-case N: {len(idxA)}\n")
        f.write(f"Model B complete-case N: {len(idxB)}\n")

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (computed from microdata)\n")
        f.write("- OLS on unstandardized variables; standardized betas computed post-estimation.\n")
        f.write("- Stars derived from model p-values (two-tailed): * <.05, ** <.01, *** <.001.\n")
        f.write("- If your extract lacks a valid Hispanic identifier, 'hispanic' will be missing and N will drop.\n\n")
        f.write("Model A DV: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin\n")
        f.write(fitA.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n\nModel B DV: count of dislikes among the 12 remaining genres\n")
        f.write(fitB.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    return {
        "ModelA_table": tableA,
        "ModelB_table": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "Missingness_audit": miss,
    }