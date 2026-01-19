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

    def clean_gss_na(x):
        """
        Conservative NA-code cleaner for this extract.
        - Coerce to numeric.
        - Treat common GSS NA sentinels as missing.
        - Do NOT treat 0 as missing globally (some vars may legitimately be 0).
        """
        x = to_num(x).copy()
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(x):
        """
        Music liking items are 1-5. Dislike is 4 or 5.
        1-3 -> 0, 4-5 -> 1, else missing.
        """
        x = clean_gss_na(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_gss_na(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, items):
        # Listwise across the items (DK treated as missing -> exclude from count construction)
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def zscore(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

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

    def fit_table2_model(df, dv_col, xcols_ordered, model_name):
        """
        Fit unstandardized OLS on original scales, then compute standardized betas post-hoc:
            beta_j = b_j * sd(x_j) / sd(y)
        (Intercept is left unstandardized.)
        """
        needed = [dv_col] + xcols_ordered
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Require enough rows
        if d.shape[0] < (len(xcols_ordered) + 2):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}, k={len(xcols_ordered)}).")

        # Drop only truly zero-variance predictors (avoid runtime errors); record them
        dropped = []
        xcols = []
        for c in xcols_ordered:
            vals = d[c].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                dropped.append(c)
                continue
            if np.nanstd(vals, ddof=0) == 0:
                dropped.append(c)
            else:
                xcols.append(c)

        if len(xcols) == 0:
            raise ValueError(f"{model_name}: all predictors have zero variance or are missing in the analytic sample.")

        y = to_num(d[dv_col])
        X = pd.DataFrame({c: to_num(d[c]) for c in xcols}, index=d.index)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # Post-hoc standardized betas for slopes
        y_sd = np.nanstd(y.to_numpy(), ddof=0)
        betas = {}
        pvals = {}
        for c in xcols_ordered:
            if c not in model.params.index:
                betas[c] = np.nan
                pvals[c] = np.nan
                continue
            x_sd = np.nanstd(X[c].to_numpy(), ddof=0)
            if not np.isfinite(y_sd) or y_sd == 0 or not np.isfinite(x_sd) or x_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(model.params[c] * (x_sd / y_sd))
            pvals[c] = float(model.pvalues[c])

        paper = pd.DataFrame(
            {
                "beta": [betas[c] for c in xcols_ordered],
                "stars": [star_from_p(pvals[c]) for c in xcols_ordered],
                "p_value(ours)": [pvals[c] for c in xcols_ordered],
            },
            index=xcols_ordered,
        )
        paper.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv_col,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dropped_zero_variance": ", ".join(dropped) if dropped else "",
                }
            ]
        )

        # Also return a compact "full" table for debugging (computed, not from paper)
        full = pd.DataFrame(
            {
                "b": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        full.index.name = "term"

        return model, paper, full, fit, d.index

    # -------------------------
    # Load + normalize columns
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required minimal columns
    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # DVs (exact fields per mapping)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = ["bigband", "blugrass", "country", "musicals", "classicl", "folk",
                     "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"]

    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
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

    # -------------------------
    # RHS controls
    # -------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    educ = clean_gss_na(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for income pc: {c}")
    realinc = clean_gss_na(df["realinc"])
    hompop = clean_gss_na(df["hompop"]).where(lambda s: s > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_gss_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    age = clean_gss_na(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies (White is reference)
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_gss_na(df["race"]).where(lambda s: s.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator: not available in provided variables.
    # Keep it but set to 0 when missing so it does not collapse the sample to n=0;
    # flag clearly in outputs that this is a structural limitation of this extract.
    df["hispanic"] = 0.0

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_na(df["relig"])
    denom = clean_gss_na(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_gss_na(df["region"]).where(lambda s: s.isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Fit models (Table 2 spec)
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

    # Ensure all exist
    for c in x_order:
        if c not in df.columns:
            raise ValueError(f"Constructed predictor missing unexpectedly: {c}")

    mA, tabA_paper, tabA_full, fitA, idxA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, tabB_paper, tabB_full, fitB, idxB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    def write_table(path, df_out, float_fmt="{: .6f}".format):
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(df_out, pd.DataFrame):
                f.write(df_out.to_string(float_format=float_fmt))
            else:
                f.write(str(df_out))
            f.write("\n")

    # Paper-style: standardized betas only (+ stars)
    write_table("./output/Table2_ModelA_paper_style.txt", tabA_paper, float_fmt=lambda x: f"{x: .3f}")
    write_table("./output/Table2_ModelB_paper_style.txt", tabB_paper, float_fmt=lambda x: f"{x: .3f}")

    # Full model summaries (computed from data; not in paper table)
    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(mA.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nNOTE: Hispanic indicator is forced to 0.0 because this extract has no Hispanic identifier.\n")
    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(mB.summary().as_text())
        f.write("\n\nFit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\nNOTE: Hispanic indicator is forced to 0.0 because this extract has no Hispanic identifier.\n")

    # Full coefficient tables (debug)
    write_table("./output/Table2_ModelA_full_coefficients.txt", tabA_full)
    write_table("./output/Table2_ModelB_full_coefficients.txt", tabB_full)

    # Overview
    overview = []
    overview.append("Table 2 replication attempt (GSS 1993).")
    overview.append("Standardized coefficients are computed post-hoc: beta = b * sd(x) / sd(y). Intercept is unstandardized.")
    overview.append("IMPORTANT LIMITATION: This data extract has no Hispanic identifier; `hispanic` is set to 0.0 for all cases.")
    overview.append("")
    overview.append("Model A DV: count of dislikes among Rap, Reggae, Blues, Jazz, Gospel, Latin")
    overview.append(fitA.to_string(index=False))
    overview.append("")
    overview.append("Model B DV: count of dislikes among the other 12 genres")
    overview.append(fitB.to_string(index=False))
    overview_text = "\n".join(overview) + "\n"
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write(overview_text)

    return {
        "ModelA_paper_style": tabA_paper,
        "ModelB_paper_style": tabB_paper,
        "ModelA_full": tabA_full,
        "ModelB_full": tabB_full,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }