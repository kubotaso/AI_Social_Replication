def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_na(series):
        """
        Conservative cleaning for common GSS-style NA codes.
        We only treat a small set of obvious sentinel values as missing to avoid over-dropping.
        """
        x = to_num(series).copy()
        x = x.replace([8, 9, 98, 99, 998, 999, 9998, 9999], np.nan)
        return x

    def likert_dislike_indicator(series):
        """
        Music items: expected values 1-5.
        dislike = 1 if {4,5}; 0 if {1,2,3}; missing otherwise.
        """
        x = clean_na(series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_na(series)
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

    def build_count_completecase(df, item_cols):
        """
        Bryson-style description: DK treated as missing and missing cases excluded.
        To be faithful and simple: require non-missing response on ALL items in the count.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in item_cols], axis=1)
        # require all items observed
        return mat.sum(axis=1, min_count=len(item_cols))

    def sig_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardized_ols_beta(df, dv, xcols, add_intercept=True):
        """
        Standardized coefficients (beta): regress z(y) on z(x) with intercept.
        Intercept is not a standardized beta; we compute and report it from an unstandardized model.
        """
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        # z-scored regression for betas
        y_z = zscore(d[dv])
        X_z = pd.DataFrame({c: zscore(d[c]) for c in xcols}, index=d.index)

        # drop any degenerate columns (all nan or zero variance) AFTER listwise deletion
        keep = []
        dropped = []
        for c in X_z.columns:
            if X_z[c].isna().any():
                dropped.append(c)
                continue
            if float(X_z[c].std(ddof=0)) == 0.0:
                dropped.append(c)
                continue
            keep.append(c)
        X_z = X_z[keep]

        # align (should already align)
        ok = y_z.notna() & np.isfinite(y_z)
        y_z = y_z.loc[ok]
        X_z = X_z.loc[ok]

        if add_intercept:
            Xz_c = sm.add_constant(X_z, has_constant="add")
        else:
            Xz_c = X_z

        m_z = sm.OLS(y_z, Xz_c).fit()

        # also fit unstandardized model to get intercept on original DV scale (and R2 comparable)
        X_un = d[keep].copy()
        if add_intercept:
            Xu_c = sm.add_constant(X_un, has_constant="add")
        else:
            Xu_c = X_un
        m_un = sm.OLS(d.loc[X_un.index, dv], Xu_c).fit()

        # Build table (paper-style): standardized betas + stars, and intercept as unstandardized
        rows = []
        # predictors
        for c in keep:
            p = float(m_z.pvalues.get(c, np.nan))
            rows.append(
                {
                    "term": c,
                    "std_beta": float(m_z.params[c]),
                    "p_value_replication": p,
                    "sig": sig_stars(p),
                }
            )

        # intercept: from unstandardized model (constant on original DV scale)
        if add_intercept and "const" in m_un.params.index:
            p0 = float(m_un.pvalues.get("const", np.nan))
            rows.append(
                {
                    "term": "Constant",
                    "std_beta": np.nan,
                    "p_value_replication": p0,
                    "sig": sig_stars(p0),
                    "b_unstd": float(m_un.params["const"]),
                }
            )

        tab = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "n": int(m_un.nobs),
                    "k_predictors_included": int(len(keep)),
                    "r2": float(m_un.rsquared),
                    "adj_r2": float(m_un.rsquared_adj),
                    "dropped_predictors_post_listwise": ", ".join(dropped) if dropped else "",
                }
            ]
        )
        return tab, fit, m_z, m_un, d.index, keep

    # -------------------------
    # Load and filter
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Missing required column: YEAR/year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables: counts of dislikes
    # -------------------------
    minority6 = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12 = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority6 + other12:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority6)
    df["dislike_other12_genres"] = build_count_completecase(df, other12)

    # -------------------------
    # Racism score (0-5): require all 5 items non-missing
    # -------------------------
    needed_rac = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in needed_rac:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrim
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educ chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack motivation
    rac_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = rac_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC/educ column.")
    educ = clean_na(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing REALINC/realinc and/or HOMPOP/hompop.")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80/prestg80.")
    df["occ_prestige"] = clean_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX/sex.")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE/age.")
    age = clean_na(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: Black, Other race (White reference)
    if "race" not in df.columns:
        raise ValueError("Missing RACE/race.")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not available in provided variables; keep as missing (cannot be estimated faithfully)
    # We still create the column for reporting alignment.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy using RELIG and DENOM (as instructed)
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing RELIG/relig and/or DENOM/denom.")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = consprot.where(~(relig.isna() | denom.isna()), np.nan)
    df["cons_protestant"] = consprot

    # No religion
    norelig = (relig == 4).astype(float)
    norelig = norelig.where(~relig.isna(), np.nan)
    df["no_religion"] = norelig

    # South
    if "region" not in df.columns:
        raise ValueError("Missing REGION/region.")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -------------------------
    # Model RHS list (Table 2 order); drop Hispanic if it is all-missing (cannot be estimated)
    # -------------------------
    rhs_order = [
        ("racism_score", "Racism score"),
        ("education_years", "Education"),
        ("hh_income_per_capita", "Household income per capita"),
        ("occ_prestige", "Occupational prestige"),
        ("female", "Female"),
        ("age_years", "Age"),
        ("black", "Black"),
        ("hispanic", "Hispanic"),
        ("other_race", "Other race"),
        ("cons_protestant", "Conservative Protestant"),
        ("no_religion", "No religion"),
        ("south", "Southern"),
    ]

    # Determine which predictors are actually usable (not all-missing), but preserve paper ordering.
    usable = []
    unusable = []
    for col, label in rhs_order:
        if col not in df.columns:
            unusable.append((col, label, "missing_column"))
            continue
        if df[col].notna().sum() == 0:
            unusable.append((col, label, "all_missing"))
            continue
        usable.append((col, label))

    xcols = [c for c, _ in usable]

    # Run Model A
    tabA, fitA, mA_z, mA_un, usedA_idx, keepA = standardized_ols_beta(
        df, "dislike_minority_genres", xcols, add_intercept=True
    )
    # Run Model B
    tabB, fitB, mB_z, mB_un, usedB_idx, keepB = standardized_ols_beta(
        df, "dislike_other12_genres", xcols, add_intercept=True
    )

    # Map terms to paper labels
    label_map = {c: lbl for c, lbl in rhs_order}
    def apply_labels(tab):
        out = tab.copy()
        out["variable"] = out["term"].map(label_map).fillna(out["term"])
        # ensure order: follow paper order for predictors then Constant last
        order = [c for c, _ in rhs_order if c in out["term"].values]
        order_labels = [label_map.get(c, c) for c in order]
        # Append Constant
        order_labels = order_labels + ["Constant"]
        out["variable"] = pd.Categorical(out["variable"], categories=order_labels, ordered=True)
        out = out.sort_values(["variable", "term"], kind="stable")
        # paper-style columns
        cols = ["variable", "std_beta", "sig"]
        if "b_unstd" in out.columns:
            cols = cols + ["b_unstd"]
        cols = cols + ["p_value_replication"]
        return out[cols]

    paperA = apply_labels(tabA)
    paperB = apply_labels(tabB)

    # Save outputs
    with open("./output/Table2_notes.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993 selected extract)\n")
        f.write("Computed from microdata in provided CSV. No numbers copied from the paper.\n\n")
        f.write("Important limitations vs Bryson (1996) Table 2:\n")
        if unusable:
            f.write("- Predictors not estimable from provided extract:\n")
            for col, lbl, why in unusable:
                f.write(f"  * {lbl} ({col}): {why}\n")
        else:
            f.write("- All Table-2 predictors present and non-missing in this extract.\n")
        f.write("\nStandardized coefficients are from OLS of z(y) on z(x) with intercept.\n")
        f.write("Intercept (Constant) is reported on the original DV scale (unstandardized).\n")
        f.write("Significance stars are replication-based (*** p<.001, ** p<.01, * p<.05).\n")

    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(mA_un.summary().as_text())
        f.write("\n\nZ-scored (beta) regression summary:\n")
        f.write(mA_z.summary().as_text())
    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(mB_un.summary().as_text())
        f.write("\n\nZ-scored (beta) regression summary:\n")
        f.write(mB_z.summary().as_text())

    with open("./output/Table2_ModelA_paper_style.txt", "w", encoding="utf-8") as f:
        f.write(paperA.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n\nFit:\n")
        f.write(fitA.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    with open("./output/Table2_ModelB_paper_style.txt", "w", encoding="utf-8") as f:
        f.write(paperB.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n\nFit:\n")
        f.write(fitB.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    # also save full machine-readable csvs
    paperA.to_csv("./output/Table2_ModelA_paper_style.csv", index=False)
    paperB.to_csv("./output/Table2_ModelB_paper_style.csv", index=False)
    fitA.to_csv("./output/Table2_ModelA_fit.csv", index=False)
    fitB.to_csv("./output/Table2_ModelB_fit.csv", index=False)

    return {
        "ModelA_paper_style": paperA.reset_index(drop=True),
        "ModelB_paper_style": paperB.reset_index(drop=True),
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }