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

    def clean_na(series):
        """
        Conservative missing-code handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        Note: We do NOT assume any particular weight variable exists.
        """
        x = to_num(series).copy()
        x = x.replace([8, 9, 98, 99, 998, 999, 9998, 9999], np.nan)
        return x

    def likert_dislike(series):
        """
        Music taste items: 1-5.
        Dislike indicator: 1 if in {4,5}, 0 if in {1,2,3}, else missing.
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

    def build_count(df, items):
        mat = pd.concat([likert_dislike(df[c]).rename(c) for c in items], axis=1)
        # Bryson: DK treated as missing; simplest faithful rule: require all items observed for the DV
        return mat.sum(axis=1, min_count=len(items))

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
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

    def fit_table2_model(df, dv, x_order, model_name):
        """
        Fit OLS with:
        - Unstandardized model for intercept, R2, etc.
        - Standardized coefficients (betas) computed by z-scoring DV and all X (including dummies)
        Stars computed from the standardized regression p-values (two-tailed).
        """
        needed = [dv] + x_order
        d = df[needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

        # Guard against empty sample
        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_order)}).")

        # Ensure predictors have variance; if any are constant, raise clearly (do not silently drop)
        zero_var = []
        for c in x_order:
            v = to_num(d[c]).astype(float)
            if v.nunique(dropna=True) <= 1:
                zero_var.append(c)
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Fix coding/sample so all listed Table 2 predictors vary."
            )

        # Unstandardized model (for intercept in DV units)
        y_un = to_num(d[dv]).astype(float)
        X_un = d[x_order].apply(to_num).astype(float)
        X_un = sm.add_constant(X_un, has_constant="add")
        m_un = sm.OLS(y_un, X_un).fit()

        # Standardized model (for betas)
        y = zscore(d[dv])
        Xz = pd.DataFrame({c: zscore(d[c]) for c in x_order}, index=d.index)
        # after zscore, ensure no NaNs produced (shouldn't be, given variance checks)
        dz = pd.concat([y.rename(dv), Xz], axis=1).dropna(axis=0, how="any")
        y = dz[dv]
        Xz = dz[x_order]
        Xz = sm.add_constant(Xz, has_constant="add")
        m_std = sm.OLS(y, Xz).fit()

        # Build "paper-style" table: standardized betas + stars; include constant unstandardized
        rows = []
        for term in x_order:
            beta = float(m_std.params.get(term, np.nan))
            p = float(m_std.pvalues.get(term, np.nan))
            rows.append(
                {
                    "term": term,
                    "beta_std": beta,
                    "sig": star_from_p(p),
                    "p_value": p,
                }
            )

        # Constant: report from unstandardized model (not standardized)
        const_b = float(m_un.params.get("const", np.nan))
        const_p = float(m_un.pvalues.get("const", np.nan))
        rows.append(
            {
                "term": "const",
                "beta_std": np.nan,
                "sig": star_from_p(const_p),
                "p_value": const_p,
                "b_unstd_const": const_b,
            }
        )

        table = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv,
                    "n": int(m_un.nobs),
                    "k_predictors": int(m_un.df_model),  # excludes intercept
                    "r2": float(m_un.rsquared),
                    "adj_r2": float(m_un.rsquared_adj),
                    "const_unstd": const_b,
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary_unstandardized.txt", "w", encoding="utf-8") as f:
            f.write(m_un.summary().as_text())
            f.write("\n\nNOTE: Table 2 in the paper reports standardized coefficients only; SEs are not shown there.\n")

        with open(f"./output/{model_name}_summary_standardized.txt", "w", encoding="utf-8") as f:
            f.write(m_std.summary().as_text())
            f.write("\n\nNOTE: Betas correspond to slopes from a regression on z-scored DV and z-scored predictors.\n")

        # Paper-style table (betas + stars)
        paper_style = table[["term", "beta_std", "sig"]].copy()
        paper_style.to_string(
            open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8"),
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )

        # Full table for auditing (includes p-values we compute from microdata)
        full = table.copy()
        full.to_string(
            open(f"./output/{model_name}_table_full_audit.txt", "w", encoding="utf-8"),
            index=False,
            float_format=lambda x: f"{x: .6f}",
        )

        fit.to_string(open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8"), index=False)

        return paper_style, full, fit

    # -----------------------------
    # Load & normalize
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["year", "id"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Construct DVs (exact genre lists from mapping)
    # -----------------------------
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

    df["dislike_minority_genres"] = build_count(df, minority_items)
    df["dislike_other12_genres"] = build_count(df, other12_items)

    # -----------------------------
    # Racism score (0-5), complete on all five items
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
    # Controls (as available in this extract)
    # -----------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ).")
    df["education_years"] = clean_na(df["educ"]).where(clean_na(df["educ"]).between(0, 20))

    # HH income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_na(df["realinc"])
    hompop = clean_na(df["hompop"]).where(clean_na(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80).")
    df["occ_prestige"] = clean_na(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex).")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age).")
    df["age_years"] = clean_na(df["age"]).where(clean_na(df["age"]).between(18, 89))

    # Race dummies: black, other_race; white omitted
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race).")
    race = clean_na(df["race"]).where(clean_na(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator not present in provided extract: create NA and exclude from model
    # (We must not proxy using ETHNIC per instruction.)
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_na(df["relig"])
    denom = clean_na(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region).")
    region = clean_na(df["region"]).where(clean_na(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Models (Table 2): two DVs, same RHS
    # Note: Hispanic cannot be included because the extract provides no direct Hispanic flag.
    # -----------------------------
    x_order = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # unavailable -> cannot estimate
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Diagnostics: write frequencies for key dummies (helps catch zero-variance issues)
    diag_lines = []
    diag_lines.append(f"Rows after YEAR==1993: {len(df)}\n")
    for v in ["female", "black", "other_race", "cons_protestant", "no_religion", "south"]:
        vc = df[v].value_counts(dropna=False).sort_index()
        diag_lines.append(f"{v} value_counts (incl NA):\n{vc.to_string()}\n")
    with open("./output/diagnostics_basic.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(diag_lines))

    # Fit both
    paperA, fullA, fitA = fit_table2_model(df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6")
    paperB, fullB, fitB = fit_table2_model(df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12")

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993 extract)\n")
        f.write("Outputs include:\n")
        f.write("- paper-style tables: standardized betas (slopes from z-scored regression) + stars based on computed p-values\n")
        f.write("- audit tables: include p-values from microdata (Table 2 in the paper does not list SEs/p-values)\n")
        f.write("\nIMPORTANT LIMITATION:\n")
        f.write("This extract does not include a direct Hispanic indicator, so the Table 2 'Hispanic' dummy cannot be estimated here.\n")
        f.write("As a result, estimates may not match the published table.\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table_paper_style": paperA,
        "ModelB_table_paper_style": paperB,
        "ModelA_table_full_audit": fullA,
        "ModelB_table_full_audit": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }