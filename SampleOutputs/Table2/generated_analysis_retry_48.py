def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Helpers
    # -----------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_missing(x):
        """
        Minimal, robust missing cleaning for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        """
        x = to_num(x).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinel))
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items are 1-5:
          1/2/3 => not disliked (0)
          4/5   => disliked (1)
        Anything else (including DK/refused/sentinels) => missing.
        """
        x = clean_missing(x)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_missing(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Count of dislikes across items, requiring all items observed (complete-case)
        to mirror "DK treated as missing and cases excluded" in DV construction.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def stars_from_p(p):
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
        Fit OLS on unstandardized variables (with intercept).
        Compute standardized betas for slopes only:
            beta_j = b_j * sd(x_j) / sd(y)
        using the analytic sample used in the regression.
        Output "paper-style" table: standardized betas + stars + constant (unstd).
        """
        needed = [dv] + x_order
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        if d.shape[0] < (len(x_order) + 2):
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # Ensure no constant predictors in the analytic sample
        zero_var = []
        for c in x_order:
            v = d[c]
            if v.nunique(dropna=True) <= 1:
                zero_var.append(c)
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Check coding/sample restrictions."
            )

        y = d[dv].astype(float)
        X = d[x_order].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas for slopes (not intercept)
        sd_y = y.std(ddof=0)
        betas = {}
        for c in x_order:
            sd_x = X[c].std(ddof=0)
            betas[c] = model.params[c] * (sd_x / sd_y) if (sd_x > 0 and sd_y > 0) else np.nan

        # Build outputs
        full = pd.DataFrame(
            {
                "term": model.params.index,
                "b_unstd": model.params.values,
                "p_value": model.pvalues.values,
            }
        )

        paper_rows = []
        for c in x_order:
            paper_rows.append(
                {"term": c, "beta_std": betas[c], "stars": stars_from_p(model.pvalues.get(c, np.nan))}
            )
        # Constant row: show unstandardized intercept (as the paper does) with stars; beta not meaningful
        const_p = model.pvalues.get("const", np.nan)
        paper_rows.append(
            {"term": "Constant", "beta_std": np.nan, "stars": stars_from_p(const_p)}
        )
        paper = pd.DataFrame(paper_rows)

        # Attach constant value separately for human readability
        const_val = float(model.params.get("const", np.nan))

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv,
                    "n": int(model.nobs),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "constant_unstd": const_val,
                }
            ]
        )

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Table 2-style output reports standardized betas for slopes (beta_j = b_j * sd(x)/sd(y)).\n")
            f.write("- Intercept is reported unstandardized; standardized beta for intercept is not meaningful.\n")

        # Save paper-style table as fixed-width text
        paper_disp = paper.copy()
        # Add coefficient display column
        def fmt_beta(x):
            if pd.isna(x):
                return ""
            return f"{x:.3f}"

        paper_disp["beta(std)"] = paper_disp["beta_std"].map(fmt_beta) + paper_disp["stars"]
        paper_disp = paper_disp[["term", "beta(std)"]]
        # Put constant value in the term row via separate file for clarity
        with open(f"./output/{model_name}_Table2_style.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write(f"DV: {dv}\n")
            f.write(f"N={int(model.nobs)}  R2={model.rsquared:.3f}  AdjR2={model.rsquared_adj:.3f}\n\n")
            f.write("Standardized OLS coefficients (slopes) with stars; Constant shown as unstandardized:\n")
            f.write("(Stars based on two-tailed p-values from this re-estimation: * p<.05, ** p<.01, *** p<.001)\n\n")
            # Print slopes
            for _, r in paper_disp.iterrows():
                if r["term"] == "Constant":
                    continue
                f.write(f"{r['term']:<22} {r['beta(std)']:>10}\n")
            # Constant line
            f.write(f"\n{'Constant (unstd)':<22} {const_val:>10.3f}{stars_from_p(const_p)}\n")

        # Save full coefficient table (for diagnostics), but do not present as "Table 2"
        full_out = full.copy()
        full_out.to_string(
            open(f"./output/{model_name}_full_coeffs.txt", "w", encoding="utf-8"),
            index=False,
            float_format=lambda z: f"{z: .6g}",
        )

        return model, paper, full, fit, d.index

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns
    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must include columns: year, id")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.shape[0] == 0:
        raise ValueError("No records after filtering to YEAR==1993.")

    # -----------------------
    # DVs (complete-case within each DV)
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing required music item column: {c}")

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------
    # Racism score (0-5), complete-case on all five components
    # -----------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing required racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls
    # -----------------------
    # Education years
    if "educ" not in df.columns:
        raise ValueError("Missing required column: educ")
    educ = clean_missing(df["educ"]).where(clean_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita = realinc / hompop
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    realinc = clean_missing(df["realinc"])
    hompop = clean_missing(df["hompop"]).where(clean_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occ prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing required column: prestg80")
    df["occ_prestige"] = clean_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing required column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing required column: age")
    df["age_years"] = clean_missing(df["age"]).where(clean_missing(df["age"]).between(18, 89))

    # Race dummies (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing required column: race")
    race = clean_missing(df["race"]).where(clean_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not present in this extract; per instructions do NOT proxy using ETHNIC.
    # Include as all-missing so models can still run only if you remove it;
    # but Table 2 requires it, so we will include it only if a proper column exists.
    hisp_col_candidates = ["hispanic", "hispan", "hisp", "hispanic16", "hispan16"]
    hisp_col = None
    for c in hisp_col_candidates:
        if c in df.columns:
            hisp_col = c
            break
    if hisp_col is None:
        # create explicit missing column, but do not include in x list
        df["hispanic"] = np.nan
        hispanic_available = False
    else:
        # try interpret as 1/0 or 1/2; keep conservative
        hx = clean_missing(df[hisp_col])
        # If values look like {1,2}, assume 1=yes 2=no; else if {0,1}, assume as-is
        if set(hx.dropna().unique()).issubset({0, 1}):
            df["hispanic"] = hx.astype(float)
        elif set(hx.dropna().unique()).issubset({1, 2}):
            df["hispanic"] = binary_from_codes(hx, true_codes=[1], false_codes=[2])
        else:
            # unknown coding; treat nonzero as 1, zero as 0
            df["hispanic"] = np.where(hx.isna(), np.nan, (hx != 0).astype(float))
        hispanic_available = True

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    relig = clean_missing(df["relig"])
    denom = clean_missing(df["denom"])
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(), np.nan, ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing required column: region")
    region = clean_missing(df["region"]).where(clean_missing(df["region"]).isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------
    # Model specs (Table 2 order)
    # -----------------------
    # Always keep table-order template (even if hispanic missing in this extract)
    table2_order = [
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

    x_order = [v for v, _ in table2_order if (v != "hispanic" or hispanic_available)]

    # -----------------------
    # Fit models
    # -----------------------
    mA, paperA, fullA, fitA, idxA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6"
    )
    mB, paperB, fullB, fitB, idxB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12"
    )

    # -----------------------
    # Build Table 2-like tables with proper labels and fixed order (no row-position errors)
    # -----------------------
    def build_labeled_table(paper_df, fit_df, model_label):
        # Map internal term to display label
        label_map = {k: v for k, v in table2_order}
        # Build template in paper order
        rows = []
        for internal, disp in table2_order:
            if internal == "hispanic" and not hispanic_available:
                rows.append({"Variable": disp, "Beta (std)": ""})
                continue
            r = paper_df.loc[paper_df["term"] == internal]
            if r.shape[0] == 0:
                rows.append({"Variable": disp, "Beta (std)": ""})
            else:
                beta = r["beta_std"].iloc[0]
                st = r["stars"].iloc[0]
                beta_str = "" if pd.isna(beta) else f"{beta:.3f}{st}"
                rows.append({"Variable": disp, "Beta (std)": beta_str})

        # Constant row
        const_val = float(fit_df["constant_unstd"].iloc[0])
        const_stars = paper_df.loc[paper_df["term"] == "Constant", "stars"].iloc[0]
        rows.append({"Variable": "Constant (unstd)", "Beta (std)": f"{const_val:.3f}{const_stars}"})

        out = pd.DataFrame(rows)
        out.attrs["model"] = model_label
        out.attrs["n"] = int(fit_df["n"].iloc[0])
        out.attrs["r2"] = float(fit_df["r2"].iloc[0])
        out.attrs["adj_r2"] = float(fit_df["adj_r2"].iloc[0])
        return out

    tableA = build_labeled_table(paperA, fitA, "Model A: Dislike minority-associated genres (6)")
    tableB = build_labeled_table(paperB, fitB, "Model B: Dislike other genres (12)")

    # Save combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Replication output for Table 2 (computed from provided microdata extract)\n")
        f.write("Outputs are standardized betas (slopes) with significance stars based on two-tailed p-values:\n")
        f.write("  * p<.05, ** p<.01, *** p<.001\n")
        if not hispanic_available:
            f.write("\nNOTE: A proper Hispanic identifier is not present in this data extract.\n")
            f.write("The 'Hispanic' row is left blank and was not included in estimation.\n")
            f.write("To fully match Table 2 specification, provide a dataset with a Hispanic dummy.\n")
        f.write("\n\n--- Model A ---\n")
        f.write(f"N={tableA.attrs['n']}  R2={tableA.attrs['r2']:.3f}  AdjR2={tableA.attrs['adj_r2']:.3f}\n\n")
        f.write(tableA.to_string(index=False))
        f.write("\n\n--- Model B ---\n")
        f.write(f"N={tableB.attrs['n']}  R2={tableB.attrs['r2']:.3f}  AdjR2={tableB.attrs['adj_r2']:.3f}\n\n")
        f.write(tableB.to_string(index=False))
        f.write("\n")

    # Also save the labeled tables as standalone text
    with open("./output/Table2_ModelA_Table2_style_labeled.txt", "w", encoding="utf-8") as f:
        f.write(f"N={tableA.attrs['n']}  R2={tableA.attrs['r2']:.3f}  AdjR2={tableA.attrs['adj_r2']:.3f}\n\n")
        f.write(tableA.to_string(index=False))
        f.write("\n")
    with open("./output/Table2_ModelB_Table2_style_labeled.txt", "w", encoding="utf-8") as f:
        f.write(f"N={tableB.attrs['n']}  R2={tableB.attrs['r2']:.3f}  AdjR2={tableB.attrs['adj_r2']:.3f}\n\n")
        f.write(tableB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_Table2_style": tableA,
        "ModelB_Table2_style": tableB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
        "ModelA_full_coeffs": fullA,
        "ModelB_full_coeffs": fullB,
    }