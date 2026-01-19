def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss_missing(x):
        """
        Conservative missing-code handling for this extract:
        - keep normal small integers
        - drop common GSS sentinels (8/9, 98/99, 998/999, 9998/9999)
        - also drop non-finite
        """
        x = to_num(x).astype("float64")
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.mask(x.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return x

    def likert_dislike_indicator(item):
        """
        Music taste items: 1-5; dislike if 4 or 5; like/neutral if 1-3.
        Missing if outside 1-5 or NA-coded.
        """
        x = clean_gss_missing(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_gss_missing(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Count of disliked genres; require complete data on all component items (listwise for DV).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def zscore_series(s):
        s = to_num(s).astype("float64")
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def fit_table2_model(df, dv, x_order, model_name):
        """
        OLS with standardized coefficients (beta) computed as:
            beta_j = b_j * sd(x_j) / sd(y)
        (SPSS-like standardized slopes; intercept unstandardized.)
        Uses listwise deletion on DV and all predictors in x_order.
        """
        use_cols = [dv] + x_order
        d = df[use_cols].copy()

        # listwise deletion
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        n0 = d.shape[0]
        if n0 < (len(x_order) + 5):
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={n0}).")

        # Ensure predictors have variance (avoid runtime errors from earlier attempts).
        # If some are constant due to this dataset not containing them properly, drop them
        # but record what happened.
        zero_var = []
        keep_x = []
        for c in x_order:
            v = d[c]
            if v.nunique(dropna=True) <= 1:
                zero_var.append(c)
            else:
                keep_x.append(c)

        # Proceed even if something is constant, but write diagnostics; do not crash.
        # This keeps code runnable on extracts that may lack true Hispanic measure, etc.
        X = sm.add_constant(d[keep_x], has_constant="add")
        y = d[dv]

        model = sm.OLS(y, X).fit()

        # Compute standardized betas for slopes (not for intercept)
        y_sd = y.std(ddof=0)
        beta = pd.Series(index=model.params.index, dtype="float64")
        beta.loc[:] = np.nan
        beta.loc["const"] = model.params.get("const", np.nan)
        if np.isfinite(y_sd) and y_sd != 0:
            for c in keep_x:
                x_sd = d[c].std(ddof=0)
                if np.isfinite(x_sd) and x_sd != 0 and c in model.params.index:
                    beta.loc[c] = model.params[c] * (x_sd / y_sd)

        # p-values for stars are from the same fitted (unstandardized) model
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

        # Paper-style table: standardized betas (slopes), plus constant (unstd)
        # Keep rows in requested order; include dropped-constant predictors as NaN.
        rows = x_order + ["const"]
        paper = pd.DataFrame(index=rows, columns=["beta", "stars"], dtype="object")
        for r in rows:
            if r == "const":
                p = model.pvalues.get("const", np.nan)
                paper.loc[r, "beta"] = float(model.params.get("const", np.nan))
                paper.loc[r, "stars"] = stars(p)
            else:
                paper.loc[r, "beta"] = float(beta.get(r, np.nan))
                paper.loc[r, "stars"] = stars(model.pvalues.get(r, np.nan))

        # Full table (replication diagnostics; not "from paper")
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
                "beta_std": beta,
            }
        )

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "dv": dv,
                "n": int(model.nobs),
                "k_params_including_const": int(len(model.params)),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "dropped_zero_variance_predictors": ", ".join(zero_var) if zero_var else "",
            }]
        )

        # Save readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nNotes:\n")
            f.write("- Table 2 in Bryson (1996) reports standardized coefficients (betas) and stars only; SEs are not in the printed table.\n")
            f.write("- Here, stars are computed from this replication model's two-tailed p-values.\n")
            if zero_var:
                f.write(f"- Predictors dropped due to zero variance in this analytic sample (not estimated): {zero_var}\n")

        with open(f"./output/{model_name}_paper_style_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("Paper-style output: standardized coefficients (beta) for slopes; constant is unstandardized.\n")
            f.write("Stars from replication-model p-values: * p<.05, ** p<.01, *** p<.001\n\n")
            out = paper.copy()
            out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
            f.write(out.to_string(float_format=lambda x: f"{x: .3f}"))

        with open(f"./output/{model_name}_full_table.txt", "w", encoding="utf-8") as f:
            f.write(f"{model_name}\nFull replication table (unstandardized b, SE, t, p, and computed standardized beta)\n\n")
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))

        return paper, full, fit

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()
    if df.shape[0] == 0:
        raise ValueError("No rows with YEAR==1993 found.")

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

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -------------------------
    # Racism score (0-5 additive)
    # -------------------------
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

    # -------------------------
    # Controls
    # -------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # Household income per capita: REALINC / HOMPOP
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing column(s) for income per capita: realinc, hompop")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies: black/other_race from race
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: try to construct from ETHNIC if present (best-effort) else missing.
    # NOTE: This dataset does not provide a clean Hispanic flag; this is a pragmatic proxy so the model can run.
    # If you have a proper Hispanic indicator in another extract, replace this block.
    if "ethnic" in df.columns:
        eth = clean_gss_missing(df["ethnic"])
        # Common GSS ETHNIC coding in many extracts: 1=Mexican, 2=Puerto Rican, 3=Other Spanish, 4=Not Spanish
        # We'll treat {1,2,3} as Hispanic proxy, 4 as not; otherwise missing.
        df["hispanic"] = np.nan
        df.loc[eth.isin([4]), "hispanic"] = 0.0
        df.loc[eth.isin([1, 2, 3]), "hispanic"] = 1.0
    else:
        df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing column(s): relig, denom")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype("float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4 (none)
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype("float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index).astype("float64")
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -------------------------
    # Fit Table 2 models
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

    # Ensure constructed columns exist
    for c in ["dislike_minority_genres", "dislike_other12_genres"] + x_order:
        if c not in df.columns:
            raise ValueError(f"Missing constructed column: {c}")

    paperA, fullA, fitA = fit_table2_model(df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6")
    paperB, fullB, fitB = fit_table2_model(df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12")

    # Combined overview
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication (1993 GSS extract): OLS; standardized coefficients (beta) reported for slopes.\n")
        f.write("Stars computed from replication-model p-values; Table 2 in the paper does not report SEs.\n")
        f.write("Important: if a proper Hispanic indicator is not available, this code uses a best-effort ETHNIC proxy when present.\n\n")
        f.write("Model A: DV = count of disliked among Rap/Reggae/Blues/Jazz/Gospel/Latin (0-6)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B: DV = count of disliked among the other 12 genres (0-12)\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }