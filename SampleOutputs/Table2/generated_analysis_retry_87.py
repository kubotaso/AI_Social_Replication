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
        Conservative missing-code cleaning for this extract:
        - treat common GSS sentinel codes as missing (8/9, 98/99, 998/999, 9998/9999)
        - keep everything else numeric
        """
        s = to_num(x).copy()
        s = s.mask(s.isin([8, 9, 98, 99, 998, 999, 9998, 9999]))
        return s

    def likert_dislike_indicator(x):
        """
        Music items: 1-5. Dislike if 4 or 5. Like/neutral if 1,2,3.
        Anything else -> missing.
        """
        s = clean_gss_missing(x)
        s = s.where(s.between(1, 5))
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin([1, 2, 3])] = 0.0
        out.loc[s.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        s = clean_gss_missing(x)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[s.isin(false_codes)] = 0.0
        out.loc[s.isin(true_codes)] = 1.0
        return out

    def build_count_complete_case(df, items):
        """
        Sum of item-level dislike indicators; require all items observed (complete-case),
        matching "DK treated as missing and cases excluded".
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        count = mat.sum(axis=1, min_count=len(items))
        return count

    def zscore_sample(s):
        """Z-score using sample SD (ddof=1), computed on the estimation sample."""
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def standardized_betas_and_fit(df_model, y_col, x_cols, model_name, pretty_names):
        """
        Fit OLS on raw DV (so intercept stays in raw units).
        Compute standardized betas as b * sd(x) / sd(y) using the SAME estimation sample.
        Also compute p-values on unstandardized coefficients, and attach stars.
        """
        needed = [y_col] + x_cols
        d = df_model[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_cols) + 2):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_cols)}).")

        y = to_num(d[y_col]).astype(float)
        X = d[x_cols].apply(to_num).astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # Standardized betas (excluding intercept)
        sd_y = y.std(ddof=1)
        betas = {}
        for c in x_cols:
            sd_x = X[c].std(ddof=1)
            b = model.params.get(c, np.nan)
            if not np.isfinite(sd_y) or sd_y == 0 or not np.isfinite(sd_x) or sd_x == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(b * (sd_x / sd_y))

        # Stars from model p-values (two-tailed)
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

        rows = []
        for c in x_cols:
            p = float(model.pvalues.get(c, np.nan))
            rows.append(
                {
                    "variable": pretty_names.get(c, c),
                    "beta_std": betas[c],
                    "p_value_model": p,
                    "sig": stars(p),
                }
            )

        # Intercept row: unstandardized constant in DV units
        p_const = float(model.pvalues.get("const", np.nan))
        rows.append(
            {
                "variable": "Constant",
                "beta_std": np.nan,
                "p_value_model": p_const,
                "sig": stars(p_const),
            }
        )

        tab = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),  # incl intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "dv_mean": float(y.mean()),
                    "dv_sd": float(sd_y) if np.isfinite(sd_y) else np.nan,
                    "const_b": float(model.params.get("const", np.nan)),
                }
            ]
        )

        # Save human-readable outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nStandardized betas computed as: beta = b * sd(x) / sd(y) on the estimation sample.\n")

        # A compact "paper-like" table: variable + beta + stars (no SEs)
        tab_paper = tab.copy()
        tab_paper["beta_std_fmt"] = tab_paper["beta_std"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        tab_paper["reported"] = tab_paper["beta_std_fmt"] + tab_paper["sig"]
        tab_out = tab_paper[["variable", "beta_std", "sig", "reported", "p_value_model"]]

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(tab_out.to_string(index=False))

        with open(f"./output/{model_name}_fit.txt", "w", encoding="utf-8") as f:
            f.write(fit.to_string(index=False))

        return model, tab_out, fit

    # -----------------------------
    # Load + basic filtering
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Required columns missing: year and/or id")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # DVs (complete-case per DV)
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

    df["dislike_minority_genres"] = build_count_complete_case(df, minority_items)
    df["dislike_other12_genres"] = build_count_complete_case(df, other12_items)

    # -----------------------------
    # Racism score (0-5), complete-case on 5 items
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object majority-black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation
    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing educ column")
    educ = clean_gss_missing(df["educ"])
    df["education_years"] = educ.where(educ.between(0, 20))

    # HH income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing prestg80 column")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing sex column")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing age column")
    age = clean_gss_missing(df["age"])
    df["age_years"] = age.where(age.between(18, 89))

    # Race dummies from RACE (1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing race column")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not directly available in provided variables; create a usable placeholder (all 0 if non-missing ETHNIC exists)
    # This avoids the prior "all NaN -> n=0" failure while clearly documenting limitations.
    # If an actual Hispanic identifier exists in the file, prefer it automatically.
    hisp_col = None
    for cand in ["hispanic", "hispan", "hisp", "ethnic16", "hispanic16"]:
        if cand in df.columns:
            hisp_col = cand
            break

    if hisp_col is not None and hisp_col != "ethnic":
        h = clean_gss_missing(df[hisp_col])
        # If it's already 0/1 or 1/2 style, try to map: 1=yes, 2=no or 1=yes, 0=no.
        # Fallback: treat non-missing as 0 (conservative) unless it's clearly 1/2.
        hisp = pd.Series(np.nan, index=df.index, dtype="float64")
        if h.dropna().isin([0, 1]).all():
            hisp = h.astype(float)
        else:
            # common yes/no coding
            hisp = binary_from_codes(h, true_codes=[1], false_codes=[2])
        df["hispanic"] = hisp
        hisp_source_note = f"Derived from column '{hisp_col}'"
    else:
        # No usable Hispanic identifier in the provided extract
        df["hispanic"] = 0.0
        hisp_source_note = "Not available in extract; set to 0 for all cases (cannot reproduce paper's Hispanic effect)."

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing region column")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------------
    # Model specifications (Table 2)
    # -----------------------------
    rhs = [
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

    for c in rhs + ["dislike_minority_genres", "dislike_other12_genres"]:
        if c not in df.columns:
            raise ValueError(f"Missing required constructed column: {c}")

    pretty = {
        "racism_score": "Racism score",
        "education_years": "Education",
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

    # -----------------------------
    # Fit models
    # -----------------------------
    modelA, tabA, fitA = standardized_betas_and_fit(
        df_model=df,
        y_col="dislike_minority_genres",
        x_cols=rhs,
        model_name="Table2_ModelA_Dislike_Minority_Associated6",
        pretty_names=pretty,
    )
    modelB, tabB, fitB = standardized_betas_and_fit(
        df_model=df,
        y_col="dislike_other12_genres",
        x_cols=rhs,
        model_name="Table2_ModelB_Dislike_Other12_Genres",
        pretty_names=pretty,
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (GSS 1993): OLS on raw DV, standardized betas computed post-estimation.\n")
        f.write("Notes:\n")
        f.write(f"- Hispanic dummy: {hisp_source_note}\n")
        f.write("- Stars are computed from this replication model's p-values (two-tailed); the paper reports stars but not SEs.\n\n")

        f.write("Model A DV: count of disliked minority-associated genres (Rap, Reggae, Blues, Jazz, Gospel, Latin)\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\n")
        f.write(tabA.to_string(index=False))
        f.write("\n\n")

        f.write("Model B DV: count of disliked other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write(tabB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }