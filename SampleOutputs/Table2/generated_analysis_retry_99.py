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
        Conservative NA handling for common GSS-style special codes.
        This subset extract is numeric; we convert known sentinel values to NaN.
        """
        x = to_num(x).copy()
        x = x.replace(
            {
                8: np.nan, 9: np.nan,
                98: np.nan, 99: np.nan,
                998: np.nan, 999: np.nan,
                9998: np.nan, 9999: np.nan,
            }
        )
        return x

    def likert_dislike_indicator(item):
        """
        Music items: 1-5 scale. Dislike if 4 or 5.
        Treat anything outside 1..5 as missing.
        """
        x = clean_gss_missing(item)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(series, true_codes, false_codes):
        x = clean_gss_missing(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_count_completecase(df, items):
        """
        Count of dislikes across items.
        To align with 'DK treated as missing and missing cases excluded', require complete data
        across all component items for the DV.
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        return mat.sum(axis=1, min_count=len(items))

    def zscore_sample(s):
        s = to_num(s)
        mu = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def standardized_ols(df, dv, xcols, model_name, label_map):
        """
        Returns:
          - fitted model on unstandardized DV and unstandardized X (for intercept on DV scale, R2)
          - standardized betas computed via z-scoring y and X (intercept included but not compared)
          - also writes human-readable summaries.
        """
        needed = [dv] + xcols
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # validate predictors have variance in estimation sample
        zero_var = [c for c in xcols if d[c].nunique(dropna=True) < 2]
        if zero_var:
            raise ValueError(f"{model_name}: zero-variance predictors in estimation sample: {zero_var}")

        # Unstandardized model (for intercept & R2 on DV scale)
        Xu = sm.add_constant(d[xcols], has_constant="add")
        yu = d[dv]
        model_u = sm.OLS(yu, Xu).fit()

        # Standardized model for betas
        yz = zscore_sample(yu)
        Xz = pd.DataFrame({c: zscore_sample(d[c]) for c in xcols}, index=d.index)

        # ensure no NaNs introduced by zscore (shouldn't, because we already dropped NA)
        bad = [c for c in xcols if not np.isfinite(Xz[c]).all()]
        if bad:
            raise ValueError(f"{model_name}: predictors became undefined after standardization: {bad}")
        if not np.isfinite(yz).all():
            raise ValueError(f"{model_name}: DV became undefined after standardization")

        Xz_c = sm.add_constant(Xz, has_constant="add")
        model_z = sm.OLS(yz, Xz_c).fit()

        # Build labeled table: standardized betas for predictors, plus unstandardized intercept
        rows = []
        # intercept (unstandardized, on DV scale)
        rows.append(
            {
                "term": "Intercept",
                "label": "Constant",
                "value": float(model_u.params["const"]),
                "type": "unstandardized_intercept",
            }
        )
        for c in xcols:
            rows.append(
                {
                    "term": c,
                    "label": label_map.get(c, c),
                    "value": float(model_z.params[c]),
                    "type": "standardized_beta",
                }
            )
        tab = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model_u.nobs),
                    "k_predictors": int(len(xcols)),
                    "r2": float(model_u.rsquared),
                    "adj_r2": float(model_u.rsquared_adj),
                }
            ]
        )

        # Save outputs
        with open(f"./output/{model_name}_summary_unstandardized.txt", "w", encoding="utf-8") as f:
            f.write(model_u.summary().as_text())

        with open(f"./output/{model_name}_summary_standardized.txt", "w", encoding="utf-8") as f:
            f.write(model_z.summary().as_text())

        with open(f"./output/{model_name}_table.txt", "w", encoding="utf-8") as f:
            f.write(
                tab.sort_values(["type", "label"])
                .to_string(index=False, float_format=lambda x: f"{x: .6f}")
            )
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

        return tab, fit, model_u, model_z

    # -----------------------------
    # Load data and normalize names
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: YEAR")
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

    df["dislike_minority_genres"] = build_count_completecase(df, minority_items)
    df["dislike_other12_genres"] = build_count_completecase(df, other12_items)

    # -----------------------------
    # Racism score (0-5)
    # -----------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation
    df["racism_score"] = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1).sum(axis=1, min_count=5)

    # -----------------------------
    # Controls
    # -----------------------------
    # Education (years)
    if "educ" not in df.columns:
        raise ValueError("Missing EDUC column (educ)")
    educ = clean_gss_missing(df["educ"]).where(lambda x: x.between(0, 20))
    df["education_years"] = educ

    # Household income per capita: REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing required income component: {c}")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(lambda x: x > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing PRESTG80 column (prestg80)")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing SEX column (sex)")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing AGE column (age)")
    df["age_years"] = clean_gss_missing(df["age"]).where(lambda x: x.between(18, 89))

    # Race dummies from RACE (1 white, 2 black, 3 other). White is reference.
    if "race" not in df.columns:
        raise ValueError("Missing RACE column (race)")
    race = clean_gss_missing(df["race"]).where(lambda x: x.isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: not present in provided variable list; do not proxy.
    # Keep as all-missing and exclude from models explicitly.
    df["hispanic"] = np.nan

    # Religion: conservative Protestant proxy, and no religion
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing religion/denomination column: {c}")
    relig = clean_gss_missing(df["relig"])
    denom = clean_gss_missing(df["denom"])

    # Conservative Protestant proxy: RELIG==1 (Protestant) and DENOM in {1,6,7}
    df["cons_protestant"] = np.where(
        relig.isna() | denom.isna(),
        np.nan,
        ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float),
    )

    # No religion: RELIG==4
    df["no_religion"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # South: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing REGION column (region)")
    region = clean_gss_missing(df["region"]).where(lambda x: x.isin([1, 2, 3, 4]))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # -----------------------------
    # Model specification
    # -----------------------------
    # IMPORTANT: Hispanic dummy cannot be included with this extract (all-missing -> would drop all rows).
    xcols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    label_map = {
        "racism_score": "Racism score (0–5)",
        "education_years": "Education (years)",
        "hh_income_per_capita": "Household income per capita",
        "occ_prestige": "Occupational prestige (PRESTG80)",
        "female": "Female (1=female)",
        "age_years": "Age",
        "black": "Black",
        "other_race": "Other race",
        "cons_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "south": "Southern",
    }

    # Ensure no_religion varies before model fit; if not, fail with a clear message
    # (This addresses the runtime error seen previously.)
    # We'll check variance on the broad 1993 subset first, then within each model sample it is checked again.
    if df["no_religion"].dropna().nunique() < 2:
        raise ValueError(
            "no_religion has zero variance after cleaning in YEAR==1993 subset; "
            "cannot estimate Table 2 specification. Check RELIG coding in the input extract."
        )

    # -----------------------------
    # Fit both models
    # -----------------------------
    tabA, fitA, modelA_u, modelA_z = standardized_ols(
        df,
        dv="dislike_minority_genres",
        xcols=xcols,
        model_name="Table2_ModelA_dislike_minority6",
        label_map=label_map,
    )

    tabB, fitB, modelB_u, modelB_z = standardized_ols(
        df,
        dv="dislike_other12_genres",
        xcols=xcols,
        model_name="Table2_ModelB_dislike_other12",
        label_map=label_map,
    )

    # Combined overview
    overview = []
    overview.append("Table 2 replication attempt (GSS 1993): OLS with standardized coefficients (betas).")
    overview.append("Notes:")
    overview.append("- Standardized betas computed by running OLS on z-scored DV and z-scored predictors.")
    overview.append("- Constant reported is the unstandardized intercept (OLS on DV scale).")
    overview.append("- This extract does not include a direct Hispanic identifier; the Hispanic dummy is not estimated.")
    overview.append("")
    overview.append("Model A DV: dislike_minority_genres (Rap, Reggae, Blues, Jazz, Gospel, Latin) count")
    overview.append(fitA.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    overview.append("")
    overview.append("Model B DV: dislike_other12_genres (12 remaining genres) count")
    overview.append(fitB.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    overview_text = "\n".join(overview)

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write(overview_text)

    results = {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }
    return results