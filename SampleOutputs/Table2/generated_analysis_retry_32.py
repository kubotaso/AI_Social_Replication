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

    def clean_gss_missing(x):
        """
        Conservative missing-code handling for this extract:
        - Coerce to numeric
        - Treat common GSS sentinel codes as missing
        - Treat negatives as missing (often 'not applicable' codes)
        """
        x = to_num(x)
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(sentinels)))
        x = x.mask(x < 0)
        return x

    def likert_dislike_indicator(x):
        """
        Music taste items: 1-5; dislike if 4 or 5.
        DK/refused/etc -> missing (handled by clean_gss_missing + range check).
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

    def build_dislike_count(df, items, require_all=True):
        cols = []
        for c in items:
            if c not in df.columns:
                raise ValueError(f"Missing required music item column: {c}")
            cols.append(likert_dislike_indicator(df[c]).rename(c))
        mat = pd.concat(cols, axis=1)
        if require_all:
            return mat.sum(axis=1, min_count=len(items))
        else:
            # Not used here; kept for completeness
            return mat.sum(axis=1, min_count=1)

    def wmean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return np.sum(w[m] * x[m]) / np.sum(w[m])

    def wvar(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        mu = np.sum(w[m] * x[m]) / np.sum(w[m])
        return np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])

    def wzscore(series, w):
        x = np.asarray(series, dtype=float)
        ww = np.asarray(w, dtype=float)
        mu = wmean(x, ww)
        vv = wvar(x, ww)
        sd = np.sqrt(vv) if np.isfinite(vv) else np.nan
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=series.index)
        return pd.Series((x - mu) / sd, index=series.index)

    def fit_table2_model(df, dv, x_order, model_name, w_col=None):
        """
        - Listwise deletion on dv + x_order (+ weight if provided)
        - Fit unstandardized OLS/WLS to get intercept on original DV scale
        - Compute standardized betas by fitting OLS/WLS on z-scored y and x (intercept included)
        - Return:
            paper_style_table: beta (standardized) + stars; constant reported as unstandardized intercept
            full_table: unstandardized coefficients + SE + p, plus standardized beta
            fit_df: N, R2, adjR2
        """
        needed = [dv] + x_order + ([w_col] if w_col else [])
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if d.shape[0] < (len(x_order) + 5):
            raise ValueError(f"{model_name}: not enough complete cases (n={d.shape[0]}, k={len(x_order)}).")

        # Check for zero variance among predictors in analytic sample (paper includes them; if zero, sample/coding is wrong)
        zero_var = []
        for c in x_order:
            v = d[c].astype(float).values
            if np.nanstd(v, ddof=0) == 0:
                zero_var.append(c)
        if zero_var:
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}. "
                f"Fix coding/sample so all listed Table 2 predictors vary."
            )

        # Weights (optional)
        if w_col:
            w = clean_gss_missing(d[w_col]).astype(float)
            w = w.where(w > 0)
            d = d.loc[w.notna()].copy()
            w = w.loc[d.index].astype(float)
            if d.shape[0] < (len(x_order) + 5):
                raise ValueError(f"{model_name}: not enough cases after weight cleaning (n={d.shape[0]}).")
        else:
            w = None

        y = d[dv].astype(float)
        X = d[x_order].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        # Unstandardized fit for intercept and reference
        if w is None:
            m_un = sm.OLS(y, Xc).fit()
        else:
            m_un = sm.WLS(y, Xc, weights=w).fit()

        # Standardized betas: regress z(y) on z(x) with intercept
        if w is None:
            yz = (y - y.mean()) / y.std(ddof=0)
            Xz = (X - X.mean()) / X.std(ddof=0)
        else:
            yz = wzscore(y, w)
            Xz = pd.DataFrame({c: wzscore(X[c], w) for c in X.columns}, index=X.index)

        # If any standardization produced NaNs (shouldn't if variance checks passed), drop just in case
        dz = pd.concat([yz.rename("y")] + [Xz[c].rename(c) for c in Xz.columns], axis=1)
        dz = dz.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        yz2 = dz["y"]
        Xz2 = dz[x_order]
        Xzc = sm.add_constant(Xz2, has_constant="add")

        if w is None:
            m_std = sm.OLS(yz2, Xzc).fit()
        else:
            w2 = w.loc[dz.index]
            m_std = sm.WLS(yz2, Xzc, weights=w2).fit()

        # Build full table with names (auditable)
        full = pd.DataFrame(
            {
                "b_unstd": m_un.params,
                "std_err": m_un.bse,
                "t": m_un.tvalues,
                "p_value": m_un.pvalues,
                "beta_std": m_std.params.reindex(m_un.params.index),
            }
        )
        full.index.name = "term"

        # Stars (computed from microdata; table 2 uses stars, but we compute them here)
        def star(p):
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        # Paper-style: standardized betas + stars; intercept as unstandardized constant
        rows = []
        for term in ["const"] + x_order:
            if term == "const":
                coef = float(m_un.params.get("const", np.nan))
                p = float(m_un.pvalues.get("const", np.nan))
                rows.append(("Constant", coef, star(p)))
            else:
                coef = float(m_std.params.get(term, np.nan))
                p = float(m_un.pvalues.get(term, np.nan))  # stars based on the same model's inference
                rows.append((term, coef, star(p)))

        paper_style = pd.DataFrame(rows, columns=["term", "coef", "sig"])
        paper_style = paper_style.set_index("term")

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(m_un.nobs),
                    "k_predictors": int(m_un.df_model),  # excludes intercept
                    "r2": float(m_un.rsquared),
                    "adj_r2": float(m_un.rsquared_adj),
                }
            ]
        )

        # Save text outputs
        with open(f"./output/{model_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(m_un.summary().as_text())
            f.write("\n\nStandardized-betas model (z-scored y and x) summary:\n")
            f.write(m_std.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        with open(f"./output/{model_name}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        with open(f"./output/{model_name}_table_paper_style.txt", "w", encoding="utf-8") as f:
            # show betas for predictors; constant is unstandardized
            f.write(paper_style.to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n")

        return paper_style, full, fit

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

    # -------------------------
    # Dependent variables (counts)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    df["dislike_minority_genres"] = build_dislike_count(df, minority_items, require_all=True)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items, require_all=True)

    # -------------------------
    # Racism score (0-5)
    # -------------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing required racism component column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # Controls
    # -------------------------
    # Education years (0-20)
    if "educ" not in df.columns:
        raise ValueError("Missing required column: educ")
    educ = clean_gss_missing(df["educ"]).where(clean_gss_missing(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita = REALINC / HOMPOP (HOMPOP > 0)
    if "realinc" not in df.columns or "hompop" not in df.columns:
        raise ValueError("Missing required column(s): realinc and/or hompop")
    realinc = clean_gss_missing(df["realinc"])
    hompop = clean_gss_missing(df["hompop"]).where(clean_gss_missing(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing required column: prestg80")
    df["occ_prestige"] = clean_gss_missing(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing required column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing required column: age")
    df["age_years"] = clean_gss_missing(df["age"]).where(clean_gss_missing(df["age"]).between(18, 89))

    # Race dummies (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing required column: race")
    race = clean_gss_missing(df["race"]).where(clean_gss_missing(df["race"]).isin([1, 2, 3]))
    df["black"] = pd.Series(np.where(race == 2, 1.0, 0.0), index=df.index)
    df.loc[race.isna(), "black"] = np.nan
    df["other_race"] = pd.Series(np.where(race == 3, 1.0, 0.0), index=df.index)
    df.loc[race.isna(), "other_race"] = np.nan

    # Hispanic: not available in provided variables; do NOT proxy. Keep as missing.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 (Protestant) & DENOM in {1,6,7}
    if "relig" not in df.columns or "denom" not in df.columns:
        raise ValueError("Missing required column(s): relig and/or denom")
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
        raise ValueError("Missing required column: region")
    region = clean_gss_missing(df["region"]).where(clean_gss_missing(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -------------------------
    # Models (Table 2)
    # -------------------------
    # Keep exact Table-2 RHS order; include hispanic even if missing in this extract (will reduce N to 0 if all missing),
    # so we handle it explicitly: if all-missing, we omit it, but we write this to overview for auditability.
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

    # If a predictor is entirely missing in the dataset, drop it (only acceptable for hispanic per instructions).
    dropped_for_all_missing = []
    for c in list(x_order):
        if c not in df.columns:
            raise ValueError(f"Constructed predictor missing unexpectedly: {c}")
        if df[c].notna().sum() == 0:
            dropped_for_all_missing.append(c)

    if dropped_for_all_missing:
        # Only allow 'hispanic' to be all-missing; otherwise raise.
        not_allowed = [c for c in dropped_for_all_missing if c != "hispanic"]
        if not_allowed:
            raise ValueError(f"Predictor(s) are entirely missing: {not_allowed}")
        x_order = [c for c in x_order if c not in dropped_for_all_missing]

    paperA, fullA, fitA = fit_table2_model(
        df, "dislike_minority_genres", x_order, "Table2_ModelA_dislike_minority6", w_col=None
    )
    paperB, fullB, fitB = fit_table2_model(
        df, "dislike_other12_genres", x_order, "Table2_ModelB_dislike_other12", w_col=None
    )

    # Overview file
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Bryson (1996) Table 2 replication attempt using provided 1993 GSS extract.\n")
        f.write("DVs: dislike counts (4/5 on 1-5 scale) for minority-associated 6 vs other 12 genres.\n")
        f.write("Racism score: sum of 5 dichotomous items (0-5) per mapping instructions.\n")
        f.write("Standardized betas computed by fitting model on z-scored y and x (within analytic sample).\n")
        f.write("Constant reported unstandardized (from original-scale model).\n\n")
        if dropped_for_all_missing:
            f.write(f"NOTE: Dropped all-missing predictors: {dropped_for_all_missing}\n\n")
        f.write("Model A fit:\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B fit:\n")
        f.write(fitB.to_string(index=False))
        f.write("\n")

    return {
        "ModelA_table_paper_style": paperA,
        "ModelB_table_paper_style": paperB,
        "ModelA_table_full": fullA,
        "ModelB_table_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }