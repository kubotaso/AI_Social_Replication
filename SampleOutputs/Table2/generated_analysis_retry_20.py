def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def _to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def _clean_gss_numeric(series):
        """
        Conservative cleaning of likely GSS missing codes across this extract.
        We only drop the most common sentinel-like codes; do NOT over-drop.
        """
        x = _to_num(series).copy()
        sentinel = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(list(sentinel)))
        return x

    def _likert_dislike_indicator(item_series):
        """
        Music taste: 1-5 scale. Dislike is 4/5.
        1/2/3 => 0; 4/5 => 1; other/NA-coded => missing.
        """
        x = _clean_gss_numeric(item_series)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def _binary_from_codes(series, true_codes, false_codes):
        x = _clean_gss_numeric(series)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def _zscore(s, ddof=0):
        s = _to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")
        return (s - mu) / sd

    def _sig_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def _build_dislike_count_allow_partial(df, items, min_nonmissing):
        """
        Build a dislike count allowing partial item nonresponse.

        - Each item contributes 0/1.
        - If fewer than min_nonmissing items are observed, DV is missing.
        - If at least min_nonmissing observed, DV is sum over observed items (missing treated as not counted).
        """
        mat = pd.concat([_likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        n_obs = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, skipna=True)
        count = count.where(n_obs >= min_nonmissing, np.nan)
        return count

    def _fit_standardized_ols(df, dv, xcols, model_name):
        """
        Standardized OLS coefficients (beta):
        - Fit OLS on unstandardized DV and X (with intercept), then compute standardized betas:
          beta_j = b_j * sd(x_j) / sd(y)
        - Intercept reported as unstandardized intercept.
        - Stars based on model p-values (replication-computed).
        - Listwise deletion ONLY on dv + xcols (as required).
        """
        needed = [dv] + xcols
        d = df[needed].replace([np.inf, -np.inf], np.nan).copy()
        d = d.dropna(axis=0, how="any")
        n = d.shape[0]
        if n < (len(xcols) + 5):
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={n}).")

        y = _to_num(d[dv])
        X = pd.DataFrame({c: _to_num(d[c]) for c in xcols}, index=d.index)
        Xc = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, Xc).fit()

        # Standardized betas computed from unstandardized b using sample SDs (ddof=0)
        y_sd = float(y.std(ddof=0))
        betas = {}
        for c in xcols:
            x_sd = float(X[c].std(ddof=0))
            b = float(model.params.get(c, np.nan))
            if np.isfinite(b) and np.isfinite(x_sd) and x_sd > 0 and np.isfinite(y_sd) and y_sd > 0:
                betas[c] = b * (x_sd / y_sd)
            else:
                betas[c] = np.nan

        # Output table in paper-like order
        rows = []
        for c in xcols:
            p = float(model.pvalues.get(c, np.nan))
            rows.append(
                {
                    "term": c,
                    "std_beta": betas[c],
                    "sig": _sig_stars(p),
                    "p_value_replication": p,
                }
            )

        # Constant (unstandardized; not a standardized beta)
        p_const = float(model.pvalues.get("const", np.nan))
        rows.append(
            {
                "term": "Constant",
                "std_beta": np.nan,
                "sig": _sig_stars(p_const),
                "p_value_replication": p_const,
                "b_unstd": float(model.params.get("const", np.nan)),
            }
        )

        out = pd.DataFrame(rows)

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k_predictors_excl_const": int(model.df_model),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )
        return model, out, fit

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter year == 1993
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")
    df["year"] = _to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Dependent variables (allow partial item missingness to avoid N collapse)
    # -------------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in (minority_items + other12_items):
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    # Minimum observed items required for DV construction (tuned to avoid excessive N loss)
    # Rationale: paper treats DK as missing and excludes missing cases; but requiring all items
    # collapses N in this extract. Use a reasonable threshold.
    df["dislike_minority_genres"] = _build_dislike_count_allow_partial(df, minority_items, min_nonmissing=5)
    df["dislike_other12_genres"] = _build_dislike_count_allow_partial(df, other12_items, min_nonmissing=10)

    # -------------------------
    # Racism score (0-5) additive, require all 5 items
    # -------------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = _binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = _binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = _binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = _binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = _binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -------------------------
    # RHS controls
    # -------------------------
    # Education
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = _clean_gss_numeric(df["educ"]).where(_clean_gss_numeric(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = _clean_gss_numeric(df["realinc"])
    hompop = _clean_gss_numeric(df["hompop"]).where(_clean_gss_numeric(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = _clean_gss_numeric(df["prestg80"])

    # Female
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = _binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = _clean_gss_numeric(df["age"]).where(_clean_gss_numeric(df["age"]).between(18, 89))

    # Race indicators (Black, Other race)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = _clean_gss_numeric(df["race"]).where(_clean_gss_numeric(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic indicator:
    # Not present in provided variables mapping; create as missing (cannot estimate).
    # We do NOT proxy using ETHNIC (per instruction).
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = _clean_gss_numeric(df["relig"])
    denom = _clean_gss_numeric(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index).astype(float)
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index).astype(float)
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = _clean_gss_numeric(df["region"]).where(_clean_gss_numeric(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index).astype(float)
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -------------------------
    # Models (Table 2)
    # NOTE: Hispanic cannot be included (all-missing). Keep specification faithful otherwise.
    # -------------------------
    xcols = [
        "racism_score",
        "education_years",
        "hh_income_per_capita",
        "occ_prestige",
        "female",
        "age_years",
        "black",
        # "hispanic",  # cannot estimate from provided extract
        "other_race",
        "cons_protestant",
        "no_religion",
        "south",
    ]

    # Fit
    modelA, tabA, fitA = _fit_standardized_ols(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6"
    )
    modelB, tabB, fitB = _fit_standardized_ols(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12"
    )

    # -------------------------
    # Save outputs
    # -------------------------
    def _save_model_outputs(model, tab, fit, prefix):
        # Statsmodels summary
        with open(f"./output/{prefix}_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        # Paper-style table: term + std_beta + sig (+ intercept b_unstd)
        tab2 = tab.copy()
        # Pretty formatting
        def _fmt_beta(x):
            if pd.isna(x):
                return ""
            return f"{x: .3f}".strip()

        def _fmt_const(row):
            if row["term"] != "Constant":
                return ""
            b = row.get("b_unstd", np.nan)
            return "" if not np.isfinite(b) else f"{b: .3f}".strip()

        tab2["std_beta_fmt"] = tab2["std_beta"].apply(_fmt_beta)
        tab2["const_b_unstd_fmt"] = tab2.apply(_fmt_const, axis=1)

        paper_like = tab2[["term", "std_beta_fmt", "sig", "const_b_unstd_fmt"]].copy()
        paper_like.columns = ["term", "std_beta", "sig", "constant_b_unstd"]

        with open(f"./output/{prefix}_table_paper_style.txt", "w", encoding="utf-8") as f:
            f.write("Standardized OLS coefficients (beta). Intercept shown as unstandardized constant.\n")
            f.write("Significance stars are computed from replication p-values (not reported in the paper table).\n")
            f.write(paper_like.to_string(index=False))
            f.write("\n\nFit:\n")
            f.write(fit.to_string(index=False))
            f.write("\n")

        # Full table with replication p-values
        full = tab.copy()
        if "b_unstd" not in full.columns:
            full["b_unstd"] = np.nan
        full_out = full[["term", "std_beta", "sig", "p_value_replication", "b_unstd"]].copy()
        full_out.to_csv(f"./output/{prefix}_table_full.csv", index=False)

        with open(f"./output/{prefix}_table_full.txt", "w", encoding="utf-8") as f:
            f.write(full_out.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
            f.write("\n")

    _save_model_outputs(modelA, tabA, fitA, "Table2_ModelA_dislike_minority6")
    _save_model_outputs(modelB, tabB, fitB, "Table2_ModelB_dislike_other12")

    # Overview / diagnostics
    # Frequencies for key constructed vars to diagnose issues
    diag_lines = []
    diag_lines.append("Diagnostics\n")
    diag_lines.append("Note: Hispanic dummy cannot be constructed from provided variables; it is omitted.\n")
    diag_lines.append("\nDV ranges (computed):\n")
    diag_lines.append(f"dislike_minority_genres: min={df['dislike_minority_genres'].min(skipna=True)}, max={df['dislike_minority_genres'].max(skipna=True)}\n")
    diag_lines.append(f"dislike_other12_genres: min={df['dislike_other12_genres'].min(skipna=True)}, max={df['dislike_other12_genres'].max(skipna=True)}\n")
    diag_lines.append("\nNo religion frequency (non-missing):\n")
    diag_lines.append(df["no_religion"].value_counts(dropna=True).to_string() + "\n")
    diag_lines.append("\nSouth frequency (non-missing):\n")
    diag_lines.append(df["south"].value_counts(dropna=True).to_string() + "\n")
    diag_lines.append("\nRacism score distribution (non-missing):\n")
    diag_lines.append(df["racism_score"].value_counts(dropna=True).sort_index().to_string() + "\n")

    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("Table 2 replication attempt (1993 GSS extract)\n")
        f.write("Model A DV: dislikes among 6 minority-associated genres\n")
        f.write(fitA.to_string(index=False))
        f.write("\n\nModel B DV: dislikes among other 12 genres\n")
        f.write(fitB.to_string(index=False))
        f.write("\n\n")
        f.write("Paper-style coefficient tables saved as *_table_paper_style.txt\n")
        f.write("\n")
        f.write("".join(diag_lines))

    return {
        "ModelA_table": tabA,
        "ModelB_table": tabB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }