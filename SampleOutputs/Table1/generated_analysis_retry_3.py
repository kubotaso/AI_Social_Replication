def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def star(p):
        try:
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
        except Exception:
            return ""
        return ""

    def zscore(s):
        s = s.astype(float)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def standardize_frame(dsub, cols):
        out = dsub.copy()
        for c in cols:
            out[c] = zscore(out[c])
        return out

    def dv_disliked_count(df, items):
        music = df[items].apply(to_num)
        music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

        # disliked: 1 if 4 or 5, 0 if 1/2/3, nan otherwise
        disliked = music.apply(lambda s: np.where(s.isna(), np.nan, np.where(s.isin([4, 5]), 1.0, 0.0)))
        disliked = pd.DataFrame(disliked, columns=items, index=df.index)

        # listwise across all 18 items: any missing -> DV missing
        dv = disliked.sum(axis=1, min_count=len(items))
        dv.loc[disliked.isna().any(axis=1)] = np.nan
        return dv

    def build_pol_intol(df):
        # Item coding per mapping instruction
        tol_items = {
            # Anti-religionist
            "spkath": 2,
            "colath": 5,
            "libath": 1,
            # Racist
            "spkrac": 2,
            "colrac": 5,
            "librac": 1,
            # Communist
            "spkcom": 2,
            "colcom": 4,
            "libcom": 1,
            # Military-rule advocate
            "spkmil": 2,
            "colmil": 5,
            "libmil": 1,
            # Homosexual
            "spkhomo": 2,
            "colhomo": 5,
            "libhomo": 1,
        }

        for v in tol_items:
            if v not in df.columns:
                df[v] = np.nan
            df[v] = to_num(df[v])

        ind = pd.DataFrame(index=df.index)
        for v, bad_code in tol_items.items():
            s = df[v]
            ind[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

        # listwise across 15 items for the scale: if any missing -> missing
        pol = ind.sum(axis=1, min_count=len(tol_items))
        pol.loc[ind.isna().any(axis=1)] = np.nan
        return pol

    def fit_standardized_ols(dsub, y_col, x_cols, model_name):
        """
        Fit OLS on standardized y and standardized X.
        - Predictors' coefficients are standardized betas.
        - Constant is unstandardized in the paper; but when y is standardized, constant is mean-adjusted.
          We'll report constant from unstandardized-y model (raw y, standardized X) for readability.
        """
        # Build model frame: listwise deletion for this model
        needed = [y_col] + x_cols
        m = dsub[needed].copy()
        for c in needed:
            m[c] = to_num(m[c])

        m = m.dropna(axis=0, how="any")
        n = int(len(m))

        # Standardize y and X for beta estimates
        m_std = m.copy()
        m_std[y_col] = zscore(m_std[y_col])
        m_std = standardize_frame(m_std, x_cols)

        y_std = m_std[y_col].astype(float)
        X_std = sm.add_constant(m_std[x_cols].astype(float), has_constant="add")
        res_std = sm.OLS(y_std, X_std).fit()

        # Also fit raw-y with standardized X to get a comparable constant in DV units
        y_raw = m[y_col].astype(float)
        X_std2 = sm.add_constant(m_std[x_cols].astype(float), has_constant="add")
        res_rawy = sm.OLS(y_raw, X_std2).fit()

        # Assemble table: report standardized betas for predictors, constant from raw-y model
        rows = []

        # Constant (DV units; X standardized)
        rows.append(
            {
                "term": "Constant",
                "beta": np.nan,
                "b_const_rawy": float(res_rawy.params.get("const", np.nan)),
                "p_value": float(res_rawy.pvalues.get("const", np.nan)),
                "sig": star(res_rawy.pvalues.get("const", np.nan)),
            }
        )

        # Predictors (standardized betas)
        for c in x_cols:
            rows.append(
                {
                    "term": c,
                    "beta": float(res_std.params.get(c, np.nan)),
                    "b_const_rawy": np.nan,
                    "p_value": float(res_std.pvalues.get(c, np.nan)),
                    "sig": star(res_std.pvalues.get(c, np.nan)),
                }
            )

        tab = pd.DataFrame(rows)

        fit = {
            "model": model_name,
            "n": n,
            "r2": float(res_std.rsquared),
            "adj_r2": float(res_std.rsquared_adj),
        }

        return res_std, res_rawy, tab, fit, m

    def write_text_outputs(model_name, fit, tab, diagnostics, path_txt):
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("=" * len(model_name) + "\n\n")
            f.write("Fit (from standardized-y and standardized-X OLS):\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.4f}\n")
            f.write(f"Adj. R^2 = {fit['adj_r2']:.4f}\n\n")

            f.write("Coefficients reported in Table-1 style:\n")
            f.write("- Predictors: standardized OLS coefficients (beta)\n")
            f.write("- Constant: from OLS with standardized predictors but raw DV (b_const_rawy)\n")
            f.write("- p-values shown only to support significance stars (paper reports stars, not SEs)\n\n")

            show = tab.copy()
            show["beta"] = show["beta"].astype(float).round(3)
            show["b_const_rawy"] = show["b_const_rawy"].astype(float).round(3)
            show["p_value"] = show["p_value"].astype(float).round(4)
            f.write(show.to_string(index=False))
            f.write("\n\n")

            f.write("Diagnostics (non-missing counts and dummy variation in the model frame):\n")
            f.write(diagnostics.to_string())
            f.write("\n")

    def diagnostics_for_model_frame(m, x_cols):
        diag = []
        for c in x_cols:
            s = m[c]
            diag.append(
                {
                    "var": c,
                    "n_nonmissing": int(s.notna().sum()),
                    "n_unique": int(s.nunique(dropna=True)),
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                }
            )
        return pd.DataFrame(diag)

    # ----------------------------
    # Read data & restrict year
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" in df.columns:
        df["year"] = to_num(df["year"])
        df = df.loc[df["year"] == 1993].copy()

    # ----------------------------
    # DV: musical exclusiveness (0-18 count), listwise across 18 genre items
    # ----------------------------
    music_items = [
        "bigband",
        "blugrass",
        "country",
        "blues",
        "musicals",
        "classicl",
        "folk",
        "gospel",
        "jazz",
        "latin",
        "moodeasy",
        "newage",
        "opera",
        "rap",
        "reggae",
        "conrock",
        "oldies",
        "hvymetal",
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items for DV: {missing_music}")

    df["music_exclusive"] = dv_disliked_count(df, music_items)

    # Save DV descriptives
    dv_desc = df["music_exclusive"].describe()
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Musical exclusiveness (# genres disliked)\n")
        f.write("Count of 18 genres rated 4/5 (dislike/dislike very much).\n")
        f.write("Listwise across the 18 music items: any missing -> DV missing.\n\n")
        f.write(dv_desc.to_string())
        f.write("\n")

    # ----------------------------
    # IVs: SES
    # ----------------------------
    df["educ"] = to_num(df.get("educ", np.nan))
    df["prestg80"] = to_num(df.get("prestg80", np.nan))

    df["realinc"] = to_num(df.get("realinc", np.nan))
    df["hompop"] = to_num(df.get("hompop", np.nan))

    # per-capita income = REALINC / HOMPOP
    df["inc_pc"] = np.where(
        df["realinc"].notna() & df["hompop"].notna() & (df["hompop"] > 0),
        df["realinc"] / df["hompop"],
        np.nan,
    )

    # ----------------------------
    # Demographics / group identity
    # ----------------------------
    df["sex"] = to_num(df.get("sex", np.nan))
    df["female"] = np.where(df["sex"] == 2, 1.0, np.where(df["sex"] == 1, 0.0, np.nan))

    df["age"] = to_num(df.get("age", np.nan))

    # Race dummies (white reference implied)
    df["race"] = to_num(df.get("race", np.nan))
    valid_race = df["race"].isin([1, 2, 3])
    df["black"] = np.where(valid_race, (df["race"] == 2).astype(float), np.nan)
    df["otherrace"] = np.where(valid_race, (df["race"] == 3).astype(float), np.nan)

    # Hispanic: use ETHNIC as provided in the available variables
    # Keep only codes 1/2 to avoid collapsing N; treat everything else missing.
    df["ethnic"] = to_num(df.get("ethnic", np.nan))
    df["hispanic"] = np.where(df["ethnic"].isin([1, 2]), (df["ethnic"] == 2).astype(float), np.nan)

    # Religion: no religion and conservative Protestant
    df["relig"] = to_num(df.get("relig", np.nan))
    df["denom"] = to_num(df.get("denom", np.nan))

    valid_relig = df["relig"].isin([1, 2, 3, 4])  # prot/cath/jew/none
    df["norelig"] = np.where(valid_relig, (df["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant (best-effort with DENOM; keep non-Protestants as 0, but only when known)
    # Avoid previous length-mismatch bug by assigning via aligned Series.
    valid_relig_denom = df["relig"].notna() & df["denom"].notna()
    is_prot = df["relig"] == 1

    # DENOM coding differs by file; use a conservative, non-empty mapping:
    # mark as conservative if Protestant and DENOM indicates Baptist or "other Protestant".
    denom_cons = df["denom"].isin([1, 6])
    cons_series = (is_prot & denom_cons).astype(float)

    df["cons_prot"] = np.nan
    df.loc[valid_relig_denom, "cons_prot"] = cons_series.loc[valid_relig_denom]

    # Region: south dummy
    df["region"] = to_num(df.get("region", np.nan))
    valid_region = df["region"].isin([1, 2, 3, 4])
    df["south"] = np.where(valid_region, (df["region"] == 3).astype(float), np.nan)

    # ----------------------------
    # Political intolerance scale (0-15), listwise across 15 items
    # ----------------------------
    df["pol_intol"] = build_pol_intol(df)

    # ----------------------------
    # Model specs
    # ----------------------------
    dv = "music_exclusive"
    model_specs = {
        "Model 1 (SES)": ["educ", "inc_pc", "prestg80"],
        "Model 2 (Demographic)": [
            "educ",
            "inc_pc",
            "prestg80",
            "female",
            "age",
            "black",
            "hispanic",
            "otherrace",
            "cons_prot",
            "norelig",
            "south",
        ],
        "Model 3 (Political intolerance)": [
            "educ",
            "inc_pc",
            "prestg80",
            "female",
            "age",
            "black",
            "hispanic",
            "otherrace",
            "cons_prot",
            "norelig",
            "south",
            "pol_intol",
        ],
    }

    all_fit = []
    tables = {}
    model_ns = {}

    # Save variable missingness overview (before listwise deletion)
    overview_vars = [dv] + sorted({v for cols in model_specs.values() for v in cols})
    miss = []
    for c in overview_vars:
        s = df[c] if c in df.columns else pd.Series(index=df.index, dtype=float)
        miss.append(
            {
                "var": c,
                "n_total": int(len(df)),
                "n_nonmissing": int(pd.Series(s).notna().sum()),
                "pct_missing": float(pd.Series(s).isna().mean() * 100.0),
                "n_unique_nonmissing": int(pd.Series(s).nunique(dropna=True)),
            }
        )
    miss_df = pd.DataFrame(miss).sort_values(["pct_missing", "var"], ascending=[False, True])
    miss_df.to_csv("./output/table1_missingness_overview.csv", index=False)

    with open("./output/table1_missingness_overview.txt", "w", encoding="utf-8") as f:
        f.write("Missingness overview (before model-wise listwise deletion)\n")
        f.write(miss_df.to_string(index=False))
        f.write("\n")

    # Fit models
    for model_name, x_cols in model_specs.items():
        res_std, res_rawy, tab, fit, mframe = fit_standardized_ols(df, dv, x_cols, model_name)
        diag = diagnostics_for_model_frame(mframe, x_cols)

        # Save text output
        safe = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        write_text_outputs(
            model_name,
            fit,
            tab,
            diag,
            f"./output/table1_{safe}.txt",
        )

        # Save compact CSV table
        tab_out = tab.copy()
        tab_out.to_csv(f"./output/table1_{safe}_betas.csv", index=False)

        all_fit.append(fit)
        tables[model_name] = tab
        model_ns[model_name] = fit["n"]

    fit_df = pd.DataFrame(all_fit)
    fit_df.to_csv("./output/table1_fit_stats.csv", index=False)

    with open("./output/table1_fit_stats.txt", "w", encoding="utf-8") as f:
        f.write("Fit statistics (standardized OLS models)\n")
        f.write(fit_df.to_string(index=False))
        f.write("\n")

    # Return a dict of results for programmatic checks
    return {
        "fit_stats": fit_df,
        "tables": tables,
        "missingness_overview": miss_df,
        "model_ns": pd.Series(model_ns),
    }