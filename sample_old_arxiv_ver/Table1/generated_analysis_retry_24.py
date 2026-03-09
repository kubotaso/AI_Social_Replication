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

    # GSS-style special missing codes (do NOT treat 0 as missing globally)
    MISSING_CODES = {8, 9, 98, 99, 998, 999, 9998, 9999}

    def mask_missing(series):
        s = to_num(series)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_codes(series, valid_codes):
        s = mask_missing(series)
        return s.where(s.isin(list(valid_codes)), np.nan)

    def pop_sd(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return np.nan
        return float(a.std(ddof=0))

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    # -------------------------
    # Political intolerance coding (15 items)
    # -------------------------
    def intolerance_indicator(col, s):
        """
        Returns intolerant (1/0) with NaN for missing/invalid.
        """
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            if col == "colcom":
                # communist teacher: 4=yes fired, 5=not fired
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)
            else:
                # other COL*: 4=allowed, 5=not allowed
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)
        return out

    def build_polintol_scale(df, tol_items, min_answered=12):
        """
        Paper describes a COUNT across 15 dichotomous items. In practice, GSS batteries
        often have some item nonresponse; to avoid collapsing N excessively, we:
          - compute sum over non-missing items
          - require at least min_answered answered items
        This retains cases while keeping the scale interpretable as a "count" of intolerance.
        """
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in tol_items})
        answered = intoler.notna().sum(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        ok = answered >= int(min_answered)
        pol.loc[ok] = intoler.loc[ok].sum(axis=1, min_count=1).astype(float)
        return pol, intoler, answered

    # -------------------------
    # OLS + standardized betas
    # -------------------------
    def fit_ols_standardized_betas(data, y, xvars, model_name):
        """
        Runs OLS on raw variables (complete cases).
        Reports standardized betas computed on the estimation sample:
            beta_j = b_j * SD(x_j) / SD(y)
        """
        cols = [y] + xvars
        dd = data.loc[:, cols].dropna(how="any").copy()
        n = int(dd.shape[0])

        # Avoid singularities: drop predictors with no variance in this estimation sample
        x_keep, dropped = [], []
        for v in xvars:
            if dd[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        if n == 0 or len(x_keep) == 0:
            # Return empty shells
            coef_rows = []
            for v in xvars:
                coef_rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": False,
                        "beta_std": np.nan,
                        "b_raw": np.nan,
                        "p_raw": np.nan,
                        "cell": "—",
                    }
                )
            coef_long = pd.DataFrame(coef_rows)
            fit = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "N": n,
                        "R2": np.nan,
                        "Adj_R2": np.nan,
                        "const_raw": np.nan,
                        "dropped_no_variance": ", ".join(dropped) if dropped else "",
                    }
                ]
            )
            return None, coef_long, fit, dd

        X = sm.add_constant(dd[x_keep].astype(float), has_constant="add")
        yy = dd[y].astype(float)
        res = sm.OLS(yy, X).fit()

        sd_y = pop_sd(yy.values)

        betas = {}
        for v in x_keep:
            sd_x = pop_sd(dd[v].values)
            b = float(res.params.get(v, np.nan))
            if np.isfinite(sd_x) and np.isfinite(sd_y) and sd_x > 0 and sd_y > 0:
                betas[v] = b * (sd_x / sd_y)
            else:
                betas[v] = np.nan

        rows = []
        for v in xvars:
            if v in x_keep:
                beta = betas.get(v, np.nan)
                p = float(res.pvalues.get(v, np.nan))
                cell = "—" if not np.isfinite(beta) else f"{beta:.3f}{stars(p)}"
                rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": True,
                        "beta_std": beta,
                        "b_raw": float(res.params.get(v, np.nan)),
                        "p_raw": p,
                        "cell": cell,
                    }
                )
            else:
                rows.append(
                    {
                        "model": model_name,
                        "term": v,
                        "included": False,
                        "beta_std": np.nan,
                        "b_raw": np.nan,
                        "p_raw": np.nan,
                        "cell": "—",
                    }
                )

        coef_long = pd.DataFrame(rows)
        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "N": n,
                    "R2": float(res.rsquared),
                    "Adj_R2": float(res.rsquared_adj),
                    "const_raw": float(res.params.get("const", np.nan)),
                    "dropped_no_variance": ", ".join(dropped) if dropped else "",
                }
            ]
        )
        return res, coef_long, fit, dd

    def build_table(coef_long, fitstats, row_order, model_order, label_map):
        wide = coef_long.pivot(index="term", columns="model", values="cell")
        wide = wide.reindex(index=row_order, columns=model_order).fillna("—")
        wide.index = [label_map.get(t, t) for t in wide.index]

        fit = fitstats.set_index("model").reindex(model_order)
        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit.loc[m, "const_raw"]) else f"{fit.loc[m, 'const_raw']:.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit.loc[m, "R2"]) else f"{fit.loc[m, 'R2']:.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit.loc[m, "Adj_R2"]) else f"{fit.loc[m, 'Adj_R2']:.3f}"
            extra.loc["N", m] = "" if pd.isna(fit.loc[m, "N"]) else str(int(fit.loc[m, "N"]))
        return pd.concat([wide, extra], axis=0)

    def missingness_table(df, vars_):
        out = []
        for v in vars_:
            out.append({"var": v, "missing": int(df[v].isna().sum()), "nonmissing": int(df[v].notna().sum())})
        return pd.DataFrame(out).sort_values(["missing", "var"], ascending=[False, True])

    # -------------------------
    # Load
    # -------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower().strip() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Required column missing: year")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -------------------------
    # Variable lists
    # -------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]

    tol_items = [
        "spkath", "colath", "libath",
        "spkrac", "colrac", "librac",
        "spkcom", "colcom", "libcom",
        "spkmil", "colmil", "libmil",
        "spkhomo", "colhomo", "libhomo"
    ]

    required = (
        ["id", "educ", "realinc", "hompop", "prestg80", "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"]
        + music_items
        + tol_items
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Clean fields
    # -------------------------
    # Music items: only 1..5 are substantive
    for c in music_items:
        df[c] = keep_codes(df[c], {1, 2, 3, 4, 5})

    # Core predictors
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    df["sex"] = keep_codes(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])

    df["race"] = keep_codes(df["race"], {1, 2, 3})
    df["region"] = keep_codes(df["region"], {1, 2, 3, 4})
    df["relig"] = keep_codes(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items validity
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    # -------------------------
    # DV: musical exclusiveness = count disliked across 18; require complete on all 18 items
    # -------------------------
    d = df.dropna(subset=music_items).copy()
    for c in music_items:
        d[f"dislike_{c}"] = d[c].isin([4, 5]).astype(int)
    dislike_cols = [f"dislike_{c}" for c in music_items]
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -------------------------
    # IVs
    # -------------------------
    # Income per capita = REALINC / HOMPOP (require HOMPOP>0)
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race dummies (white reference)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Hispanic: use ETHNIC if present in extract (do not fabricate constant)
    # In many GSS extracts, ETHNIC==1 corresponds to Hispanic; we use that rule if ETHNIC is coded.
    d["hispanic"] = np.where(d["ethnic"].notna(), (d["ethnic"] == 1).astype(float), np.nan)

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy using provided DENOM recode (coarse)
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna()
    d.loc[m_prot, "conservative_protestant"] = 0.0
    m_prot_denom = m_prot & d["denom"].notna()
    d.loc[m_prot_denom, "conservative_protestant"] = d.loc[m_prot_denom, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: partial completion allowed (min answered)
    d["political_intolerance"], intoler_df, tol_answered = build_polintol_scale(d, tol_items, min_answered=12)

    # -------------------------
    # Models
    # -------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]

    m1_res, c1, f1, s1 = fit_ols_standardized_betas(d, y, x_m1, model_names[0])
    m2_res, c2, f2, s2 = fit_ols_standardized_betas(d, y, x_m2, model_names[1])
    m3_res, c3, f3, s3 = fit_ols_standardized_betas(d, y, x_m3, model_names[2])

    coef_long = pd.concat([c1, c2, c3], ignore_index=True)
    fitstats = pd.concat([f1, f2, f3], ignore_index=True)

    # -------------------------
    # Build Table 1-style output (betas only; no SEs)
    # -------------------------
    pretty = {
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "conservative_protestant": "Conservative Protestant",
        "no_religion": "No religion",
        "southern": "Southern",
        "political_intolerance": "Political intolerance",
    }
    row_order = [
        "educ", "income_pc", "prestg80",
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern",
        "political_intolerance",
    ]

    table1 = build_table(coef_long, fitstats, row_order, model_names, pretty)

    # -------------------------
    # Diagnostics
    # -------------------------
    diag = pd.DataFrame(
        [
            {
                "N_year_1993": int(df.shape[0]),
                "N_complete_music_18": int(d.shape[0]),
                "N_M1_completecases": int(d[[y] + x_m1].dropna().shape[0]),
                "N_M2_completecases": int(d[[y] + x_m2].dropna().shape[0]),
                "N_M3_completecases": int(d[[y] + x_m3].dropna().shape[0]),
                "hispanic_nonmissing": int(d["hispanic"].notna().sum()),
                "hispanic_1_count": int((d["hispanic"] == 1).sum(skipna=True)),
                "political_intolerance_nonmissing": int(d["political_intolerance"].notna().sum()),
                "political_intolerance_min_answered": 12,
                "tol_items_answered_mean": float(tol_answered.mean()) if len(tol_answered) else np.nan,
                "tol_items_answered_min": float(tol_answered.min()) if len(tol_answered) else np.nan,
                "tol_items_answered_max": float(tol_answered.max()) if len(tol_answered) else np.nan,
            }
        ]
    )

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -------------------------
    # Save outputs (human-readable)
    # -------------------------
    lines = []
    lines.append("Replication: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append("DV (all models): Number of music genres disliked (0–18).")
    lines.append("DV construction: disliked=4/5 across 18 music items; respondents must have complete responses on all 18 items.")
    lines.append("")
    lines.append("Coefficients shown: standardized betas computed as beta = b * SD(x)/SD(y) on each model's estimation sample.")
    lines.append("Stars: from raw OLS p-values (* p<.05, ** p<.01, *** p<.001).")
    lines.append("")
    lines.append("Table 1-style standardized coefficients (betas only):")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Model fit stats:")
    lines.append(fitstats.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 1 variables:")
    lines.append(miss_m1.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 2 variables:")
    lines.append(miss_m2.to_string(index=False))
    lines.append("")
    lines.append("Missingness (within DV-complete sample) — Model 3 variables:")
    lines.append(miss_m3.to_string(index=False))
    lines.append("")
    lines.append("Raw OLS summaries (debug; not part of Table 1 formatting):")
    if m1_res is not None:
        lines.append("\n==== Model 1 (SES) ====\n" + m1_res.summary().as_text())
    if m2_res is not None:
        lines.append("\n==== Model 2 (Demographic) ====\n" + m2_res.summary().as_text())
    if m3_res is not None:
        lines.append("\n==== Model 3 (Political intolerance) ====\n" + m3_res.summary().as_text())

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write("DV: Number of music genres disliked (0–18)\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("Note: Table displays betas only (no standard errors).\n\n")
        f.write(table1.to_string())
        f.write("\n")

    table1.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fitstats.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)
    miss_m1.to_csv("./output/missingness_m1.tsv", sep="\t", index=False)
    miss_m2.to_csv("./output/missingness_m2.tsv", sep="\t", index=False)
    miss_m3.to_csv("./output/missingness_m3.tsv", sep="\t", index=False)

    return {
        "table1_style": table1,
        "fit_stats": fitstats,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "missingness_m1": miss_m1,
        "missingness_m2": miss_m2,
        "missingness_m3": miss_m3,
        "estimation_samples": {"m1": s1, "m2": s2, "m3": s3},
    }