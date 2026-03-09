def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    # Broad, conservative set of common GSS missing codes across numeric/categorical fields
    MISSING_CODES = {
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        6, 7, 8, 9,
        66, 67, 68, 69,
        76, 77, 78, 79,
        96, 97, 98, 99,
        996, 997, 998, 999,
        9996, 9997, 9998, 9999,
        99996, 99997, 99998, 99999,
    }

    def mask_missing(s):
        s = to_num(s)
        return s.mask(s.isin(MISSING_CODES), np.nan)

    def keep_valid(s, valid_values):
        s = mask_missing(s)
        valid_values = set(valid_values)
        return s.where(s.isin(valid_values), np.nan)

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

    def pop_sd(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.nan
        return float(np.std(x, ddof=0))

    def compute_std_betas_from_raw_fit(res_raw, df_model, y, xvars):
        """
        Standardized betas computed as beta = b * SD(x) / SD(y), using
        population SDs computed on the *same listwise estimation sample*.
        (This preserves a nonzero raw intercept, like the paper's table format.)
        """
        sd_y = pop_sd(df_model[y].values)
        betas = {}
        for v in xvars:
            sd_x = pop_sd(df_model[v].values)
            b = float(res_raw.params.get(v, np.nan))
            if not np.isfinite(sd_x) or not np.isfinite(sd_y) or sd_x == 0 or sd_y == 0:
                betas[v] = np.nan
            else:
                betas[v] = b * (sd_x / sd_y)
        return betas

    # Political intolerance coding per mapping (incl. COLCOM special case)
    def intolerance_indicator(col, s):
        s = mask_missing(s)
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if col.startswith("spk"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 2).astype(float)  # not allowed
        elif col.startswith("lib"):
            m = s.isin([1, 2])
            out.loc[m] = (s.loc[m] == 1).astype(float)  # remove
        elif col.startswith("col"):
            if col == "colcom":
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 4).astype(float)  # fired
            else:
                m = s.isin([4, 5])
                out.loc[m] = (s.loc[m] == 5).astype(float)  # not allowed
        return out

    def build_polintol_complete(df, items):
        intoler = pd.DataFrame({c: intolerance_indicator(c, df[c]) for c in items})
        answered = intoler.notna().sum(axis=1)
        pol = pd.Series(np.nan, index=df.index, dtype=float)
        ok = answered == len(items)
        pol.loc[ok] = intoler.loc[ok].sum(axis=1).astype(float)
        return pol, answered, intoler

    def fit_model(df_in, y, xvars, model_name):
        cols = [y] + xvars
        use = df_in.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna(how="any").copy()

        if use.shape[0] == 0:
            coef = pd.DataFrame(
                {"model": model_name, "term": xvars, "beta_std": np.nan, "p_raw": np.nan, "sig": ""}
            )
            fit = pd.DataFrame(
                [{
                    "model": model_name, "N": 0, "R2": np.nan, "Adj_R2": np.nan,
                    "const_raw": np.nan, "dropped_no_variance": ""
                }]
            )
            return None, coef, fit, use

        # Drop predictors with no variance in this estimation sample
        x_keep, dropped = [], []
        for v in xvars:
            if use[v].nunique(dropna=True) <= 1:
                dropped.append(v)
            else:
                x_keep.append(v)

        X = sm.add_constant(use[x_keep].astype(float), has_constant="add")
        yy = use[y].astype(float)
        res = sm.OLS(yy, X).fit()

        betas = compute_std_betas_from_raw_fit(res, use, y, x_keep)

        coef = pd.DataFrame({"model": model_name, "term": xvars})
        coef["included"] = coef["term"].isin(x_keep)
        coef["beta_std"] = coef["term"].map(lambda t: float(betas.get(t, np.nan)))
        coef["p_raw"] = coef["term"].map(lambda t: float(res.pvalues.get(t, np.nan)) if t in x_keep else np.nan)
        coef["sig"] = coef["p_raw"].map(stars)

        fit = pd.DataFrame(
            [{
                "model": model_name,
                "N": int(use.shape[0]),
                "R2": float(res.rsquared),
                "Adj_R2": float(res.rsquared_adj),
                "const_raw": float(res.params.get("const", np.nan)),
                "dropped_no_variance": ", ".join(dropped) if dropped else "",
            }]
        )
        return res, coef, fit, use

    def build_table1(coef_list, fit_list, model_order, row_order, label_map, dv_label):
        coef_long = pd.concat(coef_list, ignore_index=True)
        fit_long = pd.concat(fit_list, ignore_index=True).set_index("model").reindex(model_order)

        tbl = pd.DataFrame(index=row_order, columns=model_order, dtype=object)
        for m in model_order:
            cm = coef_long.loc[coef_long["model"] == m].set_index("term")
            for t in row_order:
                if t not in cm.index:
                    tbl.loc[t, m] = "—"
                else:
                    r = cm.loc[t]
                    if (not bool(r.get("included", True))) or pd.isna(r["beta_std"]):
                        tbl.loc[t, m] = "—"
                    else:
                        tbl.loc[t, m] = f"{float(r['beta_std']):.3f}{str(r['sig'])}"

        tbl.index = [label_map.get(t, t) for t in tbl.index]

        extra = pd.DataFrame(index=["Constant (raw)", "R²", "Adj. R²", "N"], columns=model_order, dtype=object)
        for m in model_order:
            if m not in fit_long.index:
                extra.loc[:, m] = ""
                continue
            extra.loc["Constant (raw)", m] = "" if pd.isna(fit_long.loc[m, "const_raw"]) else f"{float(fit_long.loc[m, 'const_raw']):.3f}"
            extra.loc["R²", m] = "" if pd.isna(fit_long.loc[m, "R2"]) else f"{float(fit_long.loc[m, 'R2']):.3f}"
            extra.loc["Adj. R²", m] = "" if pd.isna(fit_long.loc[m, "Adj_R2"]) else f"{float(fit_long.loc[m, 'Adj_R2']):.3f}"
            extra.loc["N", m] = "" if pd.isna(fit_long.loc[m, "N"]) else str(int(fit_long.loc[m, "N"]))

        header = pd.DataFrame({m: [f"DV: {dv_label}"] for m in model_order}, index=[""])
        out = pd.concat([header, tbl, extra], axis=0)
        return out, coef_long, fit_long.reset_index()

    def missingness_table(df_in, cols):
        rows = []
        for c in cols:
            rows.append({"var": c, "missing": int(df_in[c].isna().sum()), "nonmissing": int(df_in[c].notna().sum())})
        return pd.DataFrame(rows).sort_values(["missing", "var"], ascending=[False, True])

    # -----------------------------
    # Load and filter year
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower().strip() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Required column missing: year")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------------
    # Variable lists (per mapping)
    # -----------------------------
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

    required_cols = [
        "id", "educ", "realinc", "hompop", "prestg80",
        "sex", "age", "race", "ethnic", "relig", "denom", "region", "ballot"
    ] + music_items + tol_items
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -----------------------------
    # Clean/recode base columns
    # -----------------------------
    # Music items: keep 1..5; otherwise missing
    for c in music_items:
        df[c] = keep_valid(df[c], {1, 2, 3, 4, 5})

    # SES
    df["educ"] = mask_missing(df["educ"])
    df["realinc"] = mask_missing(df["realinc"])
    df["hompop"] = mask_missing(df["hompop"])
    df["prestg80"] = mask_missing(df["prestg80"])

    # Demographics
    df["sex"] = keep_valid(df["sex"], {1, 2})
    df["age"] = mask_missing(df["age"])
    df["race"] = keep_valid(df["race"], {1, 2, 3})
    df["region"] = keep_valid(df["region"], {1, 2, 3, 4})
    df["relig"] = keep_valid(df["relig"], {1, 2, 3, 4, 5})
    df["denom"] = mask_missing(df["denom"])
    df["ethnic"] = mask_missing(df["ethnic"])
    df["ballot"] = mask_missing(df["ballot"])

    # Tolerance items: keep substantive codes
    for c in tol_items:
        s = mask_missing(df[c])
        if c.startswith("spk") or c.startswith("lib"):
            s = s.where(s.isin([1, 2]), np.nan)
        elif c.startswith("col"):
            s = s.where(s.isin([4, 5]), np.nan)
        df[c] = s

    df = df.replace([np.inf, -np.inf], np.nan)

    # -----------------------------
    # DV: musical exclusiveness (strict complete-case on all 18 music items)
    # -----------------------------
    d = df.dropna(subset=music_items).copy()
    dislike_cols = []
    for c in music_items:
        dc = f"dislike_{c}"
        d[dc] = d[c].isin([4, 5]).astype(int)
        dislike_cols.append(dc)
    d["num_genres_disliked"] = d[dislike_cols].sum(axis=1).astype(float)

    # -----------------------------
    # IVs
    # -----------------------------
    # Income per capita: REALINC / HOMPOP (require hompop > 0)
    d["income_pc"] = np.nan
    m_inc = d["realinc"].notna() & d["hompop"].notna() & (d["hompop"] > 0)
    d.loc[m_inc, "income_pc"] = (d.loc[m_inc, "realinc"] / d.loc[m_inc, "hompop"]).astype(float)
    d["income_pc"] = d["income_pc"].replace([np.inf, -np.inf], np.nan)

    # Female
    d["female"] = np.where(d["sex"].notna(), (d["sex"] == 2).astype(float), np.nan)

    # Race / ethnicity dummies:
    # Key fix for N-collapse: treat ETHNIC missing as non-Hispanic (0) rather than NA,
    # so it doesn't force listwise deletion.
    # We only set Hispanic=1 when ETHNIC==1 is observed; else 0.
    d["hispanic"] = np.where(d["ethnic"].eq(1), 1.0, 0.0)

    # Black and other race from RACE; keep NA if RACE missing (cannot infer)
    d["black"] = np.where(d["race"].notna(), (d["race"] == 2).astype(float), np.nan)
    d["other_race"] = np.where(d["race"].notna(), (d["race"] == 3).astype(float), np.nan)

    # Enforce mutual exclusivity: if Hispanic==1 then race dummies set to 0 (if they were defined)
    m_h = d["hispanic"].eq(1.0)
    d.loc[m_h & d["black"].notna(), "black"] = 0.0
    d.loc[m_h & d["other_race"].notna(), "other_race"] = 0.0

    # No religion
    d["no_religion"] = np.where(d["relig"].notna(), (d["relig"] == 4).astype(float), np.nan)

    # Conservative Protestant proxy (coarse, per mapping): RELIG==1 and DENOM in {1,6,7}
    d["conservative_protestant"] = np.where(d["relig"].notna(), 0.0, np.nan)
    m_prot = d["relig"].eq(1) & d["relig"].notna() & d["denom"].notna()
    d.loc[m_prot, "conservative_protestant"] = d.loc[m_prot, "denom"].isin([1, 6, 7]).astype(float)

    # Southern
    d["southern"] = np.where(d["region"].notna(), (d["region"] == 3).astype(float), np.nan)

    # Political intolerance: strict complete-case across 15 items
    d["political_intolerance"], tol_answered, intoler_df = build_polintol_complete(d, tol_items)

    d = d.replace([np.inf, -np.inf], np.nan)

    # -----------------------------
    # Models
    # -----------------------------
    y = "num_genres_disliked"
    x_m1 = ["educ", "income_pc", "prestg80"]
    x_m2 = x_m1 + [
        "female", "age", "black", "hispanic", "other_race",
        "conservative_protestant", "no_religion", "southern"
    ]
    x_m3 = x_m2 + ["political_intolerance"]

    model_names = ["SES Model", "Demographic Model", "Political Intolerance Model"]

    res1, coef1, fit1, use1 = fit_model(d, y, x_m1, model_names[0])
    res2, coef2, fit2, use2 = fit_model(d, y, x_m2, model_names[1])
    res3, coef3, fit3, use3 = fit_model(d, y, x_m3, model_names[2])

    # -----------------------------
    # Table 1-style formatting (betas only; NO SE rows)
    # -----------------------------
    dv_label = "Number of music genres disliked"
    label_map = {
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

    table1, coef_long, fit_long = build_table1(
        coef_list=[coef1, coef2, coef3],
        fit_list=[fit1, fit2, fit3],
        model_order=model_names,
        row_order=row_order,
        label_map=label_map,
        dv_label=dv_label,
    )

    # -----------------------------
    # Diagnostics
    # -----------------------------
    diag = pd.DataFrame([{
        "N_year_1993": int(df.shape[0]),
        "N_complete_music_18": int(d.shape[0]),
        "N_model1_listwise": int(fit1.loc[0, "N"]),
        "N_model2_listwise": int(fit2.loc[0, "N"]),
        "N_model3_listwise": int(fit3.loc[0, "N"]),
        "hispanic_1_count_in_music_complete": int((d["hispanic"] == 1).sum()),
        "race_missing_in_music_complete": int(d["race"].isna().sum()),
        "political_intolerance_nonmissing_in_music_complete": int(d["political_intolerance"].notna().sum()),
        "political_intolerance_items_answered_mean": float(tol_answered.mean()) if len(tol_answered) else np.nan,
        "political_intolerance_items_answered_min": float(tol_answered.min()) if len(tol_answered) else np.nan,
        "political_intolerance_items_answered_max": float(tol_answered.max()) if len(tol_answered) else np.nan,
        "notes": (
            "Table shows standardized betas only (beta=b*SDx/SDy on model sample); "
            "no SE rows. Stars from raw OLS p-values."
        ),
    }])

    miss_m1 = missingness_table(d, [y] + x_m1)
    miss_m2 = missingness_table(d, [y] + x_m2)
    miss_m3 = missingness_table(d, [y] + x_m3)

    # -----------------------------
    # Save outputs
    # -----------------------------
    lines = []
    lines.append("Replication output: Table 1-style OLS (1993 GSS)")
    lines.append("")
    lines.append(f"DV: {dv_label}")
    lines.append("Displayed: standardized coefficients (beta) only; standard errors are not shown.")
    lines.append("Standardization: beta = b * SD(x) / SD(y), SDs computed on each model's listwise estimation sample.")
    lines.append("Stars: two-tailed p-values from raw (unstandardized) OLS on the same estimation sample: * p<.05, ** p<.01, *** p<.001")
    lines.append("")
    lines.append("Key construction rules:")
    lines.append("- DV: strict complete-case across all 18 music items; dislike={4,5}.")
    lines.append("- Income per capita: REALINC/HOMPOP with HOMPOP>0.")
    lines.append("- Hispanic: ETHNIC==1 -> 1 else 0 (ETHNIC missing treated as 0 to avoid N collapse); mutually exclusive with Black/Other race.")
    lines.append("- Political intolerance: strict complete-case across 15 items; sum of intolerant responses; COLCOM fired==4 is intolerant.")
    lines.append("- Conservative Protestant: RELIG==1 and DENOM in {1,6,7} (coarse proxy given available DENOM coding).")
    lines.append("")
    lines.append("Table 1-style standardized betas:")
    lines.append(table1.to_string())
    lines.append("")
    lines.append("Fit statistics:")
    lines.append(fit_long.to_string(index=False))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(diag.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 1 vars):")
    lines.append(miss_m1.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 2 vars):")
    lines.append(miss_m2.to_string(index=False))
    lines.append("")
    lines.append("Missingness within DV-complete sample (Model 3 vars):")
    lines.append(miss_m3.to_string(index=False))
    lines.append("")
    lines.append("Raw OLS summaries (debug):")
    if res1 is not None:
        lines.append("\n==== SES Model (raw) ====\n" + res1.summary().as_text())
    else:
        lines.append("\n==== SES Model (raw) ==== EMPTY SAMPLE\n")
    if res2 is not None:
        lines.append("\n==== Demographic Model (raw) ====\n" + res2.summary().as_text())
    else:
        lines.append("\n==== Demographic Model (raw) ==== EMPTY SAMPLE\n")
    if res3 is not None:
        lines.append("\n==== Political Intolerance Model (raw) ====\n" + res3.summary().as_text())
    else:
        lines.append("\n==== Political Intolerance Model (raw) ==== EMPTY SAMPLE\n")

    summary_text = "\n".join(lines)

    with open("./output/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("./output/regression_table_table1_style.txt", "w", encoding="utf-8") as f:
        f.write("Table 1-style output\n")
        f.write(f"DV: {dv_label}\n")
        f.write("Cells: standardized betas with stars from raw OLS p-values.\n")
        f.write("Standard errors not shown.\n")
        f.write("Stars: * p<.05, ** p<.01, *** p<.001\n\n")
        f.write(table1.to_string())
        f.write("\n")

    # Machine-readable outputs
    table1.to_csv("./output/regression_table_table1_style.tsv", sep="\t", index=True)
    coef_long.to_csv("./output/regression_coefficients_long.tsv", sep="\t", index=False)
    fit_long.to_csv("./output/model_fit_stats.tsv", sep="\t", index=False)
    diag.to_csv("./output/diagnostics_overall.tsv", sep="\t", index=False)
    miss_m1.to_csv("./output/missingness_m1.tsv", sep="\t", index=False)
    miss_m2.to_csv("./output/missingness_m2.tsv", sep="\t", index=False)
    miss_m3.to_csv("./output/missingness_m3.tsv", sep="\t", index=False)

    return {
        "table1_style": table1,
        "fit_stats": fit_long,
        "coefficients_long": coef_long,
        "diagnostics_overall": diag,
        "missingness_m1": miss_m1,
        "missingness_m2": miss_m2,
        "missingness_m3": miss_m3,
        "estimation_samples": {
            model_names[0]: use1,
            model_names[1]: use2,
            model_names[2]: use3,
        },
    }