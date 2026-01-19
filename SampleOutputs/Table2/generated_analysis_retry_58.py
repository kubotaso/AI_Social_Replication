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

    def is_gss_missing(x):
        """
        Conservative GSS missing detector for numeric-coded items.
        Treat common DK/Refused/NA high codes as missing.
        Do NOT treat legitimate category codes (e.g., 4,5 on 1-5 Likert) as missing.
        """
        x = to_num(x)
        miss = x.isna()
        # common DK/Refused/NA codes across many GSS items
        miss |= x.isin([8, 9, 98, 99, 998, 999, 9998, 9999])
        # very high sentinel bands (e.g., 1000+, 10000+)
        miss |= (x >= 1000)
        return miss

    def clean_numeric(x):
        x = to_num(x)
        x = x.mask(is_gss_missing(x), np.nan)
        return x

    def likert_dislike_indicator(x):
        """
        1-5 scale; dislike = 4 or 5; like/neutral = 1,2,3.
        Missing if not in 1..5 or if GSS-missing.
        """
        x = clean_numeric(x)
        x = x.where(x.between(1, 5), np.nan)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(x, true_codes, false_codes):
        x = clean_numeric(x)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def zscore(s, weights=None):
        s = to_num(s)
        if weights is None:
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True, ddof=0)
        else:
            w = to_num(weights)
            ok = s.notna() & w.notna() & np.isfinite(w) & (w > 0)
            if ok.sum() == 0:
                return s * np.nan
            sw = s.loc[ok].to_numpy(dtype=float)
            ww = w.loc[ok].to_numpy(dtype=float)
            mu = np.average(sw, weights=ww)
            var = np.average((sw - mu) ** 2, weights=ww)
            sd = float(np.sqrt(var))
        if not np.isfinite(sd) or sd == 0:
            return s * np.nan
        return (s - mu) / sd

    def build_count(df, items, require_complete=True):
        """
        Build count DV as sum of dislike indicators across items.
        require_complete=True: missing if any item missing (strict, as paper notes DK treated as missing and missing cases excluded).
        """
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in items], axis=1)
        if require_complete:
            return mat.sum(axis=1, min_count=len(items))
        else:
            # allow partial; not used by default
            return mat.sum(axis=1, min_count=1)

    def detect_hispanic_proxy(df):
        """
        No explicit Hispanic flag in provided variables. We must approximate to avoid dropping it.
        Use ETHNIC (ancestry/country-of-origin numeric codes) as a proxy:
        treat a set of common Hispanic-origin codes as Hispanic.
        This is an approximation; saved to output diagnostics.
        """
        if "ethnic" not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="float64"), "unavailable"

        e = clean_numeric(df["ethnic"])

        # Use plausible Hispanic-origin ancestry codes seen in many GSS extracts.
        # (kept as a small set; anything else -> 0)
        hisp_codes = {
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  # broad Spanish/Latin
            30, 31, 32, 33, 34, 35,                  # Mexico/Puerto Rico/Cuba/etc. in some schemes
            210, 211, 212, 213, 214, 215,            # sometimes 200-series used
            260, 261, 262, 263,                      # sometimes Caribbean/LatAm
            500, 501, 502, 503, 504, 505             # sometimes later schemes (kept; but 500+ may be non-missing codes)
        }

        # Guard: because we treat >=1000 as missing, 500+ remain possible valid codes.
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        out.loc[e.notna()] = 0.0
        out.loc[e.isin(list(hisp_codes))] = 1.0
        return out, "ETHNIC proxy (limited code list)"

    def fit_model_table2(df, dv, xcols, model_name, weight_col=None):
        """
        Fit OLS (or WLS if weights provided), compute standardized betas for slopes,
        and produce a "paper-style" table: beta + stars; intercept unstandardized.
        """
        needed = [dv] + xcols
        if weight_col is not None and weight_col in df.columns:
            needed = needed + [weight_col]

        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)

        # listwise deletion on model variables (and weight if used)
        d = d.dropna(axis=0, how="any").copy()
        if d.shape[0] < len(xcols) + 5:
            raise ValueError(f"{model_name}: too few complete cases after listwise deletion (n={d.shape[0]}).")

        # check zero variance in analytic sample
        zero_var = []
        for c in xcols:
            s = to_num(d[c])
            if s.nunique(dropna=True) <= 1:
                zero_var.append(c)
        if to_num(d[dv]).nunique(dropna=True) <= 1:
            raise ValueError(f"{model_name}: DV has zero variance in analytic sample.")
        if zero_var:
            raise ValueError(f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}.")

        y = to_num(d[dv]).astype(float)
        X = pd.DataFrame({c: to_num(d[c]).astype(float) for c in xcols}, index=d.index)
        Xc = sm.add_constant(X, has_constant="add")

        w = None
        if weight_col is not None and weight_col in d.columns:
            w = to_num(d[weight_col]).astype(float)
            w = w.where(np.isfinite(w) & (w > 0), np.nan)
            ok = w.notna()
            Xc = Xc.loc[ok]
            y = y.loc[ok]
            w = w.loc[ok]
            if y.shape[0] < len(xcols) + 5:
                raise ValueError(f"{model_name}: too few cases after dropping invalid weights (n={y.shape[0]}).")
            model = sm.WLS(y, Xc, weights=w).fit()
        else:
            model = sm.OLS(y, Xc).fit()

        # Standardized betas for slopes: b_j * sd(x_j)/sd(y)
        # Use ddof=0; if weights provided, compute weighted sd.
        if w is None:
            sd_y = y.std(ddof=0)
        else:
            yy = y.to_numpy()
            ww = w.to_numpy()
            mu_y = np.average(yy, weights=ww)
            var_y = np.average((yy - mu_y) ** 2, weights=ww)
            sd_y = float(np.sqrt(var_y))

        betas = {}
        for c in xcols:
            b = float(model.params[c])
            if w is None:
                sd_x = X[c].loc[model.model.data.row_labels].std(ddof=0)
            else:
                xx = X.loc[model.model.data.row_labels, c].to_numpy()
                ww = w.to_numpy()
                mu_x = np.average(xx, weights=ww)
                var_x = np.average((xx - mu_x) ** 2, weights=ww)
                sd_x = float(np.sqrt(var_x))
            if (not np.isfinite(sd_x)) or sd_x == 0 or (not np.isfinite(sd_y)) or sd_y == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (sd_x / sd_y)

        def star(p):
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        paper_rows = []
        for c in xcols:
            paper_rows.append(
                {
                    "term": c,
                    "beta": betas.get(c, np.nan),
                    "stars": star(float(model.pvalues[c])),
                    "p_value": float(model.pvalues[c]),
                }
            )
        paper = pd.DataFrame(paper_rows).set_index("term")

        full = pd.DataFrame(
            {
                "b": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        full.index.name = "term"

        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": int(model.nobs),
                    "k": int(model.df_model + 1),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                    "weighted": bool(w is not None),
                    "weight_col": (weight_col if (w is not None) else ""),
                }
            ]
        )

        return model, paper, full, fit, d.index

    # -----------------------
    # Load and filter
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    if "year" not in df.columns or "id" not in df.columns:
        raise ValueError("Dataset must include 'year' and 'id' columns.")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Dependent variables
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband", "blugrass", "country", "musicals", "classicl", "folk",
        "moodeasy", "newage", "opera", "conrock", "oldies", "hvymetal"
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_count(df, minority_items, require_complete=True)
    df["dislike_other12_genres"] = build_count(df, other12_items, require_complete=True)

    # -----------------------
    # Racism score 0-5 (complete on 5 items)
    # -----------------------
    racism_fields = ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]
    for c in racism_fields:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls
    # -----------------------
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_numeric(df["educ"]).where(clean_numeric(df["educ"]).between(0, 20), np.nan)
    df["education_years"] = educ

    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_numeric(df["realinc"])
    hompop = clean_numeric(df["hompop"]).where(clean_numeric(df["hompop"]) > 0, np.nan)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_numeric(df["prestg80"])

    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = clean_numeric(df["age"]).where(clean_numeric(df["age"]).between(18, 89), np.nan)

    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_numeric(df["race"]).where(clean_numeric(df["race"]).isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.notna(), (race == 2).astype(float), np.nan)
    df["other_race"] = np.where(race.notna(), (race == 3).astype(float), np.nan)

    # Hispanic (proxy) - required to avoid constant/NaN in model
    df["hispanic"], hisp_source = detect_hispanic_proxy(df)

    if "relig" not in df.columns:
        raise ValueError("Missing column: relig")
    relig = clean_numeric(df["relig"])

    # No religion: RELIG==4 (None)
    df["no_religion"] = np.where(relig.notna(), (relig == 4).astype(float), np.nan)

    # Conservative Protestant proxy: RELIG==1 (Protestant) and DENOM in {1,6,7}
    if "denom" not in df.columns:
        raise ValueError("Missing column: denom")
    denom = clean_numeric(df["denom"])
    consprot = np.where((relig == 1) & (denom.isin([1, 6, 7])), 1.0, 0.0)
    consprot = pd.Series(consprot, index=df.index, dtype="float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_numeric(df["region"]).where(clean_numeric(df["region"]).isin([1, 2, 3, 4]), np.nan)
    df["south"] = np.where(region.notna(), (region == 3).astype(float), np.nan)

    # -----------------------
    # Optional weights (if present)
    # -----------------------
    weight_col = None
    for cand in ["wtssall", "wtss", "wt", "weight"]:
        if cand in df.columns:
            weight_col = cand
            break

    # -----------------------
    # Fit models (Table 2 RHS)
    # -----------------------
    xcols = [
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

    # Diagnostics before fitting
    diag_lines = []
    diag_lines.append("Diagnostics (1993 only)\n")
    diag_lines.append(f"Rows in YEAR==1993: {len(df)}\n")
    diag_lines.append(f"Hispanic construction: {hisp_source}\n")
    diag_lines.append(f"Weight column used (if any): {weight_col}\n\n")

    def vc(name):
        s = df[name]
        return s.value_counts(dropna=False).to_string()

    for v in ["hispanic", "no_religion", "cons_protestant", "black", "other_race", "south"]:
        diag_lines.append(f"\nValue counts: {v}\n{vc(v)}\n")

    with open("./output/Table2_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("".join(diag_lines))

    # Fit both models
    mA, paperA, fullA, fitA, idxA = fit_model_table2(
        df, "dislike_minority_genres", xcols, "Table2_ModelA_dislike_minority6", weight_col=weight_col
    )
    mB, paperB, fullB, fitB, idxB = fit_model_table2(
        df, "dislike_other12_genres", xcols, "Table2_ModelB_dislike_other12", weight_col=weight_col
    )

    # -----------------------
    # Output formatting
    # -----------------------
    label_map = {
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
    order = [
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

    def paper_style_table(paper_df, model, model_name, dv_name):
        out = paper_df.loc[order].copy()
        out["label"] = [label_map[k] for k in out.index]
        out["beta_star"] = out["beta"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA") + out["stars"].fillna("")
        # Add constant (unstandardized) at bottom
        const = float(model.params["const"])
        const_p = float(model.pvalues["const"])
        const_star = "***" if const_p < 0.001 else ("**" if const_p < 0.01 else ("*" if const_p < 0.05 else ""))
        const_row = pd.DataFrame(
            {"beta": [np.nan], "stars": [""], "p_value": [const_p], "label": ["Constant"], "beta_star": [f"{const:.3f}{const_star}"]},
            index=["const"],
        )
        out2 = pd.concat([out[["label", "beta_star"]], const_row[["label", "beta_star"]]], axis=0)
        out2.index.name = "term"
        # Human-readable text
        lines = []
        lines.append(f"{model_name}\nDV: {dv_name}\n")
        lines.append(f"N={int(model.nobs)}, R2={model.rsquared:.3f}, AdjR2={model.rsquared_adj:.3f}\n\n")
        lines.append(out2.to_string())
        lines.append("\n")
        return out2, "\n".join(lines)

    paperA_table, paperA_text = paper_style_table(paperA, mA, "Table2_ModelA_dislike_minority6", "dislike_minority_genres")
    paperB_table, paperB_text = paper_style_table(paperB, mB, "Table2_ModelB_dislike_other12", "dislike_other12_genres")

    # Save summaries and tables
    with open("./output/Table2_ModelA_summary.txt", "w", encoding="utf-8") as f:
        f.write(mA.summary().as_text())
    with open("./output/Table2_ModelB_summary.txt", "w", encoding="utf-8") as f:
        f.write(mB.summary().as_text())

    with open("./output/Table2_ModelA_paper_style.txt", "w", encoding="utf-8") as f:
        f.write(paperA_text)
    with open("./output/Table2_ModelB_paper_style.txt", "w", encoding="utf-8") as f:
        f.write(paperB_text)

    paperA_table.to_csv("./output/Table2_ModelA_paper_style.csv")
    paperB_table.to_csv("./output/Table2_ModelB_paper_style.csv")

    fullA.to_csv("./output/Table2_ModelA_full_coefficients.csv")
    fullB.to_csv("./output/Table2_ModelB_full_coefficients.csv")
    fitA.to_csv("./output/Table2_ModelA_fit.csv", index=False)
    fitB.to_csv("./output/Table2_ModelB_fit.csv", index=False)

    overview = []
    overview.append("Table 2 replication attempt (computed from provided data)\n")
    overview.append("Models: OLS/WLS with standardized betas for slopes; unstandardized intercept.\n")
    overview.append("Dislike coding: 4/5 on 1-5 music items.\n")
    overview.append(f"Hispanic: {hisp_source}\n")
    overview.append(f"Weights used: {weight_col if weight_col is not None else 'None'}\n\n")
    overview.append("Model A fit:\n")
    overview.append(fitA.to_string(index=False))
    overview.append("\n\nModel B fit:\n")
    overview.append(fitB.to_string(index=False))
    overview.append("\n")
    with open("./output/Table2_overview.txt", "w", encoding="utf-8") as f:
        f.write("".join(overview))

    return {
        "ModelA_paper_style": paperA_table,
        "ModelB_paper_style": paperB_table,
        "ModelA_full": fullA,
        "ModelB_full": fullB,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }