def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def star(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def zscore(s):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def recode_gss_missing(series):
        """
        Conservative missing recode:
        - negatives => missing (GSS commonly uses negative codes for NA)
        - keep non-negative values as-is
        """
        s = to_num(series)
        s = s.where(~(s < 0), np.nan)
        return s

    def valid_range(series, lo=None, hi=None, allowed=None):
        s = series.copy()
        if allowed is not None:
            s = s.where(s.isin(allowed), np.nan)
        if lo is not None:
            s = s.where(s >= lo, np.nan)
        if hi is not None:
            s = s.where(s <= hi, np.nan)
        return s

    def make_dummy(series, one, valid=None):
        s = series.copy()
        if valid is not None:
            s = s.where(s.isin(valid), np.nan)
        return pd.Series(np.where(s.isna(), np.nan, (s == one).astype(float)), index=s.index)

    def standardized_betas_from_unstd(unstd_res, df_model, dv, x_cols):
        """
        Compute standardized betas from an unstandardized OLS fit:
            beta_j = b_j * sd(x_j) / sd(y)
        """
        y = df_model[dv]
        y_sd = y.std(ddof=0)
        betas = {}
        for c in x_cols:
            if c not in unstd_res.params.index:
                betas[c] = np.nan
                continue
            x_sd = df_model[c].std(ddof=0)
            b = unstd_res.params[c]
            if pd.isna(y_sd) or y_sd == 0 or pd.isna(x_sd) or x_sd == 0 or pd.isna(b):
                betas[c] = np.nan
            else:
                betas[c] = float(b) * float(x_sd) / float(y_sd)
        return betas

    def fit_table1_model(df, dv, x_cols, model_name):
        # model frame
        d = df[[dv] + x_cols].copy()
        nonmissing_before = d.notna().sum()

        # listwise deletion per model
        d = d.dropna(axis=0, how="any").copy()

        # drop constant predictors after deletion (avoid singular matrix and NaN estimates)
        kept = []
        dropped = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            coef = pd.DataFrame([{"term": "Constant", "b": np.nan, "beta": np.nan, "sig": ""}]
                                + [{"term": c, "b": np.nan, "beta": np.nan, "sig": ""} for c in x_cols])
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            return coef, fit, nonmissing_before, dropped, d

        # unstandardized OLS for intercept, fit stats, and p-values for stars
        y = d[dv].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        # standardized betas (Table 1 uses standardized coefficients for predictors; intercept unstandardized)
        betas = standardized_betas_from_unstd(res, d, dv, kept)

        rows = []
        # Constant (unstandardized)
        p_const = res.pvalues.get("const", np.nan)
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "sig": star(p_const)
        })

        # Predictors: standardized beta + stars from unstandardized model p-values (pragmatic replication)
        for c in x_cols:
            if c in kept:
                p = res.pvalues.get(c, np.nan)
                rows.append({
                    "term": c,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "sig": star(p),
                })
            else:
                rows.append({"term": c, "b": np.nan, "beta": np.nan, "sig": ""})

        coef = pd.DataFrame(rows)
        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return coef, fit, nonmissing_before, dropped, d

    def format_table_for_txt(coef_df, label_map=None, decimals=3):
        tab = coef_df.copy()
        if label_map is not None:
            tab["term"] = tab["term"].map(lambda t: label_map.get(t, t))

        # create display column with beta + stars; constant displays b
        tab["display"] = ""
        is_const = tab["term"].str.lower().eq("constant")

        # constant: unstandardized b
        b = pd.to_numeric(tab["b"], errors="coerce").round(decimals)
        beta = pd.to_numeric(tab["beta"], errors="coerce").round(decimals)

        tab.loc[is_const, "display"] = b[is_const].map(lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}") + tab.loc[is_const, "sig"].fillna("")
        tab.loc[~is_const, "display"] = beta[~is_const].map(lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}") + tab.loc[~is_const, "sig"].fillna("")

        # keep only what Table 1 shows: term + display
        out = tab[["term", "display"]].copy()
        return out

    def write_model_txt(path, model_name, fit, nonmissing_before, dropped_predictors, table_txt, notes=""):
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append("Non-missing counts BEFORE listwise deletion (model variables):")
        lines.append(nonmissing_before.to_string())
        lines.append("")
        if dropped_predictors:
            lines.append("Dropped predictors (no variance after listwise deletion):")
            lines.append(", ".join(dropped_predictors))
            lines.append("")
        lines.append("Fit statistics:")
        lines.append(f"N = {fit['n']}")
        lines.append(f"R^2 = {fit['r2']:.6f}" if pd.notna(fit["r2"]) else "R^2 = NA")
        lines.append(f"Adj R^2 = {fit['adj_r2']:.6f}" if pd.notna(fit["adj_r2"]) else "Adj R^2 = NA")
        lines.append("")
        lines.append("Table 1-style output:")
        lines.append("- Predictors shown as standardized coefficients (β) with stars")
        lines.append("- Constant shown as unstandardized intercept (b) with stars")
        lines.append("- Stars from two-tailed p-values: * <.05, ** <.01, *** <.001")
        lines.append("")
        lines.append(table_txt.to_string(index=False))
        if notes:
            lines.append("")
            lines.append("Notes:")
            lines.append(notes)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0-18)
    # disliked = 4 or 5, valid responses 1..5, else missing
    # listwise across all 18 items (if any missing => DV missing)
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = df[music_items].apply(recode_gss_missing)
    music = music.apply(lambda s: valid_range(s, allowed=[1, 2, 3, 4, 5]))
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Number of music genres disliked (0-18)\n")
        f.write("Count of 18 genre ratings coded 4 ('dislike') or 5 ('dislike very much').\n")
        f.write("Listwise across 18 items; nonpositive codes treated as missing.\n\n")
        f.write(dv_desc.to_string())
        f.write("\n")

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = valid_range(recode_gss_missing(df.get("educ", np.nan)), lo=0, hi=30)
    df["prestg80"] = valid_range(recode_gss_missing(df.get("prestg80", np.nan)), lo=0, hi=100)

    realinc = recode_gss_missing(df.get("realinc", np.nan))
    hompop = recode_gss_missing(df.get("hompop", np.nan))
    hompop = hompop.where(hompop > 0, np.nan)
    inc_pc = realinc / hompop
    inc_pc = inc_pc.replace([np.inf, -np.inf], np.nan)
    # avoid absurd negatives (already handled) but keep 0+; cap extreme outliers only if present
    df["inc_pc"] = inc_pc

    # Demographics / identity
    sex = valid_range(recode_gss_missing(df.get("sex", np.nan)), allowed=[1, 2])
    df["female"] = make_dummy(sex, one=2, valid=[1, 2])

    df["age"] = valid_range(recode_gss_missing(df.get("age", np.nan)), lo=18, hi=89)

    race = valid_range(recode_gss_missing(df.get("race", np.nan)), allowed=[1, 2, 3])
    df["black"] = make_dummy(race, one=2, valid=[1, 2, 3])
    df["otherrace"] = make_dummy(race, one=3, valid=[1, 2, 3])

    # Hispanic: use 'ethnic' (available variable list). Common GSS convention: 1=not Hispanic, 2=Hispanic.
    # Keep only 1/2 as valid; others missing.
    if "ethnic" in df.columns:
        ethnic = valid_range(recode_gss_missing(df["ethnic"]), allowed=[1, 2])
        df["hispanic"] = make_dummy(ethnic, one=2, valid=[1, 2])
    else:
        df["hispanic"] = np.nan

    # Religion: norelig from RELIG == 4 (none). Keep valid 1..13 as given by mapping.
    relig = recode_gss_missing(df.get("relig", np.nan))
    relig = valid_range(relig, lo=1, hi=13)
    df["norelig"] = make_dummy(relig, one=4, valid=list(range(1, 14)))

    # Conservative Protestant: Protestant (RELIG==1) and denomination in a conservative set.
    # Use denom codes present; treat denom missing as missing for this dummy (to avoid falsely classifying).
    denom = recode_gss_missing(df.get("denom", np.nan))
    # allow 0..14 as plausible; keep broad to avoid wiping out sample
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)

    is_prot = relig == 1
    # pragmatic conservative set: Baptist(1) and "other"(6) among Protestants (matches prior approach; keep as-is)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = pd.Series(np.where(is_prot & denom.notna(), denom_cons.astype(float), np.where(relig.notna() & denom.notna(), 0.0, np.nan)), index=df.index)

    region = recode_gss_missing(df.get("region", np.nan))
    region = valid_range(region, lo=1, hi=9)
    df["south"] = make_dummy(region, one=3, valid=list(range(1, 10)))

    # Political intolerance (0-15): sum intolerant responses; listwise across 15 items
    tol_items = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    for v in tol_items:
        if v not in df.columns:
            df[v] = np.nan
        df[v] = recode_gss_missing(df[v])

    intoler = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        s = df[v]
        # keep as missing if missing; otherwise 1 if intolerant code else 0
        intoler[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    df["pol_intol"] = intoler.sum(axis=1)
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan
    df["pol_intol"] = valid_range(df["pol_intol"], lo=0, hi=15)

    # ----------------------------
    # Models
    # ----------------------------
    dv = "num_genres_disliked"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    label_map = {
        "Constant": "Constant",
        "educ_yrs": "Education (years)",
        "inc_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "otherrace": "Other race",
        "cons_prot": "Conservative Protestant",
        "norelig": "No religion",
        "south": "Southern",
        "pol_intol": "Political intolerance",
    }

    coef1, fit1, before1, dropped1, d1 = fit_table1_model(df, dv, m1_x, "Model 1 (SES)")
    coef2, fit2, before2, dropped2, d2 = fit_table1_model(df, dv, m2_x, "Model 2 (Demographic)")
    coef3, fit3, before3, dropped3, d3 = fit_table1_model(df, dv, m3_x, "Model 3 (Political intolerance)")

    tab1 = format_table_for_txt(coef1, label_map=label_map, decimals=3)
    tab2 = format_table_for_txt(coef2, label_map=label_map, decimals=3)
    tab3 = format_table_for_txt(coef3, label_map=label_map, decimals=3)

    # Write model text files
    write_model_txt("./output/table1_model1_ses.txt", "Model 1 (SES)", fit1, before1, dropped1, tab1)
    write_model_txt("./output/table1_model2_demographic.txt", "Model 2 (Demographic)", fit2, before2, dropped2, tab2)
    write_model_txt("./output/table1_model3_political_intolerance.txt", "Model 3 (Political intolerance)", fit3, before3, dropped3, tab3)

    # Write a combined summary
    fit_df = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2"]]
    with open("./output/table1_fit_stats.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 fit statistics\n")
        f.write("======================\n\n")
        f.write(fit_df.to_string(index=False))
        f.write("\n")

    # Quick diagnostics to help ensure N isn't collapsing
    diag_vars = [dv] + sorted(set(m3_x))
    diag = pd.DataFrame({
        "nonmissing": df[diag_vars].notna().sum(),
        "missing": df[diag_vars].isna().sum(),
        "nunique": df[diag_vars].nunique(dropna=True),
    }).sort_index()
    with open("./output/table1_missingness_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("Missingness diagnostics (post-recode)\n")
        f.write("====================================\n\n")
        f.write(diag.to_string())
        f.write("\n")

    # Return key results
    return {
        "fit_stats": fit_df,
        "tables": {
            "Model 1 (SES)": tab1,
            "Model 2 (Demographic)": tab2,
            "Model 3 (Political intolerance)": tab3,
        },
        "raw_coef_tables": {
            "Model 1 (SES)": coef1,
            "Model 2 (Demographic)": coef2,
            "Model 3 (Political intolerance)": coef3,
        },
        "model_frames": {
            "Model 1 (SES)": d1,
            "Model 2 (Demographic)": d2,
            "Model 3 (Political intolerance)": d3,
        },
        "dv_describe": dv_desc,
        "missingness_diagnostics": diag,
    }