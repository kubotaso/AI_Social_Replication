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

    def star(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def gss_na_to_nan(s):
        """
        Conservative GSS missing recode:
        - treat common special codes as missing (>= 90) and common small-code missings (8,9)
        - keep legitimate categories like 0/1/2/etc.
        This works well for the provided extract where valid codes are small integers and
        missings often appear as 8/9/98/99.
        """
        s = to_num(s)
        s = s.where(~s.isin([8, 9, 98, 99]), np.nan)
        s = s.where(~(s >= 90), np.nan)  # catches 90-99 etc. (e.g., 97, 98, 99)
        return s

    def make_dummy_from_codes(s, one_codes, valid_codes=None):
        s = gss_na_to_nan(s)
        if valid_codes is not None:
            s = s.where(s.isin(valid_codes), np.nan)
        one_codes = set(one_codes if isinstance(one_codes, (list, tuple, set)) else [one_codes])
        return pd.Series(np.where(s.isna(), np.nan, s.isin(one_codes).astype(float)), index=s.index)

    def z_beta_from_unstd(res, X, y):
        """
        Compute standardized betas from an unstandardized OLS fit:
            beta_j = b_j * sd(x_j) / sd(y)
        Binary predictors are left as-is in X; this formula still yields standardized betas.
        Uses sample SD (ddof=1) on the estimation sample.
        """
        y_sd = np.nanstd(y, ddof=1)
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = np.nanstd(X[col], ddof=1)
            if not np.isfinite(x_sd) or x_sd == 0 or not np.isfinite(y_sd) or y_sd == 0:
                betas[col] = np.nan
            else:
                betas[col] = float(res.params[col]) * (x_sd / y_sd)
        return betas

    def format_table(table_df):
        # Create a compact "beta(stars)" string and keep constant separate
        out = table_df.copy()
        out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
        out["p"] = pd.to_numeric(out["p"], errors="coerce")
        out["sig"] = out["p"].apply(star)

        def fmt_beta(row):
            if row["term"] == "Constant":
                if pd.isna(row["b"]):
                    return ""
                return f"{row['b']:.3f}"
            if pd.isna(row["beta"]):
                return ""
            return f"{row['beta']:.3f}{row['sig']}"

        out["beta_star"] = out.apply(fmt_beta, axis=1)
        return out[["term", "beta_star"]]

    def fit_model(df, dv, xcols, model_name, label_map):
        needed = [dv] + xcols
        d = df[needed].copy()

        # Ensure numeric
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum().sort_values(ascending=False)

        # Model-specific listwise deletion
        d = d.dropna(axis=0, how="any").copy()

        # Drop predictors with zero variance in estimation sample (prevents singular fits / NaN output)
        kept = []
        dropped = []
        for c in xcols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            # Empty model frame
            tab = pd.DataFrame(
                [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan}]
                + [{"term": label_map.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan} for c in xcols]
            )
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            return tab, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        res = sm.OLS(y, Xc).fit()

        betas = z_beta_from_unstd(res, Xc, y)

        rows = []
        # Constant: unstandardized b, p-value for completeness (we won't star it in display)
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
        })

        # Predictors: standardized beta + p from unstandardized regression (t-tests)
        for c in xcols:
            lab = label_map.get(c, c)
            if c in kept:
                rows.append({
                    "term": lab,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": float(res.pvalues.get(c, np.nan)),
                })
            else:
                rows.append({"term": lab, "b": np.nan, "beta": np.nan, "p": np.nan})

        tab = pd.DataFrame(rows)
        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return tab, fit, nonmissing_before, dropped, d

    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    # -----------------------------
    # Read data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # -----------------------------
    # DV: # music genres disliked (0-18), listwise across 18 items
    # disliked = 4 or 5; valid = 1..5; DK/etc -> missing
    # -----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = df[music_items].apply(gss_na_to_nan)
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float)
    disliked = disliked.where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1, min_count=len(music_items))
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0-18)\n"
        "Coding: each genre disliked if response in {4,5}; DK/NA treated as missing.\n"
        "Listwise across all 18 genre items.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # -----------------------------
    # Predictors
    # -----------------------------
    # SES
    df["educ_yrs"] = gss_na_to_nan(df.get("educ", np.nan))
    df["prestg80"] = gss_na_to_nan(df.get("prestg80", np.nan))

    # Income per capita: REALINC / HOMPOP
    df["realinc"] = gss_na_to_nan(df.get("realinc", np.nan))
    df["hompop"] = gss_na_to_nan(df.get("hompop", np.nan))
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    # Female: SEX 1=male, 2=female
    df["female"] = make_dummy_from_codes(df.get("sex", np.nan), one_codes=[2], valid_codes=[1, 2])

    # Age: keep 89 as 89 (already numeric)
    df["age"] = gss_na_to_nan(df.get("age", np.nan))

    # Race: 1=white, 2=black, 3=other (in this extract)
    race = gss_na_to_nan(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = pd.Series(np.where(race.isna(), np.nan, (race == 2).astype(float)), index=df.index)

    # Hispanic: use ETHNIC if present in extract.
    # In many GSS extracts, ETHNIC is Hispanic self-report (often 1=not hispanic, 2=hispanic).
    eth = gss_na_to_nan(df.get("ethnic", np.nan))
    eth = eth.where(eth.isin([1, 2]), np.nan)
    df["hispanic"] = pd.Series(np.where(eth.isna(), np.nan, (eth == 2).astype(float)), index=df.index)

    # Other race: define as RACE==3 AND not Hispanic (paper treats Hispanic separately)
    # If either race/eth missing, leave missing to respect listwise deletion.
    df["otherrace"] = np.nan
    mask_known = race.notna() & df["hispanic"].notna()
    df.loc[mask_known, "otherrace"] = ((race == 3) & (df["hispanic"] == 0)).astype(float)

    # Religion
    relig = gss_na_to_nan(df.get("relig", np.nan))
    # RELIG in classic GSS: 1=Protestant, 2=Catholic, 3=Jewish, 4=None, 5=Other
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = pd.Series(np.where(relig.isna(), np.nan, (relig == 4).astype(float)), index=df.index)

    # Conservative Protestant: best-effort from RELIG==1 and DENOM categories.
    # In this extract DENOM seems numeric; treat common missings as NA, keep small codes.
    denom = gss_na_to_nan(df.get("denom", np.nan))
    # Allow a reasonable set of denom codes; if outside, mark missing
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)

    # Approximation: "conservative Protestant" often includes Baptist and some evangelical groups.
    # Here: treat DENOM==1 (Baptist) and DENOM==6 (Other Protestant) as conservative, among Protestants.
    df["cons_prot"] = np.nan
    known_rel_denom = relig.notna() & denom.notna()
    df.loc[known_rel_denom, "cons_prot"] = ((relig == 1) & denom.isin([1, 6])).astype(float)

    # South: REGION==3
    region = gss_na_to_nan(df.get("region", np.nan))
    # region codes in GSS are small ints; accept 1..9 here
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = pd.Series(np.where(region.isna(), np.nan, (region == 3).astype(float)), index=df.index)

    # Political intolerance scale (0-15): sum of 15 "intolerant" responses; listwise across 15
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
        df[v] = gss_na_to_nan(df[v])

    tol = df[list(tol_items.keys())].copy()
    intoler = pd.DataFrame(index=df.index)
    for v, bad in tol_items.items():
        s = tol[v]
        intoler[v] = pd.Series(np.where(s.isna(), np.nan, (s == bad).astype(float)), index=df.index)

    df["pol_intol"] = intoler.sum(axis=1, min_count=len(tol_items))
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan

    # -----------------------------
    # Missingness audit (key to avoiding tiny N)
    # -----------------------------
    audit_vars = (["num_genres_disliked"] +
                  ["educ_yrs", "inc_pc", "prestg80", "female", "age", "black", "hispanic", "otherrace",
                   "cons_prot", "norelig", "south", "pol_intol"])
    audit = []
    for v in audit_vars:
        if v in df.columns:
            s = df[v]
            audit.append({
                "var": v,
                "nonmissing": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "pct_missing": float(s.isna().mean() * 100.0)
            })
    audit_df = pd.DataFrame(audit).sort_values(["pct_missing", "var"], ascending=[False, True])
    write_text("./output/table1_missingness_audit.txt", audit_df.to_string(index=False) + "\n")

    # -----------------------------
    # Models (Table 1)
    # -----------------------------
    dv = "num_genres_disliked"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    label_map = {
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
        "pol_intol": "Political intolerance (0-15)",
    }

    tab1, fit1, before1, dropped1, frame1 = fit_model(df, dv, m1_x, "Model 1 (SES)", label_map)
    tab2, fit2, before2, dropped2, frame2 = fit_model(df, dv, m2_x, "Model 2 (Demographic)", label_map)
    tab3, fit3, before3, dropped3, frame3 = fit_model(df, dv, m3_x, "Model 3 (Political intolerance)", label_map)

    fit_stats = pd.DataFrame([fit1, fit2, fit3])

    def model_txt(model_name, fit, before, dropped, tab):
        # Display: standardized betas with stars; constant as unstandardized b (no stars)
        disp = format_table(tab)

        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append("Non-missing counts BEFORE listwise deletion (model variables):")
        lines.append(before.to_string())
        lines.append("")
        if dropped:
            lines.append("Dropped predictors due to zero variance AFTER listwise deletion:")
            lines.append(", ".join(dropped))
            lines.append("")
        lines.append("Fit statistics:")
        lines.append(f"N = {fit['n']}")
        lines.append(f"R^2 = {fit['r2']:.6f}" if np.isfinite(fit["r2"]) else "R^2 = NA")
        lines.append(f"Adj R^2 = {fit['adj_r2']:.6f}" if np.isfinite(fit["adj_r2"]) else "Adj R^2 = NA")
        lines.append("")
        lines.append("Coefficients shown to match Table 1 style:")
        lines.append("- Constant: unstandardized intercept (raw DV units), printed without stars")
        lines.append("- Predictors: standardized beta (β) with stars from two-tailed p-values")
        lines.append("- Stars: * p<.05, ** p<.01, *** p<.001")
        lines.append("")
        lines.append(disp.to_string(index=False))
        lines.append("")
        return "\n".join(lines)

    write_text("./output/table1_model1_ses.txt", model_txt("Model 1 (SES)", fit1, before1, dropped1, tab1))
    write_text("./output/table1_model2_demographic.txt", model_txt("Model 2 (Demographic)", fit2, before2, dropped2, tab2))
    write_text("./output/table1_model3_political_intolerance.txt", model_txt("Model 3 (Political intolerance)", fit3, before3, dropped3, tab3))

    # Combined summary
    combined = []
    combined.append("Table 1 replication summary (computed from provided data extract)")
    combined.append("===============================================================")
    combined.append("")
    combined.append("Fit statistics:")
    combined.append(fit_stats.to_string(index=False))
    combined.append("")
    combined.append("Notes:")
    combined.append("- If N is far below expected, inspect ./output/table1_missingness_audit.txt")
    combined.append("- Hispanic is derived from ETHNIC in this extract (1=not, 2=yes).")
    combined.append("- Other race is defined as RACE==3 and not Hispanic (to avoid overlap).")
    combined.append("- Conservative Protestant is approximated using RELIG==Protestant and DENOM in {1,6}.")
    combined.append("")
    write_text("./output/table1_summary.txt", "\n".join(combined))

    # Return results
    tables = {
        "fit_stats": fit_stats,
        "Model 1 (SES)": tab1,
        "Model 2 (Demographic)": tab2,
        "Model 3 (Political intolerance)": tab3,
        "missingness_audit": audit_df,
        "model_frames": {
            "Model 1 (SES)": frame1,
            "Model 2 (Demographic)": frame2,
            "Model 3 (Political intolerance)": frame3,
        },
    }
    return tables