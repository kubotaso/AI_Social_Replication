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
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def coerce_gss_missing(s):
        """
        Conservative missing recode used across many GSS numeric items.
        Do NOT treat small positive integers (like valid categorical codes) as missing.
        """
        s = to_num(s)
        # Common GSS missing/value labels often appear as negative codes.
        s = s.mask(s.isin([-1, -2, -3, -4, -5, -6, -7, -8, -9]))
        return s

    def standardize_for_beta(s, ddof=0):
        s = to_num(s)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=ddof)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index), mu, sd
        return (s - mu) / sd, mu, sd

    def standardized_betas_from_unstandardized(y, X, b_unstd):
        """
        Compute standardized betas as: beta_j = b_j * sd(X_j)/sd(Y)
        using the estimation sample (already listwise-deleted).
        """
        y_sd = float(np.std(y, ddof=0))
        betas = {}
        for col in X.columns:
            if col == "const":
                continue
            x_sd = float(np.std(X[col], ddof=0))
            if y_sd == 0 or x_sd == 0 or not np.isfinite(y_sd) or not np.isfinite(x_sd):
                betas[col] = np.nan
            else:
                betas[col] = float(b_unstd[col]) * (x_sd / y_sd)
        return betas

    def format_table1_like(model_name, res, X_cols, betas_std, labels_map):
        """
        Return DataFrame with: term, beta, sig
        - Constant is UNSTANDARDIZED intercept (raw units).
        - Predictors are standardized betas (beta).
        - Stars based on p-values from unstandardized OLS.
        """
        rows = []
        # constant first
        const_name = "const"
        const_val = float(res.params[const_name]) if const_name in res.params.index else np.nan
        const_p = float(res.pvalues[const_name]) if const_name in res.pvalues.index else np.nan
        rows.append(
            {
                "term": "Constant",
                "beta": const_val,
                "sig": "",  # Table 1 typically doesn't star the constant; keep blank
                "_p": const_p,
            }
        )

        for c in X_cols:
            lbl = labels_map.get(c, c)
            b = betas_std.get(c, np.nan)
            p = float(res.pvalues[c]) if c in res.pvalues.index else np.nan
            rows.append({"term": lbl, "beta": b, "sig": star(p), "_p": p})

        tab = pd.DataFrame(rows)
        # display formatting (no p-values)
        tab["beta"] = pd.to_numeric(tab["beta"], errors="coerce")
        tab["beta_fmt"] = tab["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        tab["beta_star"] = tab["beta_fmt"] + tab["sig"].astype(str)
        # show constant as raw with 3 decimals
        tab.loc[tab["term"] == "Constant", "beta_star"] = tab.loc[tab["term"] == "Constant", "beta"].map(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"
        )
        return tab[["term", "beta_star"]].rename(columns={"beta_star": model_name})

    def fit_model(df, dv, x_cols, model_name, labels_map):
        needed = [dv] + x_cols
        d = df[needed].copy()
        for c in needed:
            d[c] = coerce_gss_missing(d[c])

        nonmissing_before = d.notna().sum()

        # Listwise deletion PER MODEL
        d = d.dropna(axis=0, how="any").copy()

        # Drop predictors with no variance (rare, but guard)
        kept = []
        dropped = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            tab = pd.DataFrame({"term": ["Constant"] + [labels_map.get(c, c) for c in x_cols], model_name: [""] * (1 + len(x_cols))})
            return tab, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas_std = standardized_betas_from_unstandardized(y.to_numpy(), Xc.to_numpy()[:, 1:], res.params)  # not used directly
        # Recompute properly using DataFrames (more robust)
        betas_std = standardized_betas_from_unstandardized(y.to_numpy(), Xc, res.params)

        # Build table (include all requested x_cols; blanks for dropped)
        betas_all = {c: (betas_std[c] if c in betas_std else np.nan) for c in x_cols}
        tab = format_table1_like(model_name, res, x_cols, betas_all, labels_map)

        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return tab, fit, nonmissing_before, dropped, d

    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[coerce_gss_missing(df["year"]) == 1993].copy()

    # -----------------------------
    # DV: number of music genres disliked (0-18)
    # Disliked = 4 or 5; valid = 1..5; other/missing -> NA; listwise across 18
    # -----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music item columns: {missing_music}")

    music = df[music_items].apply(coerce_gss_missing)
    # keep only 1..5 as valid responses
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    # listwise across 18 music items
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # Save DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0-18)\n"
        "Constructed as count of 18 genre ratings coded 4 ('dislike') or 5 ('dislike very much').\n"
        "Any missing/non-1..5 response on any of the 18 items -> DV set to missing (listwise across items).\n\n"
        f"{dv_desc.to_string()}\n",
    )

    # -----------------------------
    # Predictors
    # -----------------------------
    # SES
    df["educ_yrs"] = coerce_gss_missing(df.get("educ", np.nan))
    df["prestg80"] = coerce_gss_missing(df.get("prestg80", np.nan))

    df["realinc"] = coerce_gss_missing(df.get("realinc", np.nan))
    df["hompop"] = coerce_gss_missing(df.get("hompop", np.nan))
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = coerce_gss_missing(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age"] = coerce_gss_missing(df.get("age", np.nan))

    race = coerce_gss_missing(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if present; in this extract it appears numeric; treat 1/2 as valid
    # (1=not hispanic, 2=hispanic) per common GSS coding; keep conservative validity to avoid collapsing N.
    if "ethnic" in df.columns:
        eth = coerce_gss_missing(df["ethnic"])
        eth = eth.where(eth.isin([1, 2]), np.nan)
        df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
    else:
        df["hispanic"] = np.nan

    # Religion
    relig = coerce_gss_missing(df.get("relig", np.nan))
    # RELIG typically 1..5; keep any positive integer as potentially valid but require finite and >=1
    relig = relig.where((relig >= 1) & (relig <= 13), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))  # 4=none in GSS

    # Conservative Protestant: derived from RELIG==1 and DENOM categories.
    # Use a simple, documentation-supported operationalization that does not create massive missingness:
    # define when RELIG and DENOM observed; otherwise missing.
    denom = coerce_gss_missing(df.get("denom", np.nan))
    # denom commonly 0..13 with 0=NA/none; keep 0..13; let 0 be a valid code but treat as not conservative
    denom = denom.where((denom >= 0) & (denom <= 14), np.nan)
    known_rel_denom = relig.notna() & denom.notna()
    is_prot = relig == 1
    denom_cons = denom.isin([1, 6])  # Baptist or Other Protestant (approximation; avoids over-missing)
    df["cons_prot"] = np.where(known_rel_denom, (is_prot & denom_cons).astype(float), np.nan)

    # South
    region = coerce_gss_missing(df.get("region", np.nan))
    region = region.where((region >= 1) & (region <= 9), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum of 15 intolerant responses; listwise across items
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
        df[v] = coerce_gss_missing(df[v])

    tol = df[list(tol_items.keys())].copy()
    intoler = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        s = tol[v]
        # Keep only finite values; do not attempt to impose a universal valid range (varies by item).
        intoler[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    df["pol_intol"] = intoler.sum(axis=1)
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan

    # -----------------------------
    # Missingness audit (pre-model)
    # -----------------------------
    audit_vars = [
        "num_genres_disliked",
        "educ_yrs", "inc_pc", "prestg80",
        "female", "age", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol",
    ]
    audit = pd.DataFrame(
        {
            "var": audit_vars,
            "nonmissing": [int(df[v].notna().sum()) if v in df.columns else 0 for v in audit_vars],
            "missing": [int(df[v].isna().sum()) if v in df.columns else len(df) for v in audit_vars],
        }
    )
    audit["pct_missing"] = (audit["missing"] / max(len(df), 1) * 100.0).round(1)
    write_text("./output/table1_missingness_audit.txt", audit.to_string(index=False) + "\n")

    # -----------------------------
    # Models
    # -----------------------------
    dv = "num_genres_disliked"

    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    labels = {
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

    tab1, fit1, before1, dropped1, frame1 = fit_model(df, dv, m1_x, "Model 1 (SES)", labels)
    tab2, fit2, before2, dropped2, frame2 = fit_model(df, dv, m2_x, "Model 2 (Demographic)", labels)
    tab3, fit3, before3, dropped3, frame3 = fit_model(df, dv, m3_x, "Model 3 (Political intolerance)", labels)

    # Merge tables side-by-side on term
    table = tab1.merge(tab2, on="term", how="outer").merge(tab3, on="term", how="outer")

    fit_stats = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2"]]
    fit_stats["r2"] = pd.to_numeric(fit_stats["r2"], errors="coerce").round(3)
    fit_stats["adj_r2"] = pd.to_numeric(fit_stats["adj_r2"], errors="coerce").round(3)

    # Write human-readable outputs
    def model_diagnostics_text(model_name, x_cols, before, dropped, fit, frame):
        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append("Variables in model:")
        lines.append(", ".join([labels.get(c, c) for c in x_cols]))
        lines.append("")
        lines.append("Non-missing BEFORE listwise deletion (DV + X):")
        before_named = before.copy()
        before_named.index = [("DV: " + i) if i == dv else labels.get(i, i) for i in before_named.index]
        lines.append(before_named.to_string())
        lines.append("")
        if dropped:
            lines.append("Dropped predictors due to zero variance AFTER listwise deletion:")
            lines.append(", ".join([labels.get(c, c) for c in dropped]))
            lines.append("")
        lines.append("Fit statistics:")
        lines.append(f"N = {fit['n']}")
        lines.append(f"R^2 = {fit['r2']:.6f}" if np.isfinite(fit["r2"]) else "R^2 = NA")
        lines.append(f"Adj R^2 = {fit['adj_r2']:.6f}" if np.isfinite(fit["adj_r2"]) else "Adj R^2 = NA")
        lines.append("")
        if len(frame) > 0:
            # quick distribution checks for dummies to detect coding collapse
            dummy_vars = [c for c in x_cols if c in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]]
            if dummy_vars:
                lines.append("Dummy means in estimation sample (proportion=1):")
                for c in dummy_vars:
                    v = frame[c]
                    lines.append(f"- {labels.get(c,c)}: mean={v.mean():.3f}, n={int(v.notna().sum())}")
                lines.append("")
        return "\n".join(lines) + "\n"

    write_text("./output/table1_model1_ses_diagnostics.txt", model_diagnostics_text("Model 1 (SES)", m1_x, before1, dropped1, fit1, frame1))
    write_text("./output/table1_model2_demographic_diagnostics.txt", model_diagnostics_text("Model 2 (Demographic)", m2_x, before2, dropped2, fit2, frame2))
    write_text("./output/table1_model3_political_intolerance_diagnostics.txt", model_diagnostics_text("Model 3 (Political intolerance)", m3_x, before3, dropped3, fit3, frame3))

    write_text(
        "./output/table1_regression_table.txt",
        "Standardized OLS coefficients (β) for predictors; unstandardized constant.\n"
        "Stars: * p<.05, ** p<.01, *** p<.001 (two-tailed). Constant not star-marked.\n\n"
        + table.to_string(index=False)
        + "\n\nFit statistics:\n"
        + fit_stats.to_string(index=False)
        + "\n"
    )
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # Return results as dict of DataFrames
    return {
        "table1": table,
        "fit_stats": fit_stats,
        "missingness_audit": audit,
        "model_frames": {
            "Model 1 (SES)": frame1,
            "Model 2 (Demographic)": frame2,
            "Model 3 (Political intolerance)": frame3,
        },
    }