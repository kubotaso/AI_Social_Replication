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

    def gss_clean_numeric(s, extra_na=None):
        """
        Conservative GSS missing handling:
        - Always treat {7,8,9,97,98,99,997,998,999} as missing where present.
        - Optionally add variable-specific missing codes via extra_na.
        """
        x = to_num(s)
        na_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999}
        if extra_na:
            na_codes |= set(extra_na)
        x = x.where(~x.isin(list(na_codes)), np.nan)
        return x

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

    def safe_sd(s):
        s = pd.to_numeric(s, errors="coerce")
        sd = s.std(ddof=0)
        return np.nan if (pd.isna(sd) or sd == 0) else float(sd)

    def standardized_betas_from_unstd(y, X, params_unstd):
        """
        y: Series
        X: DataFrame (WITHOUT constant)
        params_unstd: Series including 'const' and X columns
        returns: dict {col: beta}
        """
        sd_y = safe_sd(y)
        betas = {}
        for c in X.columns:
            b = float(params_unstd.get(c, np.nan))
            sd_x = safe_sd(X[c])
            if pd.isna(b) or pd.isna(sd_x) or pd.isna(sd_y):
                betas[c] = np.nan
            else:
                betas[c] = b * (sd_x / sd_y)
        return betas

    def fit_table1_model(df, dv, x_cols, label_map, model_name):
        """
        - model-specific listwise deletion: drop only missing on dv + x_cols
        - OLS on unstandardized variables (for constant)
        - compute standardized betas for predictors from unstandardized slopes
        """
        needed = [dv] + x_cols
        d = df[needed].copy()

        nonmissing_before = d.notna().sum()

        # listwise deletion for THIS model only
        d = d.dropna(axis=0, how="any").copy()

        # drop zero-variance predictors after deletion (prevents singularities)
        kept = []
        dropped = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0 or len(kept) == 0:
            # produce an empty-but-structured output
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in x_cols:
                rows.append({"term": label_map.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            fit = {"model": model_name, "n": int(len(d)), "r2": np.nan, "adj_r2": np.nan}
            return tab, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstd(y, X, res.params)

        rows = []
        # Table 1: constant shown (unstandardized). Stars in paper are for predictors;
        # we compute p-values anyway; keep stars for predictors only in the printed "beta_star".
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": star(float(res.pvalues.get("const", np.nan))),
        })

        for c in x_cols:
            lab = label_map.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": lab,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": star(p),
                })
            else:
                rows.append({"term": lab, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return tab, fit, nonmissing_before, dropped, d

    def write_text(path, txt):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

    def format_model_txt(model_name, fit, nonmissing_before, dropped, tab):
        # Build a Table 1-like display: Constant (b) and predictors (beta with stars)
        out = tab.copy()

        # create display column
        beta_disp = []
        for _, r in out.iterrows():
            if r["term"] == "Constant":
                if pd.isna(r["b"]):
                    beta_disp.append("")
                else:
                    beta_disp.append(f"{float(r['b']):.3f}")
            else:
                if pd.isna(r["beta"]):
                    beta_disp.append("")
                else:
                    beta_disp.append(f"{float(r['beta']):.3f}{r['sig']}")
        out["Table1"] = beta_disp

        # keep a compact table for text
        show = out[["term", "Table1"]].copy()

        lines = []
        lines.append(model_name)
        lines.append("=" * len(model_name))
        lines.append("")
        lines.append("Non-missing counts BEFORE model-specific listwise deletion:")
        lines.append(nonmissing_before.to_string())
        lines.append("")
        if dropped:
            lines.append("Dropped predictors due to zero variance AFTER listwise deletion:")
            lines.append(", ".join(dropped))
            lines.append("")
        lines.append("Fit statistics:")
        lines.append(f"N = {fit['n']}")
        lines.append(f"R^2 = {fit['r2'] if pd.notna(fit['r2']) else np.nan}")
        lines.append(f"Adj R^2 = {fit['adj_r2'] if pd.notna(fit['adj_r2']) else np.nan}")
        lines.append("")
        lines.append("Coefficients (Table 1 style):")
        lines.append("- Constant shown as unstandardized intercept (b)")
        lines.append("- Predictors shown as standardized betas (β) with stars from two-tailed p-values")
        lines.append("- Stars: * p<.05, ** p<.01, *** p<.001")
        lines.append("")
        lines.append(show.to_string(index=False))
        lines.append("")
        return "\n".join(lines)

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0-18), listwise across 18 items
    # - valid: 1..5
    # - disliked: 4 or 5
    # - DK/NA/etc -> missing
    # - if any of 18 missing -> DV missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: gss_clean_numeric(df[c]) for c in music_items})
    # only 1..5 are valid substantive responses
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float)
    disliked = disliked.where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1, min_count=len(music_items))
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Constructed as count of 18 genre items rated 4 ('dislike') or 5 ('dislike very much').\n"
        "Non-1..5 and GSS missing codes treated as missing. Listwise across all 18 items.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (with conservative missing handling)
    # ----------------------------
    # SES
    df["educ_yrs"] = gss_clean_numeric(df.get("educ", np.nan), extra_na=[0])
    df["prestg80"] = gss_clean_numeric(df.get("prestg80", np.nan), extra_na=[0])

    # Income per capita: REALINC / HOMPOP
    df["realinc"] = gss_clean_numeric(df.get("realinc", np.nan), extra_na=[0])
    df["hompop"] = gss_clean_numeric(df.get("hompop", np.nan), extra_na=[0])
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    # also treat nonpositive per-capita as missing (shouldn't happen, but guards coding)
    df.loc[df["inc_pc"] <= 0, "inc_pc"] = np.nan

    # Demographics
    sex = gss_clean_numeric(df.get("sex", np.nan))
    # GSS: 1=male, 2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age"] = gss_clean_numeric(df.get("age", np.nan), extra_na=[0])

    # Race: 1=white, 2=black, 3=other
    race = gss_clean_numeric(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if available in this extract.
    # In this file ETHNIC appears numeric (examples include 29, 1, 21, 97, 8).
    # Many GSS extracts use ETHNIC as an ancestry/ethnic group code, not a yes/no Hispanic flag.
    # Best effort here: if values are 1/2 (common yes/no), use 2=yes. Otherwise treat as missing.
    if "ethnic" in df.columns:
        eth = gss_clean_numeric(df["ethnic"])
        # If looks binary 1/2 in the observed data, use it. Otherwise set to missing.
        # (prevents collapsing the sample by forcing almost all to NA)
        uniq = sorted([u for u in pd.unique(eth.dropna()) if np.isfinite(u)])
        if set(uniq).issubset({1.0, 2.0}) and len(uniq) >= 1:
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            df["hispanic"] = np.nan
    else:
        df["hispanic"] = np.nan

    # Religion: RELIG (example row shows 1..5); treat 0/7/8/9 etc as missing
    relig = gss_clean_numeric(df.get("relig", np.nan), extra_na=[0])
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: derived from RELIG==1 and DENOM in a conservative set.
    # Without the paper's exact scheme, use a conservative, non-collapsing proxy:
    # Protestant (RELIG==1) with DENOM indicating Baptist or other conservative/sectarian groups.
    denom = gss_clean_numeric(df.get("denom", np.nan), extra_na=[0])
    # DENOM codes vary; in this extract values like 4, 6 appear. We'll treat denom valid if 1..14.
    denom = denom.where(denom.isin(list(range(1, 15))), np.nan)
    is_prot = (relig == 1)
    # Proxy conservative denominations: Baptist(1) and "other Protestant"(6) (common in GSS)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where((relig.notna()) & (denom.notna()), (is_prot & denom_cons).astype(float), np.nan)

    # Southern: REGION==3
    region = gss_clean_numeric(df.get("region", np.nan), extra_na=[0])
    # valid regions often 1..9
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum of 15 intolerant indicators; listwise across 15 items.
    tol_items = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    tol = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        if v in df.columns:
            s = gss_clean_numeric(df[v], extra_na=[0])
        else:
            s = pd.Series(np.nan, index=df.index)
        tol[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    df["pol_intol"] = tol.sum(axis=1, min_count=len(tol_items))
    df.loc[tol.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
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
        "pol_intol": "Political intolerance (0–15)",
    }

    tab1, fit1, before1, dropped1, frame1 = fit_table1_model(df, dv, m1_x, label_map, "Model 1 (SES)")
    tab2, fit2, before2, dropped2, frame2 = fit_table1_model(df, dv, m2_x, label_map, "Model 2 (Demographic)")
    tab3, fit3, before3, dropped3, frame3 = fit_table1_model(df, dv, m3_x, label_map, "Model 3 (Political intolerance)")

    # Save human-readable outputs
    write_text("./output/table1_model1_ses.txt", format_model_txt("Model 1 (SES)", fit1, before1, dropped1, tab1))
    write_text("./output/table1_model2_demographic.txt", format_model_txt("Model 2 (Demographic)", fit2, before2, dropped2, tab2))
    write_text("./output/table1_model3_political_intolerance.txt", format_model_txt("Model 3 (Political intolerance)", fit3, before3, dropped3, tab3))

    # Save a compact fit stats file
    fit_df = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2"]]
    write_text("./output/table1_fit_stats.txt", fit_df.to_string(index=False) + "\n")

    # Save missingness diagnostics for constructed variables
    diag_vars = [dv] + m3_x
    miss = pd.DataFrame({
        "variable": diag_vars,
        "nonmissing": [int(df[v].notna().sum()) if v in df.columns else 0 for v in diag_vars],
        "missing": [int(df[v].isna().sum()) if v in df.columns else len(df) for v in diag_vars],
        "pct_missing": [
            float(df[v].isna().mean() * 100.0) if v in df.columns and len(df) > 0 else np.nan
            for v in diag_vars
        ],
    })
    write_text("./output/table1_missingness_diagnostics.txt", miss.to_string(index=False) + "\n")

    # Return results as dict of DataFrames
    return {
        "fit_stats": fit_df,
        "Model 1 (SES)": tab1,
        "Model 2 (Demographic)": tab2,
        "Model 3 (Political intolerance)": tab3,
        "missingness": miss,
        "model_frames": {
            "Model 1 (SES)": frame1,
            "Model 2 (Demographic)": frame2,
            "Model 3 (Political intolerance)": frame3,
        },
    }