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
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def coerce_gss_missing(series):
        """
        Conservative missing-code handler:
        - numeric coercion
        - set common GSS 'missing-ish' codes to NaN: -9..-1 and large sentinels (97/98/99, 997/998/999, 9998/9999)
        - note: we then ALSO apply variable-specific valid-range filters where appropriate
        """
        s = to_num(series).copy()
        s = s.replace([-9, -8, -7, -6, -5, -4, -3, -2, -1], np.nan)
        s = s.replace([97, 98, 99, 997, 998, 999, 9997, 9998, 9999], np.nan)
        return s

    def valid_range(s, lo=None, hi=None, valid_set=None):
        s = coerce_gss_missing(s)
        if valid_set is not None:
            return s.where(s.isin(valid_set), np.nan)
        if lo is not None:
            s = s.where(s >= lo, np.nan)
        if hi is not None:
            s = s.where(s <= hi, np.nan)
        return s

    def dummy_from_codes(s, one_codes, valid_codes):
        s = valid_range(s, valid_set=valid_codes)
        return pd.Series(np.where(s.isna(), np.nan, s.isin(one_codes).astype(float)), index=s.index)

    def zscore_sample(x):
        x = to_num(x)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=1)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index)
        return (x - mu) / sd

    def standardized_betas_from_unstandardized(y, X, params):
        """
        Compute standardized betas from an unstandardized OLS:
            beta_j = b_j * sd(X_j) / sd(y)
        Uses estimation-sample SDs (ddof=1).
        """
        y = to_num(y)
        sd_y = y.std(skipna=True, ddof=1)
        out = {}
        if pd.isna(sd_y) or sd_y == 0:
            for c in X.columns:
                out[c] = np.nan
            return out

        for c in X.columns:
            sd_x = to_num(X[c]).std(skipna=True, ddof=1)
            b = params.get(c, np.nan)
            if pd.isna(sd_x) or sd_x == 0 or pd.isna(b):
                out[c] = np.nan
            else:
                out[c] = float(b) * float(sd_x) / float(sd_y)
        return out

    def fit_table1_model(df, dv, xcols, model_name, pretty_labels):
        """
        - Listwise deletion on DV + xcols
        - OLS with intercept
        - Report: unstandardized constant; standardized betas for predictors + stars
        - Save a human-readable text file for each model
        """
        needed = [dv] + xcols
        d = df[needed].copy()

        # Ensure numeric
        for c in needed:
            d[c] = to_num(d[c]).replace([np.inf, -np.inf], np.nan)

        nonmissing_before = d.notna().sum().sort_index()
        d = d.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors after listwise deletion (prevents NaN estimates)
        kept = []
        dropped = []
        for c in xcols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0:
            fit = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            tab = pd.DataFrame([{"term": "Constant", "beta": np.nan, "sig": ""}] + [
                {"term": pretty_labels.get(c, c), "beta": np.nan, "sig": ""} for c in xcols
            ])
            return tab, fit, nonmissing_before, dropped, d

        y = d[dv].astype(float)
        X = d[kept].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstandardized(y, X, res.params)
        pvals = res.pvalues.to_dict()

        rows = []
        # Constant: unstandardized
        const_label = "Constant"
        rows.append({
            "term": const_label,
            "beta": float(res.params.get("const", np.nan)),
            "sig": ""  # Table 1 typically does not star constants
        })

        for c in xcols:
            label = pretty_labels.get(c, c)
            if c in kept:
                bstd = betas.get(c, np.nan)
                p = pvals.get(c, np.nan)
                rows.append({"term": label, "beta": bstd, "sig": star(p)})
            else:
                rows.append({"term": label, "beta": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)

        fit = {
            "model": model_name,
            "n": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
        }
        return tab, fit, nonmissing_before, dropped, d

    def format_table_for_txt(tab):
        out = tab.copy()
        out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
        # show 3 decimals like typical journal tables
        out["beta_fmt"] = out["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        out["beta_star"] = out.apply(
            lambda r: (r["beta_fmt"] if r["term"] == "Constant" else (r["beta_fmt"] + r["sig"] if r["beta_fmt"] != "" else "")),
            axis=1
        )
        return out[["term", "beta_star"]]

    def write_model_txt(path, model_name, tab, fit, nonmissing_before, dropped, dv_name, xcols, pretty_labels):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{model_name}\n")
            f.write("=" * len(model_name) + "\n\n")

            f.write("Model variables (internal -> label):\n")
            f.write(f"DV: {dv_name}\n")
            for c in xcols:
                f.write(f"  {c} -> {pretty_labels.get(c, c)}\n")
            f.write("\n")

            f.write("Non-missing counts BEFORE listwise deletion (DV + model IVs):\n")
            f.write(nonmissing_before.to_string())
            f.write("\n\n")

            if dropped:
                f.write("Dropped predictors due to zero variance AFTER listwise deletion:\n")
                f.write(", ".join(dropped) + "\n\n")

            f.write("Fit statistics:\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.6f}\n")
            f.write(f"Adj R^2 = {fit['adj_r2']:.6f}\n\n")

            f.write("Coefficients (Table 1 style):\n")
            f.write("  - Constant: unstandardized (raw DV units)\n")
            f.write("  - Predictors: standardized betas (beta = b * SD(X)/SD(Y))\n")
            f.write("  - Stars from two-tailed p-values: * p<.05, ** p<.01, *** p<.001\n\n")

            f.write(format_table_for_txt(tab).to_string(index=False))
            f.write("\n")

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: # music genres disliked (0-18), listwise on 18 items
    # 1..5 valid; 4 or 5 = disliked
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: valid_range(df[c], valid_set=[1, 2, 3, 4, 5]) for c in music_items})
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1, min_count=len(music_items))
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv_desc = df["num_genres_disliked"].describe()
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Number of music genres disliked\n")
        f.write("Construction: count across 18 genres where response is 4 ('dislike') or 5 ('dislike very much').\n")
        f.write("Missing handling: responses not in 1..5 treated as missing; DV requires complete responses on all 18 items.\n\n")
        f.write(dv_desc.to_string())
        f.write("\n")

    # ----------------------------
    # Predictors (recoding with conservative valid ranges)
    # ----------------------------
    # SES
    df["educ_yrs"] = valid_range(df.get("educ", np.nan), lo=0, hi=30)

    # PRESTG80 typical range is around 10-90; keep broadly
    df["prestg80"] = valid_range(df.get("prestg80", np.nan), lo=0, hi=100)

    # REALINC is continuous dollars; HOMPOP household size
    df["realinc"] = valid_range(df.get("realinc", np.nan), lo=0, hi=None)
    df["hompop"] = valid_range(df.get("hompop", np.nan), lo=1, hi=50)
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    # SEX: 1 male, 2 female
    df["female"] = dummy_from_codes(df.get("sex", np.nan), one_codes=[2], valid_codes=[1, 2])

    # AGE: keep plausible; retain 89 top-code as 89
    df["age"] = valid_range(df.get("age", np.nan), lo=18, hi=89)

    # RACE: 1 White, 2 Black, 3 Other
    race = valid_range(df.get("race", np.nan), valid_set=[1, 2, 3])
    df["black"] = pd.Series(np.where(race.isna(), np.nan, (race == 2).astype(float)), index=df.index)
    df["otherrace"] = pd.Series(np.where(race.isna(), np.nan, (race == 3).astype(float)), index=df.index)

    # Hispanic: use ETHNIC if present (GSS often: 1=not hispanic, 2=hispanic)
    # Keep only 1/2 as valid; others missing.
    if "ethnic" in df.columns:
        eth = valid_range(df["ethnic"], valid_set=[1, 2])
        df["hispanic"] = pd.Series(np.where(eth.isna(), np.nan, (eth == 2).astype(float)), index=df.index)
    else:
        df["hispanic"] = np.nan

    # Religion: RELIG (1 Protestant, 2 Catholic, 3 Jewish, 4 None, 5 Other)
    relig = valid_range(df.get("relig", np.nan), valid_set=[1, 2, 3, 4, 5])
    df["norelig"] = pd.Series(np.where(relig.isna(), np.nan, (relig == 4).astype(float)), index=df.index)

    # Conservative Protestant: approximate using RELIG==1 and DENOM in a conservative set.
    # This is an approximation but should not destroy N; crucially, do NOT set most cases to missing.
    denom = valid_range(df.get("denom", np.nan), lo=0, hi=20)
    # Define conservative denom codes (commonly: Baptist=1; "other" often includes Pentecostal/fundamentalist)
    denom_cons = denom.isin([1, 6, 7])  # 7 sometimes "no denomination"; include to reduce missing/over-strictness
    df["cons_prot"] = pd.Series(
        np.where(relig.isna(), np.nan, ((relig == 1) & denom_cons.fillna(False)).astype(float)),
        index=df.index
    )
    # If denom missing but relig known, treat as 0 rather than missing to avoid collapsing sample
    df.loc[(relig.notna()) & (denom.isna()), "cons_prot"] = 0.0

    # Region: REGION==3 is South (per mapping)
    region = valid_range(df.get("region", np.nan), lo=1, hi=9)
    df["south"] = pd.Series(np.where(region.isna(), np.nan, (region == 3).astype(float)), index=df.index)

    # Political intolerance (0-15): sum of 15 "intolerant" indicators; require complete across 15 items
    tol_items = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    # Use explicit valid sets per item type (prevents turning valid codes into missing)
    valid_spk = [1, 2]            # allowed / not allowed
    valid_col_notcom = [4, 5]     # allowed/should be allowed vs not allowed (varies by item)
    valid_col_com = [4, 5]        # for COLCOM specifically: 4 yes fired, 5 not fired (per mapping)
    valid_lib = [1, 2]            # remove / not remove

    intoler = pd.DataFrame(index=df.index)
    for v, bad in tol_items.items():
        if v not in df.columns:
            df[v] = np.nan
        if v.startswith("spk"):
            s = valid_range(df[v], valid_set=valid_spk)
        elif v.startswith("col"):
            s = valid_range(df[v], valid_set=(valid_col_com if v == "colcom" else valid_col_notcom))
        elif v.startswith("lib"):
            s = valid_range(df[v], valid_set=valid_lib)
        else:
            s = coerce_gss_missing(df[v])
        intoler[v] = np.where(s.isna(), np.nan, (s == bad).astype(float))

    df["pol_intol"] = intoler.sum(axis=1, min_count=len(tol_items))
    df.loc[intoler.isna().any(axis=1), "pol_intol"] = np.nan

    pol_desc = df["pol_intol"].describe()
    with open("./output/table1_pol_intol_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("Political intolerance scale (0-15)\n")
        f.write("Construction: sum of 15 intolerant responses across 5 groups x 3 contexts.\n")
        f.write("Missing handling: requires complete responses across all 15 items.\n\n")
        f.write(pol_desc.to_string())
        f.write("\n")

    # ----------------------------
    # Labels for Table 1
    # ----------------------------
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
        "pol_intol": "Political intolerance (0–15)",
    }

    # ----------------------------
    # Fit three models (listwise per model)
    # ----------------------------
    dv = "num_genres_disliked"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    tab1, fit1, before1, dropped1, frame1 = fit_table1_model(df, dv, m1_x, "Model 1 (SES)", labels)
    tab2, fit2, before2, dropped2, frame2 = fit_table1_model(df, dv, m2_x, "Model 2 (Demographic)", labels)
    tab3, fit3, before3, dropped3, frame3 = fit_table1_model(df, dv, m3_x, "Model 3 (Political intolerance)", labels)

    # Save model outputs
    write_model_txt("./output/table1_model1_ses.txt", "Model 1 (SES)", tab1, fit1, before1, dropped1, dv, m1_x, labels)
    write_model_txt("./output/table1_model2_demographic.txt", "Model 2 (Demographic)", tab2, fit2, before2, dropped2, dv, m2_x, labels)
    write_model_txt("./output/table1_model3_political_intolerance.txt", "Model 3 (Political intolerance)", tab3, fit3, before3, dropped3, dv, m3_x, labels)

    # Combined fit stats table + combined coefficient table
    fit_stats = pd.DataFrame([
        {"model": fit1["model"], "n": fit1["n"], "r2": fit1["r2"], "adj_r2": fit1["adj_r2"]},
        {"model": fit2["model"], "n": fit2["n"], "r2": fit2["r2"], "adj_r2": fit2["adj_r2"]},
        {"model": fit3["model"], "n": fit3["n"], "r2": fit3["r2"], "adj_r2": fit3["adj_r2"]},
    ])

    def wide_coeff_table(tabs, model_names):
        # Merge by term
        out = None
        for tab, mn in zip(tabs, model_names):
            t = format_table_for_txt(tab).copy()
            t = t.rename(columns={"beta_star": mn})
            if out is None:
                out = t
            else:
                out = out.merge(t, on="term", how="outer")
        return out

    coeff_wide = wide_coeff_table(
        [tab1, tab2, tab3],
        ["Model 1 (SES)", "Model 2 (Demographic)", "Model 3 (Political intolerance)"]
    )

    # Save combined summary
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication summary (computed from raw data)\n")
        f.write("===================================================\n\n")
        f.write("Fit statistics:\n")
        f.write(fit_stats.to_string(index=False))
        f.write("\n\n")
        f.write("Coefficients (standardized betas for predictors; unstandardized constant):\n")
        f.write(coeff_wide.to_string(index=False))
        f.write("\n")

    # Return results as dict of DataFrames
    return {
        "fit_stats": fit_stats,
        "coefficients_wide": coeff_wide,
        "Model 1 (SES)": tab1,
        "Model 2 (Demographic)": tab2,
        "Model 3 (Political intolerance)": tab3,
        "model_frames": {
            "Model 1 (SES)": frame1,
            "Model 2 (Demographic)": frame2,
            "Model 3 (Political intolerance)": frame3,
        },
    }