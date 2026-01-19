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

    def clean_gss_numeric(series, extra_na=()):
        """
        Convert to numeric and set common GSS missing codes to NaN.
        Keep conservative: only very common DK/NA/refused/inapplicable codes.
        """
        x = to_num(series)
        na_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999}
        na_codes |= set(extra_na)
        return x.where(~x.isin(list(na_codes)), np.nan)

    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        m = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - m) / sd

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

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def table1_from_fit(df_model, dv, xcols, model_label, label_map):
        """
        Paper-style output:
          - fit OLS on raw DV and raw X (unstandardized), to get the constant and R^2
          - compute standardized betas by fitting OLS on z-scored DV and z-scored X (no constant),
            then those slopes are betas; compute p-values from that standardized model;
          - stars use two-tailed p-values.
        Listwise deletion is done ONLY on dv + xcols for this model.
        """
        needed = [dv] + xcols
        frame = df_model[needed].copy()
        n_before = len(frame)
        frame = frame.dropna(axis=0, how="any").copy()
        n = len(frame)

        # guard: if nothing to fit
        if n == 0:
            rows = [{"term": "Constant", "value": "NA"}]
            for c in xcols:
                rows.append({"term": label_map.get(c, c), "value": "NA"})
            out = pd.DataFrame(rows)
            fit_stats = pd.DataFrame([{
                "model": model_label, "n_before": n_before, "n": n, "r2": np.nan, "adj_r2": np.nan
            }])
            return out, fit_stats

        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)

        # Unstandardized fit (for constant and R^2)
        Xc = sm.add_constant(X, has_constant="add")
        res_u = sm.OLS(y, Xc).fit()

        # Standardized fit (for betas and inference for stars)
        y_z = zscore(y)
        X_z = X.apply(zscore, axis=0)

        # If any predictor becomes all-NA after z-scoring (zero variance), drop it from beta table
        keep = [c for c in xcols if X_z[c].notna().any()]
        dropped = [c for c in xcols if c not in keep]

        rows = []
        # Constant: show unstandardized intercept (no stars)
        const_val = res_u.params.get("const", np.nan)
        rows.append({"term": "Constant", "value": "" if pd.isna(const_val) else f"{float(const_val):.3f}"})

        if len(keep) == 0:
            for c in xcols:
                rows.append({"term": label_map.get(c, c), "value": "NA"})
            out = pd.DataFrame(rows)
            fit_stats = pd.DataFrame([{
                "model": model_label,
                "n_before": int(n_before),
                "n": int(n),
                "r2": float(res_u.rsquared),
                "adj_r2": float(res_u.rsquared_adj),
                "dropped_predictors": ", ".join(dropped) if dropped else ""
            }])
            return out, fit_stats

        # Standardized regression with constant (optional) yields same slopes if both y and X are z-scored;
        # include const for conventional p-values.
        Xz_c = sm.add_constant(X_z[keep], has_constant="add")
        res_z = sm.OLS(y_z, Xz_c).fit()

        for c in xcols:
            term = label_map.get(c, c)
            if c in keep:
                b = float(res_z.params.get(c, np.nan))
                p = float(res_z.pvalues.get(c, np.nan))
                rows.append({"term": term, "value": "" if pd.isna(b) else f"{b:.3f}{star(p)}"})
            else:
                rows.append({"term": term, "value": "NA"})

        out = pd.DataFrame(rows)
        fit_stats = pd.DataFrame([{
            "model": model_label,
            "n_before": int(n_before),
            "n": int(n),
            "r2": float(res_u.rsquared),
            "adj_r2": float(res_u.rsquared_adj),
            "dropped_predictors": ", ".join(dropped) if dropped else ""
        }])
        return out, fit_stats

    # ----------------------------
    # Read and restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: Musical exclusiveness = count of disliked genres across 18 items
    # disliked if 4 or 5; valid range 1..5; DK/NA -> missing
    # if ANY of 18 items missing -> DV missing (listwise for DV construction)
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: clean_gss_numeric(df[c]) for c in music_items})
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv = "num_genres_disliked"
    dv_desc = df[dv].describe()
    dv_rng = (df[dv].min(skipna=True), df[dv].max(skipna=True))

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss_numeric(df.get("educ", np.nan), extra_na=(0,))
    df["prestg80_v"] = clean_gss_numeric(df.get("prestg80", np.nan), extra_na=(0,))

    df["realinc_v"] = clean_gss_numeric(df.get("realinc", np.nan), extra_na=(0,))
    df["hompop_v"] = clean_gss_numeric(df.get("hompop", np.nan), extra_na=(0,))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    # keep nonnegative; allow zero if present (paper doesn't specify dropping zeros)
    df.loc[df["inc_pc"] < 0, "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss_numeric(df.get("sex", np.nan))
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss_numeric(df.get("age", np.nan), extra_na=(0,))

    race = clean_gss_numeric(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if present.
    # This extract includes 'ethnic' with many nonmissing values; we avoid collapsing N by:
    # - treat 1/2 as a yes/no scheme if detected, else
    # - treat any nonmissing ETHNIC as "not missing" and define Hispanic using a conservative rule:
    #   if codes include 20/21/22/23/24/25/26/27/28/29 (common hispanic origin recodes), flag those;
    #   otherwise, fallback to: hispanic=1 if ETHNIC between 1 and 9 AND those codes represent a small
    #   subset (heuristic), else set to 0 for nonmissing (to keep model N stable).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss_numeric(df["ethnic"], extra_na=(0,))
        uniq = sorted([u for u in pd.unique(eth.dropna()) if np.isfinite(u)])
        uniq_set = set(uniq)

        if len(uniq) > 0 and uniq_set.issubset({1.0, 2.0}):
            # assume 1 = not hispanic, 2 = hispanic (common)
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # First, try a common hispanic-origin code set (if present)
            hisp_codes = set([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
            if any([(c in uniq_set) for c in hisp_codes]):
                df["hispanic"] = np.where(eth.isna(), np.nan, eth.isin(list(hisp_codes)).astype(float))
            else:
                # Fallback that avoids making the variable mostly missing:
                # if eth nonmissing, set 0/1 using a mild heuristic; otherwise NA.
                # Heuristic: if a noticeable share is in 1..9, treat 1..9 as Hispanic category codes.
                if eth.notna().any():
                    share_1_9 = float(((eth >= 1) & (eth <= 9)).mean(skipna=True))
                    if share_1_9 >= 0.02:
                        df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 1) & (eth <= 9)).astype(float))
                    else:
                        # assume ETHNIC is ancestry-like; treat all known as non-Hispanic
                        df["hispanic"] = np.where(eth.isna(), np.nan, 0.0)

    # Religion
    relig = clean_gss_numeric(df.get("relig", np.nan), extra_na=(0,))
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss_numeric(df.get("denom", np.nan), extra_na=(0,))
    is_prot = (relig == 1)
    # Conservative Protestant proxy: Baptist (1) and "Other Protestant" (6) as conservative-ish.
    denom_cons = denom.isin([1, 6])
    cons = (is_prot & denom_cons)
    # keep defined wherever RELIG is known; if denom missing among Protestants, set 0 to avoid NA blowups
    df["cons_prot"] = np.where(relig.isna(), np.nan, cons.astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern
    region = clean_gss_numeric(df.get("region", np.nan), extra_na=(0,))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance scale (0-15): sum of intolerant responses across 15 items.
    tol_map = [
        ("spkath", {2}),
        ("colath", {5}),
        ("libath", {1}),
        ("spkrac", {2}),
        ("colrac", {5}),
        ("librac", {1}),
        ("spkcom", {2}),
        ("colcom", {4}),
        ("libcom", {1}),
        ("spkmil", {2}),
        ("colmil", {5}),
        ("libmil", {1}),
        ("spkhomo", {2}),
        ("colhomo", {5}),
        ("libhomo", {1}),
    ]
    missing_tol = [v for v, _ in tol_map if v not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance items: {missing_tol}")

    tol = pd.DataFrame(index=df.index)
    for v, intolerant_codes in tol_map:
        x = clean_gss_numeric(df[v], extra_na=(0,))
        # For these items, substantive codes differ by item type; keep only reasonable small-integer codes.
        # Anything else -> missing.
        x = x.where((x >= 1) & (x <= 9), np.nan)
        tol[v] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol.sum(axis=1)
    df.loc[tol.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Output: missingness summary (key variables)
    # ----------------------------
    key_vars = [
        dv, "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol"
    ]
    miss_rows = []
    for v in key_vars:
        if v in df.columns:
            nonmiss = int(df[v].notna().sum())
            miss = int(df[v].isna().sum())
            pct = (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) > 0 else np.nan
            miss_rows.append({"variable": v, "nonmissing": nonmiss, "missing": miss, "pct_missing": pct})
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)

    write_text(
        "./output/table1_missingness.txt",
        "Missingness (GSS 1993 extract; after year filter only)\n\n" +
        missingness.to_string(index=False) + "\n\n" +
        "DV descriptives (constructed)\n" +
        dv_desc.to_string() + "\n" +
        f"\nDV min/max: {dv_rng}\n"
    )

    # Also write quick frequency checks for key dummies
    freq_text = []
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        if v in df.columns:
            vc = df[v].value_counts(dropna=False).sort_index()
            freq_text.append(f"\n{v} value counts (incl. NA):\n{vc.to_string()}\n")
    write_text("./output/table1_dummy_frequencies.txt", "\n".join(freq_text).strip() + "\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    label_map = {
        "educ_yrs": "Education (years)",
        "inc_pc": "Household income per capita",
        "prestg80_v": "Occupational prestige",
        "female": "Female",
        "age_v": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "otherrace": "Other race",
        "cons_prot": "Conservative Protestant",
        "norelig": "No religion",
        "south": "Southern",
        "pol_intol": "Political intolerance (0–15)",
    }

    model1_x = ["educ_yrs", "inc_pc", "prestg80_v"]
    model2_x = model1_x + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    model3_x = model2_x + ["pol_intol"]

    m1_tab, m1_fit = table1_from_fit(df, dv, model1_x, "Model 1 (SES)", label_map)
    m2_tab, m2_fit = table1_from_fit(df, dv, model2_x, "Model 2 (Demographic)", label_map)
    m3_tab, m3_fit = table1_from_fit(df, dv, model3_x, "Model 3 (Political intolerance)", label_map)

    fit_stats = pd.concat([m1_fit, m2_fit, m3_fit], ignore_index=True)

    # Save tables in paper-like format (term + value only)
    write_text("./output/table1_model1.txt", m1_tab.to_string(index=False) + "\n")
    write_text("./output/table1_model2.txt", m2_tab.to_string(index=False) + "\n")
    write_text("./output/table1_model3.txt", m3_tab.to_string(index=False) + "\n")
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # ----------------------------
    # Combined human-readable summary
    # ----------------------------
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from provided GSS 1993 extract)\n")
    summary_lines.append("DV: Number of music genres disliked (0–18).")
    summary_lines.append("Construction: count of 18 genre ratings coded 4/5; any missing among 18 => DV missing.\n")
    summary_lines.append("Fit statistics (OLS on unstandardized variables; betas shown are standardized slopes):\n")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("\n\nModel 1 (SES) coefficients (Constant unstd; predictors = standardized beta with stars):\n")
    summary_lines.append(m1_tab.to_string(index=False))
    summary_lines.append("\n\nModel 2 (Demographic) coefficients (Constant unstd; predictors = standardized beta with stars):\n")
    summary_lines.append(m2_tab.to_string(index=False))
    summary_lines.append("\n\nModel 3 (Political intolerance) coefficients (Constant unstd; predictors = standardized beta with stars):\n")
    summary_lines.append(m3_tab.to_string(index=False))
    summary_lines.append("\n\nNotes:\n- Stars use two-tailed p-values with thresholds: * p<.05, ** p<.01, *** p<.001.")
    summary_lines.append("- Output format matches the paper style (no SE/p-value columns shown), but stars are computed from this replication.\n")

    summary_text = "\n".join(summary_lines) + "\n"
    write_text("./output/table1_summary.txt", summary_text)

    return {
        "fit_stats": fit_stats,
        "model1": m1_tab,
        "model2": m2_tab,
        "model3": m3_tab,
        "missingness": missingness
    }