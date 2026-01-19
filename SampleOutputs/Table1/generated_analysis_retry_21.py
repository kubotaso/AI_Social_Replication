def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

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

    def sd_sample(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        return float(np.sqrt(v)) if pd.notna(v) and v > 0 else np.nan

    def clean_gss_numeric(series, extra_na=()):
        """
        Conservative numeric cleaner: coerce to numeric and set common GSS
        missing-value codes to NA. Do NOT treat '0' as missing by default
        (0 can be a valid code in some items/indices).
        """
        x = pd.to_numeric(series, errors="coerce")
        na_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999}
        na_codes |= set(extra_na)
        return x.where(~x.isin(list(na_codes)), np.nan)

    def standardize(series):
        s = pd.to_numeric(series, errors="coerce")
        mu = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            return s * np.nan
        return (s - mu) / sd

    def standardized_betas_from_unstandardized(y, X_no_const, params):
        """
        beta_j = b_j * sd(x_j) / sd(y)
        """
        sdy = sd_sample(y)
        betas = {}
        for c in X_no_const.columns:
            b = float(params.get(c, np.nan))
            sdx = sd_sample(X_no_const[c])
            if pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (sdx / sdy)
        return betas

    def fit_model(df, dv, xcols, model_name, label_map):
        needed = [dv] + xcols
        frame = df[needed].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors (within model frame) to avoid singularities
        kept = []
        dropped = []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(frame) == 0 or len(kept) == 0:
            # Empty model
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": label_map.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return {
                "model": model_name,
                "n": int(len(frame)),
                "r2": np.nan,
                "adj_r2": np.nan,
                "dropped_predictors": dropped,
                "table": tab,
                "frame": frame,
            }

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas_from_unstandardized(y, X, res.params)

        rows = [{
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # no stars on constant (paper style)
        }]

        for c in xcols:
            lab = label_map.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": lab,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": star(p)
                })
            else:
                rows.append({"term": lab, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        return {
            "model": model_name,
            "n": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "dropped_predictors": dropped,
            "table": tab,
            "frame": frame,
            "res": res
        }

    def to_table1_style(tab):
        """
        Display like the paper: Constant as unstandardized b,
        all predictors as standardized beta with stars.
        """
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                v = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                v = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append((r["term"], v))
        return pd.DataFrame(out, columns=["term", "Table1"])

    # ----------------------------
    # Read data and restrict year
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" in df.columns:
        df = df.loc[clean_gss_numeric(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: musical exclusiveness (0-18)
    # Count "dislike" (4) + "dislike very much" (5) across 18 items.
    # Any missing across 18 => DV missing.
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
    # Only 1..5 valid. Everything else missing.
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    dv_series = df["num_genres_disliked"]
    dv_desc = {
        "n_nonmissing": int(dv_series.notna().sum()),
        "mean": float(dv_series.mean(skipna=True)) if dv_series.notna().any() else np.nan,
        "sd": float(dv_series.std(ddof=1, skipna=True)) if dv_series.notna().any() else np.nan,
        "min": float(dv_series.min(skipna=True)) if dv_series.notna().any() else np.nan,
        "max": float(dv_series.max(skipna=True)) if dv_series.notna().any() else np.nan,
    }
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4 ('dislike') or 5 ('dislike very much').\n"
        "Responses outside 1..5 treated as missing; if any of 18 items missing => DV missing.\n\n"
        + "\n".join([f"{k}: {v}" for k, v in dv_desc.items()])
        + "\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # Education
    df["educ_yrs"] = clean_gss_numeric(df.get("educ", np.nan), extra_na=())

    # Prestige
    df["prestg80_v"] = clean_gss_numeric(df.get("prestg80", np.nan), extra_na=())

    # Income per capita: REALINC / HOMPOP (HOMPOP must be > 0)
    df["realinc_v"] = clean_gss_numeric(df.get("realinc", np.nan), extra_na=())
    df["hompop_v"] = clean_gss_numeric(df.get("hompop", np.nan), extra_na=())
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Female dummy from SEX: 1=male, 2=female
    sex = clean_gss_numeric(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss_numeric(df.get("age", np.nan), extra_na=())
    # keep as-is; if age has unusual codes, they become NA above via common codes

    # Race dummies from RACE: 1=white, 2=black, 3=other
    race = clean_gss_numeric(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: use ETHNIC as provided in this extract.
    # Common in some GSS extracts: ETHNIC 1=Not Hispanic, 2=Hispanic.
    # If ETHNIC is not binary, treat it as missing (to avoid inventing categories).
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss_numeric(df["ethnic"], extra_na=())
        eth_u = set(pd.unique(eth.dropna()))
        if eth_u.issubset({1.0, 2.0}):
            # Assume 2 = Hispanic
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # If not a clean binary flag, leave missing rather than miscode.
            df["hispanic"] = np.nan

    # Religion: RELIG (4 = none)
    relig = clean_gss_numeric(df.get("relig", np.nan), extra_na=())
    # Keep only plausible categories if present; otherwise allow numeric
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: proxy using RELIG==1 (Protestant) and DENOM in {1,6}
    denom = clean_gss_numeric(df.get("denom", np.nan), extra_na=())
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # Baptist, Other Protestant (proxy)
    cons = (is_prot & denom_cons)
    df["cons_prot"] = np.where(relig.isna(), np.nan, cons.astype(float))
    # If Protestant with missing denom, set 0 to keep non-missing (avoid shrinking sample)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION == 3
    region = clean_gss_numeric(df.get("region", np.nan), extra_na=())
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant indicators
    tol_map = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    missing_tol = [v for v, _ in tol_map if v not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance items: {missing_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for v, intolerant_codes in tol_map:
        x = clean_gss_numeric(df[v], extra_na=())
        # Keep only substantive codes (varies by item); treat anything else as missing.
        # Speech items: {1,2}; College items: vary; Library items: {1,2}
        # We'll accept a broad set {1..5} and NA codes handled above.
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        tol_df[v] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan
    # Ensure within 0..15 where nonmissing
    df.loc[df["pol_intol"].notna() & ((df["pol_intol"] < 0) | (df["pol_intol"] > 15)), "pol_intol"] = np.nan

    # ----------------------------
    # Missingness audit (key analysis vars)
    # ----------------------------
    key_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in key_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss = int(df[v].isna().sum())
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) > 0 else np.nan,
            "unique_nonmissing": int(df[v].nunique(dropna=True))
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness_audit.txt", missingness.to_string(index=False) + "\n")

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

    m1_x = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2_x = m1_x + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    # IMPORTANT: do not pre-drop rows globally; fit_model does per-model listwise deletion.
    m1 = fit_model(df, "num_genres_disliked", m1_x, "Model 1 (SES)", label_map)
    m2 = fit_model(df, "num_genres_disliked", m2_x, "Model 2 (Demographic)", label_map)
    m3 = fit_model(df, "num_genres_disliked", m3_x, "Model 3 (Political intolerance)", label_map)

    fit_stats = pd.DataFrame([
        {"model": m1["model"], "n": m1["n"], "r2": m1["r2"], "adj_r2": m1["adj_r2"]},
        {"model": m2["model"], "n": m2["n"], "r2": m2["r2"], "adj_r2": m2["adj_r2"]},
        {"model": m3["model"], "n": m3["n"], "r2": m3["r2"], "adj_r2": m3["adj_r2"]},
    ])

    # Save paper-style tables (Constant=b; others=beta+stars)
    t1_m1 = to_table1_style(m1["table"])
    t1_m2 = to_table1_style(m2["table"])
    t1_m3 = to_table1_style(m3["table"])

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/table1_model1_ses.txt", t1_m1.to_string(index=False) + "\n")
    write_text("./output/table1_model2_demographic.txt", t1_m2.to_string(index=False) + "\n")
    write_text("./output/table1_model3_political_intolerance.txt", t1_m3.to_string(index=False) + "\n")

    # Also save full coefficient tables (b, beta, p, stars) for debugging
    write_text("./output/table1_model1_full.txt", m1["table"].to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", m2["table"].to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", m3["table"].to_string(index=False) + "\n")

    # Human-readable summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from raw data)\n")
    summary_lines.append("Fit statistics:\n" + fit_stats.to_string(index=False) + "\n")
    summary_lines.append("\nModel 1 (SES) Table 1-style:\n" + t1_m1.to_string(index=False) + "\n")
    summary_lines.append("\nModel 2 (Demographic) Table 1-style:\n" + t1_m2.to_string(index=False) + "\n")
    summary_lines.append("\nModel 3 (Political intolerance) Table 1-style:\n" + t1_m3.to_string(index=False) + "\n")
    write_text("./output/table1_summary.txt", "\n".join(summary_lines))

    return {
        "fit_stats": fit_stats,
        "Model 1 (SES)": m1["table"],
        "Model 2 (Demographic)": m2["table"],
        "Model 3 (Political intolerance)": m3["table"],
        "Model 1 (SES) Table1-style": t1_m1,
        "Model 2 (Demographic) Table1-style": t1_m2,
        "Model 3 (Political intolerance) Table1-style": t1_m3,
        "missingness": missingness,
    }