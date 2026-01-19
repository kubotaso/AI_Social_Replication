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

    def clean_gss(series, extra_na=()):
        """
        Conservative GSS missing recode:
        - Always treat: 7,8,9,97,98,99,997,998,999 as missing
        - Plus any extra_na passed for a specific variable
        """
        s = to_num(series)
        na_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999}
        na_codes |= set(extra_na)
        return s.where(~s.isin(list(na_codes)), np.nan)

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

    def sd_sample(s):
        s = pd.to_numeric(s, errors="coerce")
        v = s.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        sdy = sd_sample(y)
        out = {}
        for col in X.columns:
            b = float(params.get(col, np.nan))
            sdx = sd_sample(X[col])
            if pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0:
                out[col] = np.nan
            else:
                out[col] = b * (sdx / sdy)
        return out

    def fit_ols_standardized_table(df, dv, xcols, model_name, label_map):
        """
        Fit OLS on unstandardized variables (so constant is comparable),
        compute standardized betas for predictors on estimation sample.
        Returns:
          - coef table (paper-style display column)
          - fit stats row
          - full coef detail (b, beta, p, sig)
          - estimation frame (for debugging)
        """
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors in-frame to avoid singularities
        kept = []
        dropped = []
        for c in xcols:
            nun = frame[c].nunique(dropna=True)
            if nun <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(frame) == 0 or len(kept) == 0:
            empty_rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                empty_rows.append({"term": label_map.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            coef_detail = pd.DataFrame(empty_rows)
            fit_stats = pd.DataFrame([{"model": model_name, "n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan, "dropped": ", ".join(dropped)}])
            paper = coef_detail.copy()
            paper["Table1"] = [""] * len(paper)
            return paper[["term", "Table1"]], fit_stats, coef_detail, frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant to match Table 1 style
        })

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

        coef_detail = pd.DataFrame(rows)

        # Paper-style display: intercept shown as unstandardized constant; predictors shown as standardized beta + stars
        paper = coef_detail.copy()
        disp = []
        for _, r in paper.iterrows():
            if r["term"] == "Constant":
                disp.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                disp.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        paper["Table1"] = disp

        fit_stats = pd.DataFrame([{
            "model": model_name,
            "n": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "dropped": ", ".join(dropped)
        }])

        return paper[["term", "Table1"]], fit_stats, coef_detail, frame

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read data / restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Dataset must include YEAR column.")
    df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: # genres disliked across 18 items
    # Rules:
    # - valid responses: 1..5
    # - disliked: 4 or 5
    # - if any of 18 missing => DV missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: clean_gss(df[c]) for c in music_items})
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float)
    disliked = disliked.where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # ----------------------------
    # Predictors
    # ----------------------------
    # Education years
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan), extra_na=(0,))

    # Occupational prestige
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan), extra_na=(0,))

    # Income per capita: REALINC / HOMPOP (per instruction)
    # Do NOT drop zero/negative income aggressively (could be valid 0 in some extracts),
    # but do require positive household size.
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan), extra_na=(0,))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan), extra_na=(0,))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Female dummy: SEX (1=male, 2=female)
    sex = clean_gss(df.get("sex", np.nan), extra_na=(0,))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan), extra_na=(0,))

    # Race dummies: RACE (1=white, 2=black, 3=other)
    race = clean_gss(df.get("race", np.nan), extra_na=(0,))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy:
    # Use ETHNIC from this extract. Treat as a binary indicator if values are {1,2}.
    # Keep missing only when ETHNIC is missing; otherwise always code 0/1.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"], extra_na=(0,))
        # If binary-coded, assume 1=not Hispanic, 2=Hispanic (common in GSS extracts)
        uniq = sorted(pd.unique(eth.dropna()))
        if set(uniq).issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # If not binary-coded, do not force NA for all; instead keep a low-assumption rule:
            # If ETHNIC is a percent Hispanic origin type coding: treat 1 as "Hispanic" if appears as small minority,
            # else treat as missing to avoid misclassification.
            # Here, we implement a conservative fallback: if ETHNIC has a clear "Hispanic" code 1/2 convention is absent,
            # then code non-missing as 0 (unknown/not usable) and keep missing as NA.
            df["hispanic"] = np.where(eth.isna(), np.nan, 0.0)

    # Religion: No religion dummy RELIG==4
    relig = clean_gss(df.get("relig", np.nan), extra_na=(0,))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: Protestant (RELIG==1) and DENOM in a conservative set.
    # We avoid making denom missing create NA; for non-Protestants, cons_prot=0 if relig known.
    denom = clean_gss(df.get("denom", np.nan), extra_na=(0,))
    is_prot = (relig == 1)
    # Common conservative Protestant signals in coarse denom coding:
    # 1=Baptist, 4=Pentecostal, 6=Other Protestant (often includes fundamentalist/evangelical),
    # 7=No denomination (many evangelical nondenoms)
    denom_cons = denom.isin([1, 4, 6, 7])
    cons = (is_prot & denom_cons)
    df["cons_prot"] = np.where(relig.isna(), np.nan, 0.0)
    df.loc[cons.fillna(False), "cons_prot"] = 1.0

    # Southern: REGION==3
    region = clean_gss(df.get("region", np.nan), extra_na=(0,))
    # Keep as missing if region missing; else 0/1
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum of 15 items coded as "intolerant"
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

    tol_items = []
    for v, intolerant_codes in tol_map:
        x = clean_gss(df[v], extra_na=(0,))
        # Keep only known substantive codes for these question types:
        # SPK*: 1/2; COL*: 4/5; LIB*: 1/2/3 (typically)
        # We'll just treat nonmissing as substantive and apply intolerant coding;
        # values outside expected sets will remain nonmissing and be coded 0 if not intolerant,
        # which is safer for N than coercing to missing.
        ind = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))
        tol_items.append(pd.Series(ind, index=df.index, name=v))

    tol_df = pd.concat(tol_items, axis=1)
    df["pol_intol"] = tol_df.sum(axis=1)
    # Require complete on all 15 items for the scale (paper-style listwise for battery)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Diagnostics: missingness + DV descriptives
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
        nm = int(df[v].notna().sum())
        m = int(df[v].isna().sum())
        miss_rows.append({"variable": v, "nonmissing": nm, "missing": m, "pct_missing": (100.0 * m / max(1, nm + m))})
    missingness = pd.DataFrame(miss_rows).sort_values(["pct_missing", "variable"], ascending=[False, True])

    dv_desc = df["num_genres_disliked"].describe()

    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4 ('dislike') or 5 ('dislike very much');\n"
        "responses outside 1..5 treated as missing; if any of 18 items missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

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

    dv = "num_genres_disliked"

    m1_x = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2_x = m1_x + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    m1_paper, m1_fit, m1_detail, m1_frame = fit_ols_standardized_table(df, dv, m1_x, "Model 1 (SES)", label_map)
    m2_paper, m2_fit, m2_detail, m2_frame = fit_ols_standardized_table(df, dv, m2_x, "Model 2 (Demographic)", label_map)
    m3_paper, m3_fit, m3_detail, m3_frame = fit_ols_standardized_table(df, dv, m3_x, "Model 3 (Political intolerance)", label_map)

    fit_stats = pd.concat([m1_fit, m2_fit, m3_fit], ignore_index=True)

    # Save human-readable outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/table1_model1.txt", m1_paper.to_string(index=False) + "\n")
    write_text("./output/table1_model2.txt", m2_paper.to_string(index=False) + "\n")
    write_text("./output/table1_model3.txt", m3_paper.to_string(index=False) + "\n")

    # Also save coefficient details (b, beta, p) for debugging/validation (not paper format)
    write_text("./output/table1_model1_details.txt", m1_detail.to_string(index=False) + "\n")
    write_text("./output/table1_model2_details.txt", m2_detail.to_string(index=False) + "\n")
    write_text("./output/table1_model3_details.txt", m3_detail.to_string(index=False) + "\n")

    # Save sample sizes and quick checks on key predictors in each model frame
    def frame_quickcheck(frame, cols, name):
        lines = []
        lines.append(f"{name} estimation sample: n={len(frame)}")
        for c in cols:
            if c in frame.columns:
                vc = frame[c].value_counts(dropna=False)
                # keep it short for continuous variables; show mean/sd
                if frame[c].nunique(dropna=True) > 10:
                    lines.append(f"  {c}: mean={frame[c].mean():.4f}, sd={frame[c].std(ddof=1):.4f}, missing={frame[c].isna().sum()}")
                else:
                    lines.append(f"  {c} value_counts:\n{vc.to_string()}")
        return "\n".join(lines) + "\n"

    qc = []
    qc.append(frame_quickcheck(m1_frame, [dv] + m1_x, "Model 1"))
    qc.append(frame_quickcheck(m2_frame, [dv] + m2_x, "Model 2"))
    qc.append(frame_quickcheck(m3_frame, [dv] + m3_x, "Model 3"))
    write_text("./output/table1_sample_quickchecks.txt", "\n".join(qc))

    # Return results as dict of DataFrames
    return {
        "fit_stats": fit_stats,
        "model1_table": m1_paper,
        "model2_table": m2_paper,
        "model3_table": m3_paper,
        "model1_details": m1_detail,
        "model2_details": m2_detail,
        "model3_details": m3_detail,
        "missingness": missingness,
    }