def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    GSS_NA_CODES = {
        0, 7, 8, 9, 97, 98, 99,
        997, 998, 999, 9997, 9998, 9999
    }

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clean_gss(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA_CODES) | set(extra_na)
        return x.where(~x.isin(list(na)), np.nan)

    def sig_star(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y), computed on the estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on dv + xcols (per paper)
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors in this analytic sample
        kept, dropped = [], []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ",".join(dropped) if dropped else ""
        }

        # Empty / degenerate cases
        if len(frame) == 0 or len(kept) == 0:
            rows = [{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame, None

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)

        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)
        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows = [{
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        }]

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame, res

    def table1_display(tab):
        # Constant: unstandardized b; Predictors: standardized beta + stars
        out_vals = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out_vals.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out_vals.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out_vals})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def freq_table(s):
        s = pd.Series(s)
        return s.value_counts(dropna=False).sort_index()

    # ----------------------------
    # Read + year restriction (GSS 1993)
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: Number of music genres disliked (0–18)
    # Rule: for each of 18 items, disliked=1 if 4/5 else 0 if 1/2/3; DK/NA -> missing;
    # DV missing if ANY of 18 items missing.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music_cols = [c for c in music_items if c not in df.columns]
    if missing_music_cols:
        raise ValueError(f"Missing required music columns: {missing_music_cols}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Household income per capita: REALINC / HOMPOP (per mapping instruction)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics: Female
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1 male, 2 female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race: Black, Other (White is implicit reference)
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if available.
    # IMPORTANT: code 0/1 for all nonmissing ETHNIC; set missing only when ETHNIC is missing/NA-coded.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Common pattern in GSS extracts: 1=not Hispanic, 2=Hispanic
        if set(pd.unique(eth.dropna())).issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Best-effort fallback: treat 1 as "not Hispanic"; any other positive code as Hispanic-origin
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion: No religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: derived from RELIG and DENOM (approximation; keep simple and auditable)
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # 1 Baptist, 6 Other Protestant (common conservative bucket in coarse coding)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # Avoid dropping Protestants solely due to missing denom: treat missing denom among Protestants as not conservative
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 (per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum of 15 intolerant responses; missing if any item missing
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    missing_tol_cols = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol_cols:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol_cols}")

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Audited mapping (save)
    # ----------------------------
    mapping_lines = [
        "AUDITED VARIABLE MAPPING (constructed from raw columns)\n",
        "DV: num_genres_disliked = sum_{18 genres} I(response in {4,5}); DV missing if any genre missing\n",
        f"  Genres: {', '.join(music_items)}\n",
        "SES:\n",
        "  Education (years) = educ_yrs from EDUC (cleaned NA codes)\n",
        "  Household income per capita = inc_pc = REALINC / HOMPOP (cleaned NA codes; HOMPOP<=0 -> missing)\n",
        "  Occupational prestige = prestg80_v from PRESTG80 (cleaned NA codes)\n",
        "Demographics:\n",
        "  Female = 1 if SEX==2 else 0 if SEX==1\n",
        "  Age = AGE (cleaned NA codes; <=0 -> missing)\n",
        "  Black = 1 if RACE==2 else 0 if RACE in {1,3}\n",
        "  Other race = 1 if RACE==3 else 0 if RACE in {1,2}\n",
        "  Hispanic = derived from ETHNIC: if binary {1,2}, Hispanic=1 if 2 else 0; otherwise Hispanic=1 if ETHNIC!=1\n",
        "Religion:\n",
        "  No religion = 1 if RELIG==4 else 0\n",
        "  Conservative Protestant (approx) = 1 if RELIG==1 and DENOM in {1,6}; Protestant w/ missing DENOM -> 0\n",
        "Region:\n",
        "  Southern = 1 if REGION==3 else 0\n",
        "Political intolerance:\n",
        "  pol_intol = sum of 15 intolerant indicators; missing if any of 15 items missing\n",
        "  Intolerant codes:\n",
        "    SPK*: 2; COLATH/COLRAC/COLMIL/COLHOMO: 5; COLCOM: 4; LIB*: 1\n",
    ]
    write_text("./output/table1_mapping_audit.txt", "".join(mapping_lines))

    # ----------------------------
    # Diagnostics: DV distribution + key frequencies
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # frequency diagnostics for dummies and core categorical inputs
    diag_text = []
    for name, series in [
        ("SEX (raw cleaned)", sex),
        ("RACE (raw cleaned)", race),
        ("ETHNIC (raw cleaned)", clean_gss(df["ethnic"]) if "ethnic" in df.columns else pd.Series([], dtype=float)),
        ("RELIG (raw cleaned)", relig),
        ("DENOM (raw cleaned)", denom),
        ("REGION (raw cleaned)", region),
        ("female (constructed)", df["female"]),
        ("black (constructed)", df["black"]),
        ("hispanic (constructed)", df["hispanic"]),
        ("otherrace (constructed)", df["otherrace"]),
        ("cons_prot (constructed)", df["cons_prot"]),
        ("norelig (constructed)", df["norelig"]),
        ("south (constructed)", df["south"]),
    ]:
        if series is None or (isinstance(series, pd.Series) and series.size == 0):
            continue
        diag_text.append(f"\n{name}\n")
        diag_text.append(freq_table(series).to_string() + "\n")
    write_text("./output/table1_key_frequencies.txt", "".join(diag_text))

    # Missingness summary for analysis variables
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss = int(df[v].isna().sum())
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    labels = {
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
        "pol_intol": "Political intolerance",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1, res1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2, res2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3, res3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table-1-style displays (beta only for predictors; unstd constant)
    t1_1 = table1_display(tab1)
    t1_2 = table1_display(tab2)
    t1_3 = table1_display(tab3)

    # ----------------------------
    # Save human-readable outputs
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", t1_1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1_2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1_3.to_string(index=False) + "\n")

    # Also write a compact combined "Table 1" view for convenience
    combined = t1_1[["term"]].copy()
    combined = combined.merge(t1_1[["term", "Table1"]].rename(columns={"Table1": "Model 1 (SES)"}), on="term", how="left")
    combined = combined.merge(t1_2[["term", "Table1"]].rename(columns={"Table1": "Model 2 (Demographic)"}), on="term", how="left")
    combined = combined.merge(t1_3[["term", "Table1"]].rename(columns={"Table1": "Model 3 (Political intolerance)"}), on="term", how="left")
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Save model sample sizes and variable SDs used for standardization (audit)
    def sd_audit(frame, dv, xcols):
        rows = []
        rows.append({"variable": dv, "sd_in_estimation_sample": sample_sd(frame[dv])})
        for c in xcols:
            rows.append({"variable": c, "sd_in_estimation_sample": sample_sd(frame[c])})
        return pd.DataFrame(rows)

    sd1 = sd_audit(frame1, "num_genres_disliked", m1)
    sd2 = sd_audit(frame2, "num_genres_disliked", m2)
    sd3 = sd_audit(frame3, "num_genres_disliked", m3)
    write_text("./output/model1_sd_audit.txt", sd1.to_string(index=False) + "\n")
    write_text("./output/model2_sd_audit.txt", sd2.to_string(index=False) + "\n")
    write_text("./output/model3_sd_audit.txt", sd3.to_string(index=False) + "\n")

    # Return key results as a dict of DataFrames
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1_1,
        "model2_table1style": t1_2,
        "model3_table1style": t1_3,
        "table1_combined_table1style": combined,
        "missingness": missingness,
        "model1_estimation_frame": frame1,
        "model2_estimation_frame": frame2,
        "model3_estimation_frame": frame3,
    }