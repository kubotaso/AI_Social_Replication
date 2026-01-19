def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS missing/NA codes (covers many integer-coded items)
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

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
        # beta_j = b_j * SD(x_j) / SD(y), computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def ensure_cols(df, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def fit_model(df, dv, xcols, model_name, labels):
        # Model-specific listwise deletion ONLY on dv + xcols
        frame = df[[dv] + xcols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors (but also report as a replication failure)
        kept, dropped = [], []
        for c in xcols:
            nun = frame[c].nunique(dropna=True)
            if nun <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped_predictors": ",".join(dropped) if dropped else ""
        }

        rows = []
        if len(frame) == 0 or len(kept) == 0:
            # empty model shell
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        })

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

        return meta, pd.DataFrame(rows), frame

    def table1_display(tab):
        # Constant: unstandardized b; predictors: standardized beta + stars
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def audit_dummies(frame, spec):
        # spec: list of (col, expected_set, min_prop, max_prop) where expected_set is {0,1}
        rows = []
        for col, expected_set, minp, maxp in spec:
            s = frame[col]
            vals = set(pd.unique(s.dropna()))
            ok_vals = vals.issubset(expected_set)
            prop1 = float((s == 1).mean()) if len(s) else np.nan
            rows.append({
                "var": col,
                "n": int(len(s)),
                "n_nonmiss": int(s.notna().sum()),
                "unique_vals": ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in sorted(vals)) if vals else "",
                "ok_0_1": bool(ok_vals),
                "prop_1": prop1,
                "prop_1_in_[min,max]": (False if pd.isna(prop1) else (prop1 >= minp and prop1 <= maxp)),
                "min_expected": minp,
                "max_expected": maxp
            })
        return pd.DataFrame(rows)

    # ----------------------------
    # Read + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # DV: number of music genres disliked (0–18)
    # rule: count of 18 items where response 4 or 5; any missing on 18 => DV missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    ensure_cols(df, music_items)

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

    write_text(
        "./output/dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: sum over 18 genre items of I(response in {4,5}); DK/NA treated missing; any missing genre item => DV missing.\n\n"
        + df["num_genres_disliked"].describe().to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Female
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male, 2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race + Hispanic (mutually exclusive categories to match "White ref" with 3 dummies)
    # Use race (1 white, 2 black, 3 other) and ethnic as Hispanic origin.
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)

    eth = clean_gss(df.get("ethnic", np.nan))
    # Try a conservative binary interpretation: if only {1,2} appear, treat 2 as Hispanic.
    # Otherwise: treat code==1 as not Hispanic and any code >=2 as Hispanic-origin.
    hisp = pd.Series(np.nan, index=df.index, dtype=float)
    if "ethnic" in df.columns:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            hisp = np.where(eth.isna(), np.nan, ((eth >= 2) & (eth <= 99)).astype(float))
        hisp = pd.Series(hisp, index=df.index, dtype=float)

    # Construct mutually exclusive dummies:
    # Hispanic dummy takes precedence when hisp==1 (regardless of race).
    # Black dummy only for non-Hispanic blacks (race==2 & hisp==0).
    # Other race dummy only for non-Hispanic other (race==3 & hisp==0).
    # White non-Hispanic (race==1 & hisp==0) is reference.
    df["hispanic"] = hisp

    df["black"] = np.nan
    df["otherrace"] = np.nan

    # Only define race dummies where both race and hispanic are known (avoids accidental imputation)
    known = race.notna() & df["hispanic"].notna()
    df.loc[known, "black"] = ((race.loc[known] == 2) & (df.loc[known, "hispanic"] == 0)).astype(float)
    df.loc[known, "otherrace"] = ((race.loc[known] == 3) & (df.loc[known, "hispanic"] == 0)).astype(float)

    # If Hispanic==1 and known, force black/otherrace to 0
    df.loc[known & (df["hispanic"] == 1), "black"] = 0.0
    df.loc[known & (df["hispanic"] == 1), "otherrace"] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant approximation (documentation-supported): Protestant AND denomination in {Baptist, Other Protestant}
    # Keep denom missing among Protestants as 0 to avoid unnecessary case loss
    is_prot = (relig == 1)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin([1, 6])).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern (mapping instruction: REGION==3 => South)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15)
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    ensure_cols(df, [c for c, _ in tol_items])

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # Keep plausible response codes; others missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    write_text(
        "./output/pol_intol_descriptives.txt",
        "Political intolerance scale (0–15)\n"
        "Construction: sum of 15 intolerant indicators (5 target groups × 3 contexts); any missing item => scale missing.\n\n"
        + df["pol_intol"].describe().to_string()
        + "\n"
    )

    # ----------------------------
    # Missingness diagnostics (overall, not per model)
    # ----------------------------
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
    write_text("./output/missingness_overall.txt", missingness.to_string(index=False) + "\n")

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
        "pol_intol": "Political intolerance (0–15)",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Replication checks (fail-fast-ish; record issues to file)
    # ----------------------------
    expected_n = {
        "Model 1 (SES)": 787,
        "Model 2 (Demographic)": 756,
        "Model 3 (Political intolerance)": 503,
    }

    checks = []

    def add_check(model, check, ok, detail=""):
        checks.append({"model": model, "check": check, "ok": bool(ok), "detail": str(detail)})

    # n checks (do not hard fail; record)
    for meta in [meta1, meta2, meta3]:
        exp = expected_n.get(meta["model"])
        if exp is not None:
            add_check(meta["model"], "n_matches_paper", meta["n"] == exp, f"n={meta['n']} expected={exp}")

    # predictors present
    for model_name, xcols, frame in [
        (meta1["model"], m1, frame1),
        (meta2["model"], m2, frame2),
        (meta3["model"], m3, frame3),
    ]:
        add_check(model_name, "all_predictors_present_in_frame", all(c in frame.columns for c in xcols), "")

    # dropped predictors (should be none)
    for meta in [meta1, meta2, meta3]:
        add_check(meta["model"], "no_dropped_predictors", meta["dropped_predictors"] == "", meta["dropped_predictors"])

    # dummy coding sanity + proportions (very broad bounds; meant to catch broken coding)
    dummy_spec = [
        ("female", {0.0, 1.0}, 0.35, 0.65),
        ("black", {0.0, 1.0}, 0.00, 0.40),
        ("hispanic", {0.0, 1.0}, 0.00, 0.40),
        ("otherrace", {0.0, 1.0}, 0.00, 0.40),
        ("cons_prot", {0.0, 1.0}, 0.00, 0.60),
        ("norelig", {0.0, 1.0}, 0.00, 0.60),
        ("south", {0.0, 1.0}, 0.00, 0.60),
    ]
    for model_name, frame in [(meta1["model"], frame1), (meta2["model"], frame2), (meta3["model"], frame3)]:
        cols_in = [c for c, _, _, _ in dummy_spec if c in frame.columns]
        if cols_in:
            aud = audit_dummies(frame, [s for s in dummy_spec if s[0] in cols_in])
            # flag any obvious problems
            for _, r in aud.iterrows():
                add_check(model_name, f"dummy_{r['var']}_ok_0_1", r["ok_0_1"], f"unique={r['unique_vals']}")
            # also save model-level audit table
            write_text(f"./output/audit_dummies_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.txt",
                       aud.to_string(index=False) + "\n")

    # race reference-category check: mutually exclusive indicators + implied White non-Hispanic reference
    # In estimation frames for models 2/3, ensure no case has black==1 and hispanic==1 etc.
    def race_exclusivity_report(frame):
        s = frame[["black", "hispanic", "otherrace"]].copy()
        # all 0/1 already (if not, will show as NA)
        any_overlap = (
            ((s["black"] == 1) & (s["hispanic"] == 1))
            | ((s["black"] == 1) & (s["otherrace"] == 1))
            | ((s["hispanic"] == 1) & (s["otherrace"] == 1))
        )
        all_zero = (s.sum(axis=1) == 0)
        return {
            "n": int(len(s)),
            "n_overlap": int(any_overlap.sum()),
            "n_all_zero_reference": int(all_zero.sum()),
            "prop_reference": float(all_zero.mean()) if len(s) else np.nan
        }

    for model_name, frame in [(meta2["model"], frame2), (meta3["model"], frame3)]:
        if set(["black", "hispanic", "otherrace"]).issubset(frame.columns):
            rep = race_exclusivity_report(frame)
            add_check(model_name, "race_dummies_mutually_exclusive", rep["n_overlap"] == 0, rep)
            write_text(
                f"./output/audit_race_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.txt",
                pd.Series(rep).to_string() + "\n"
            )

    checks_df = pd.DataFrame(checks)
    write_text("./output/replication_checks.txt", checks_df.to_string(index=False) + "\n")

    # ----------------------------
    # Save regression tables
    # ----------------------------
    def save_model_outputs(meta, tab, model_id):
        write_text(f"./output/{model_id}_fit.txt", pd.Series(meta).to_string() + "\n")
        write_text(f"./output/{model_id}_full.txt", tab.to_string(index=False) + "\n")
        write_text(f"./output/{model_id}_table1style.txt", table1_display(tab).to_string(index=False) + "\n")

    save_model_outputs(meta1, tab1, "model1")
    save_model_outputs(meta2, tab2, "model2")
    save_model_outputs(meta3, tab3, "model3")

    write_text("./output/fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    # ----------------------------
    # Combine into a Table-1-like wide table (union of terms across models)
    # ----------------------------
    t1 = table1_display(tab1).rename(columns={"Table1": "Model 1 (SES)"})
    t2 = table1_display(tab2).rename(columns={"Table1": "Model 2 (Demographic)"})
    t3 = table1_display(tab3).rename(columns={"Table1": "Model 3 (Political intolerance)"})

    # preserve Table 1 ordering: constant, SES, demographics, pol_intol
    term_order = [
        "Constant",
        "Education (years)",
        "Household income per capita",
        "Occupational prestige",
        "Female",
        "Age",
        "Black",
        "Hispanic",
        "Other race",
        "Conservative Protestant",
        "No religion",
        "Southern",
        "Political intolerance (0–15)",
    ]
    base = pd.DataFrame({"term": term_order})
    wide = base.merge(t1, on="term", how="left").merge(t2, on="term", how="left").merge(t3, on="term", how="left")
    write_text("./output/table1_like_wide.txt", wide.to_string(index=False) + "\n")

    # ----------------------------
    # Return results
    # ----------------------------
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "table1_like_wide": wide,
        "replication_checks": checks_df,
        "missingness_overall": missingness
    }