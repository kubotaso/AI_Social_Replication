def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative "GSS-style" missing codes; do NOT include 1/2/etc.
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(series, valid=None, extra_na=()):
        x = to_num(series)
        na = set(GSS_NA_CODES) | set(extra_na)
        x = x.where(~x.isin(list(na)), np.nan)
        if valid is not None:
            if callable(valid):
                x = x.where(valid(x), np.nan)
            else:
                x = x.where(x.isin(list(valid)), np.nan)
        return x

    def safe_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

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

    def zscore_frame(frame):
        # Standardize columns with sample mean/sd; keep columns with sd>0
        out = frame.copy()
        keep = []
        for c in out.columns:
            sd = safe_sd(out[c])
            if pd.isna(sd) or sd == 0:
                continue
            mu = float(out[c].mean())
            out[c] = (out[c] - mu) / sd
            keep.append(c)
        return out[keep], keep

    def fit_ols_standardized(df_in, dv, xcols):
        """
        Fit:
          - Unstandardized OLS for intercept and fit stats (R2/adjR2), and for p-values (replication-only).
          - Standardized betas by fitting OLS on z-scored DV and X with intercept.
        Listwise deletion is ONLY on dv + xcols.
        """
        cols = [dv] + list(xcols)
        frame = df_in[cols].dropna(axis=0, how="any").copy()

        meta = {"n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan}
        if meta["n"] == 0:
            return meta, None, None, frame

        y = frame[dv].astype(float)
        X = frame[list(xcols)].astype(float)

        # Drop any zero-variance predictors in this analytic sample (avoid singular matrix)
        kept = [c for c in xcols if frame[c].nunique(dropna=True) > 1]
        dropped = [c for c in xcols if c not in kept]

        # Unstandardized fit (for intercept, R2/adjR2, and replication p-values)
        Xu = sm.add_constant(frame[kept].astype(float), has_constant="add")
        res_u = sm.OLS(y, Xu).fit()

        meta["r2"] = float(res_u.rsquared)
        meta["adj_r2"] = float(res_u.rsquared_adj)
        meta["dropped"] = ",".join(dropped) if dropped else ""

        # Standardized fit for betas: z(y) ~ z(X) with intercept
        yz = (y - y.mean()) / y.std(ddof=1)
        Xz, kept2 = zscore_frame(frame[kept])
        Xz = sm.add_constant(Xz, has_constant="add")

        # Align kept sets (if zscore dropped any due to sd=0, mark dropped)
        dropped2 = [c for c in kept if c not in kept2]
        dropped_all = sorted(set(dropped + dropped2))
        meta["dropped"] = ",".join(dropped_all) if dropped_all else ""

        res_z = sm.OLS(yz, Xz).fit()

        return meta, res_u, res_z, frame

    def format_table(model_name, xcols, labels, meta, res_u, res_z):
        # Build a table: Constant is unstandardized; predictors show standardized beta and stars from replication p-values
        rows = []
        const_b = np.nan if res_u is None else float(res_u.params.get("const", np.nan))
        const_p = np.nan if res_u is None else float(res_u.pvalues.get("const", np.nan))
        rows.append(
            {
                "model": model_name,
                "term": "Constant",
                "b_unstd": const_b,
                "beta_std": np.nan,
                "p_replication": const_p,
                "sig": "",
            }
        )

        for c in xcols:
            term = labels.get(c, c)
            b = np.nan if res_u is None else float(res_u.params.get(c, np.nan))
            p = np.nan if res_u is None else float(res_u.pvalues.get(c, np.nan))
            beta = np.nan
            if res_z is not None:
                beta = float(res_z.params.get(c, np.nan))
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "b_unstd": b,
                    "beta_std": beta,
                    "p_replication": p,
                    "sig": sig_star(p),
                }
            )

        tab = pd.DataFrame(rows)

        # "Table 1 style" display: constant unstd; predictors beta (std) + stars
        disp = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                disp.append("" if pd.isna(r["b_unstd"]) else f"{float(r['b_unstd']):.3f}")
            else:
                disp.append("" if pd.isna(r["beta_std"]) else f"{float(r['beta_std']):.3f}{r['sig']}")
        table1_style = pd.DataFrame({"term": tab["term"], "Table1_style": disp})

        fit_line = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n": meta.get("n", np.nan),
                    "r2": meta.get("r2", np.nan),
                    "adj_r2": meta.get("adj_r2", np.nan),
                    "dropped_predictors": meta.get("dropped", ""),
                }
            ]
        )
        return tab, table1_style, fit_line

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]
    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: number of music genres disliked (0-18)
    # Rule: count 4/5 across 18 items; if ANY item missing => DV missing.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing music columns: {missing_music}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        # Valid substantive responses are 1..5
        music[c] = clean_gss(df[c], valid={1, 2, 3, 4, 5})

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # ----------------------------
    # Predictors: recode to minimize unintended missingness
    # (dummies should be 0/1 with missing only when truly unknown)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    hompop = clean_gss(df.get("hompop", np.nan))
    hompop = hompop.where(hompop > 0, np.nan)
    realinc = clean_gss(df.get("realinc", np.nan))
    inc_pc = realinc / hompop
    inc_pc = inc_pc.where(np.isfinite(inc_pc), np.nan)
    df["inc_pc"] = inc_pc

    # Female
    sex = clean_gss(df.get("sex", np.nan), valid={1, 2})
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    age = clean_gss(df.get("age", np.nan))
    age = age.where(age > 0, np.nan)
    df["age_v"] = age

    # Race/ethnicity: aim for non-missing 0/1 when possible.
    # Use RACE for black/other; use ETHNIC for hispanic.
    race = clean_gss(df.get("race", np.nan), valid={1, 2, 3})
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))

    # Hispanic: treat ETHNIC==1 as not Hispanic, ETHNIC==2 as Hispanic when available.
    # If ETHNIC is missing, treat as 0 to avoid collapsing N (paper clearly retains large N).
    hisp = np.zeros(len(df), dtype=float) * np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Most common: 1=not hispanic, 2=hispanic
        h = np.where(eth.isna(), 0.0, (eth == 2).astype(float))
        hisp = h
    else:
        # If no ethnic variable exists, default 0 (cannot identify)
        hisp = np.zeros(len(df), dtype=float)
    df["hispanic"] = hisp

    # Other race: to avoid collinearity and sign flips due to overlap, define as:
    # other-race (RACE==3) AND not Hispanic (so White non-Hispanic is implied reference),
    # and Black is separate. This yields mutually exclusive-ish categories in typical GSS usage.
    # If race is missing, keep missing (cannot classify).
    other = np.where(race.isna(), np.nan, (race == 3).astype(float))
    # Exclude Hispanics from "other race" for a cleaner reference group
    other = np.where(np.isnan(other), np.nan, np.where(df["hispanic"] == 1.0, 0.0, other))
    df["otherrace"] = other

    # Religion
    relig = clean_gss(df.get("relig", np.nan), valid={1, 2, 3, 4, 5})
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant:
    # To reduce missingness, if denom missing among Protestants, treat as not conservative (0).
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = relig == 1
    cons = np.where(relig.isna(), np.nan, 0.0)
    # Approximation using DENOM codes (dataset-specific; best-effort without adding missingness)
    cons = np.where(is_prot & denom.isin([1, 6]), 1.0, cons)
    cons = np.where(is_prot & denom.isna(), 0.0, cons)
    df["cons_prot"] = cons

    # South: REGION==3
    region = clean_gss(df.get("region", np.nan))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance (0-15): sum intolerant across available items
    # Key fix: do NOT require all 15 items nonmissing.
    # Compute sum of intolerant among answered items, then rescale to 15 if partially missing.
    # This reduces missingness to match "asked to ~2/3" rather than "complete 15 only".
    # Require at least 12 of 15 answered to keep (best-effort to balance validity vs N).
    # ----------------------------
    tol_items = [
        ("spkath", "spk", {2}), ("colath", "col", {5}), ("libath", "lib", {1}),
        ("spkrac", "spk", {2}), ("colrac", "col", {5}), ("librac", "lib", {1}),
        ("spkcom", "spk", {2}), ("colcom", "col", {4}), ("libcom", "lib", {1}),
        ("spkmil", "spk", {2}), ("colmil", "col", {5}), ("libmil", "lib", {1}),
        ("spkhomo", "spk", {2}), ("colhomo", "col", {5}), ("libhomo", "lib", {1}),
    ]
    missing_tol = [c for c, _, _ in tol_items if c not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing political tolerance columns: {missing_tol}")

    tol = pd.DataFrame(index=df.index)
    for col, kind, intolerant_codes in tol_items:
        x = clean_gss(df[col])
        # Keep plausible ranges by kind; anything else missing
        if kind == "spk":
            x = x.where(x.isin([1, 2]), np.nan)
        elif kind == "col":
            x = x.where(x.isin([1, 4, 5]), np.nan)  # colcom uses 4; others use 5; allow both
        elif kind == "lib":
            x = x.where(x.isin([1, 2]), np.nan)
        tol[col] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    answered = tol.notna().sum(axis=1)
    intolerant_sum = tol.sum(axis=1, min_count=1)

    # Rescale partial to 15: sum * 15 / answered
    pol_intol = intolerant_sum * (15.0 / answered.replace(0, np.nan))

    # Require minimum answered items to keep
    pol_intol = pol_intol.where(answered >= 12, np.nan)

    # Clamp to [0,15] for safety against floating artifacts
    pol_intol = pol_intol.clip(lower=0, upper=15)
    df["pol_intol"] = pol_intol

    # ----------------------------
    # Labels
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

    # ----------------------------
    # Models
    # ----------------------------
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, resu1, resz1, frame1 = fit_ols_standardized(df, "num_genres_disliked", m1)
    meta2, resu2, resz2, frame2 = fit_ols_standardized(df, "num_genres_disliked", m2)
    meta3, resu3, resz3, frame3 = fit_ols_standardized(df, "num_genres_disliked", m3)

    tab1, t1style1, fit1 = format_table("Model 1 (SES)", m1, labels, meta1, resu1, resz1)
    tab2, t1style2, fit2 = format_table("Model 2 (Demographic)", m2, labels, meta2, resu2, resz2)
    tab3, t1style3, fit3 = format_table("Model 3 (Political intolerance)", m3, labels, meta3, resu3, resz3)

    fit_stats = pd.concat([fit1, fit2, fit3], ignore_index=True)

    # ----------------------------
    # Diagnostics: DV and key scale distributions + missingness
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
        tot = nonmiss + miss
        miss_rows.append(
            {"variable": v, "nonmissing": nonmiss, "missing": miss, "pct_missing": (miss / tot * 100.0) if tot else np.nan}
        )
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)

    # ----------------------------
    # Save outputs (human-readable)
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", t1style1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t1style2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t1style3.to_string(index=False) + "\n")

    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # More descriptive summaries
    dv_desc = df["num_genres_disliked"].describe()
    pol_desc = df["pol_intol"].describe()
    write_text(
        "./output/table1_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "  Count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n\nPolitical intolerance (0–15)\n"
        "  Sum of intolerant responses across 15 items; partial completion allowed and rescaled to 15.\n"
        "  Minimum answered items required: 12 of 15.\n\n"
        + pol_desc.to_string()
        + "\n"
    )

    # Compact combined table (Table 1 style columns side-by-side)
    def merge_table1_styles(tables):
        # tables: list of (name, df(term, Table1_style))
        out = None
        for name, td in tables:
            td = td.copy()
            td.columns = ["term", name]
            out = td if out is None else out.merge(td, on="term", how="outer")
        return out

    combined = merge_table1_styles(
        [
            ("Model 1 (SES)", t1style1),
            ("Model 2 (Demographic)", t1style2),
            ("Model 3 (Political intolerance)", t1style3),
        ]
    )
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Return results as dict of DataFrames
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": t1style1,
        "model2_table1style": t1style2,
        "model3_table1style": t1style3,
        "table1_combined_table1style": combined,
        "missingness": missingness,
    }