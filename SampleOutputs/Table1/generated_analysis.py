def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    df = pd.read_csv(data_source)

    # --- Helpers ---
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def zscore(s):
        s = s.astype(float)
        m = s.mean()
        sd = s.std(ddof=0)
        return (s - m) / sd if sd and np.isfinite(sd) and sd > 0 else s * np.nan

    def ols_std_beta(data, y, xvars):
        d = data[[y] + xvars].dropna()
        y_std = zscore(d[y])
        X_std = pd.concat([zscore(d[x]) for x in xvars], axis=1)
        X_std.columns = xvars
        X = sm.add_constant(X_std, has_constant="add")
        model = sm.OLS(y_std, X).fit()

        # Standardized coefficients for predictors are coefficients on z-scored predictors
        betas = model.params.drop("const")
        ses = model.bse.drop("const")
        tvals = model.tvalues.drop("const")
        pvals = model.pvalues.drop("const")

        out = pd.DataFrame(
            {
                "beta_std": betas,
                "se": ses,
                "t": tvals,
                "p": pvals,
            }
        )
        out.index.name = "term"

        info = {
            "N": int(model.nobs),
            "R2": float(model.rsquared),
            "Adj_R2": float(model.rsquared_adj),
            "F": float(model.fvalue) if model.fvalue is not None else np.nan,
            "F_p": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
            "const_unstd_y": float(sm.OLS(d[y], sm.add_constant(d[xvars], has_constant="add")).fit().params["const"]),
        }
        return out, info, model

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    # --- Construct DV: musical exclusiveness (count of disliked across 18 genres) ---
    music_vars = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl", "folk", "gospel",
        "jazz", "latin", "moodeasy", "newage", "opera", "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    for v in music_vars:
        df[v] = to_num(df.get(v))

    # disliked = 1 if 4 or 5; 0 if 1,2,3; else missing
    disliked = pd.DataFrame(index=df.index)
    for v in music_vars:
        x = df[v]
        disliked[v] = np.where(x.isin([1, 2, 3]), 0, np.where(x.isin([4, 5]), 1, np.nan))

    # listwise on all 18 for DV
    df["music_exclusive"] = disliked.sum(axis=1, min_count=len(music_vars))
    df.loc[disliked.isna().any(axis=1), "music_exclusive"] = np.nan

    # --- SES predictors ---
    df["educ"] = to_num(df.get("educ"))
    df["prestg80"] = to_num(df.get("prestg80"))
    df["realinc"] = to_num(df.get("realinc"))
    df["hompop"] = to_num(df.get("hompop"))
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df.loc[(df["hompop"] <= 0) | (~np.isfinite(df["inc_pc"])), "inc_pc"] = np.nan

    # --- Demographics / group identity ---
    df["sex"] = to_num(df.get("sex"))
    df["female"] = np.where(df["sex"] == 2, 1, np.where(df["sex"] == 1, 0, np.nan))

    df["age"] = to_num(df.get("age"))

    df["race"] = to_num(df.get("race"))
    df["black"] = np.where(df["race"] == 2, 1, np.where(df["race"].isin([1, 3]), 0, np.nan))
    df["otherrace"] = np.where(df["race"] == 3, 1, np.where(df["race"].isin([1, 2]), 0, np.nan))

    # Hispanic indicator: use 'ethnic' if present (best available in provided variables)
    df["ethnic"] = to_num(df.get("ethnic"))
    # Heuristic: treat explicit codes 20-29 as Hispanic/Latino; otherwise 0 for non-missing.
    df["hispanic"] = np.where(df["ethnic"].between(20, 29), 1, np.where(df["ethnic"].notna(), 0, np.nan))

    df["relig"] = to_num(df.get("relig"))
    df["denom"] = to_num(df.get("denom"))
    df["norelig"] = np.where(df["relig"] == 4, 1, np.where(df["relig"].notna(), 0, np.nan))

    # Conservative Protestant proxy: Protestant (relig==1) and denom in (Baptist/other/no denom)
    # (minimal, documentation-compatible approximation; adjust mapping if you have a preferred scheme)
    df["cons_prot"] = np.nan
    is_prot = df["relig"] == 1
    denom_cons = df["denom"].isin([1, 6, 7])  # Baptist, Other, No denom (common "conservative" buckets)
    df.loc[df["relig"].notna() & df["denom"].notna(), "cons_prot"] = np.where(is_prot & denom_cons, 1, 0)

    df["region"] = to_num(df.get("region"))
    df["south"] = np.where(df["region"] == 3, 1, np.where(df["region"].notna(), 0, np.nan))

    # --- Political intolerance scale (0-15): count of intolerant responses ---
    tol_items = {
        "spkath": lambda s: np.where(s == 2, 1, np.where(s.notna(), 0, np.nan)),
        "colath": lambda s: np.where(s == 5, 1, np.where(s.notna(), 0, np.nan)),
        "libath": lambda s: np.where(s == 1, 1, np.where(s.notna(), 0, np.nan)),
        "spkrac": lambda s: np.where(s == 2, 1, np.where(s.notna(), 0, np.nan)),
        "colrac": lambda s: np.where(s == 5, 1, np.where(s.notna(), 0, np.nan)),
        "librac": lambda s: np.where(s == 1, 1, np.where(s.notna(), 0, np.nan)),
        "spkcom": lambda s: np.where(s == 2, 1, np.where(s.notna(), 0, np.nan)),
        "colcom": lambda s: np.where(s == 4, 1, np.where(s.notna(), 0, np.nan)),
        "libcom": lambda s: np.where(s == 1, 1, np.where(s.notna(), 0, np.nan)),
        "spkmil": lambda s: np.where(s == 2, 1, np.where(s.notna(), 0, np.nan)),
        "colmil": lambda s: np.where(s == 5, 1, np.where(s.notna(), 0, np.nan)),
        "libmil": lambda s: np.where(s == 1, 1, np.where(s.notna(), 0, np.nan)),
        "spkhomo": lambda s: np.where(s == 2, 1, np.where(s.notna(), 0, np.nan)),
        "colhomo": lambda s: np.where(s == 5, 1, np.where(s.notna(), 0, np.nan)),
        "libhomo": lambda s: np.where(s == 1, 1, np.where(s.notna(), 0, np.nan)),
    }

    tol_df = pd.DataFrame(index=df.index)
    for k, fn in tol_items.items():
        s = to_num(df.get(k))
        tol_df[k] = fn(s)

    df["pol_intol"] = tol_df.sum(axis=1, min_count=len(tol_items))
    # listwise for scale (battery not asked -> missing -> drop in model 3 via listwise on predictors)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # --- Restrict to year 1993 (as specified) ---
    df["year"] = to_num(df.get("year"))
    df93 = df.loc[df["year"] == 1993].copy()

    # --- Models (standardized OLS coefficients) ---
    y = "music_exclusive"

    m1_x = ["educ", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    results = {}
    infos = {}

    for name, xvars in [("model1_ses", m1_x), ("model2_demo", m2_x), ("model3_polintol", m3_x)]:
        tab, info, model = ols_std_beta(df93, y, xvars)
        tab["sig"] = tab["p"].map(stars)
        results[name] = tab
        infos[name] = info

        # Save a human-readable text summary
        with open(f"./output/{name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"{name}\n")
            f.write(f"DV: {y}\n")
            f.write(f"N={info['N']}, R2={info['R2']:.4f}, Adj_R2={info['Adj_R2']:.4f}\n")
            f.write(f"Unstandardized constant (from unstd OLS): {info['const_unstd_y']:.6f}\n\n")
            f.write("Standardized coefficients (beta):\n")
            f.write(tab[["beta_std", "se", "t", "p", "sig"]].to_string(float_format=lambda x: f"{x: .6f}"))
            f.write("\n\n")
            f.write(model.summary().as_text())

        # Save a compact "table" view
        tab_out = tab[["beta_std", "se", "t", "p", "sig"]].copy()
        tab_out.to_csv(f"./output/{name}_table.csv")

    # DV descriptives
    dv = df93[y].dropna()
    dv_desc = pd.DataFrame(
        {
            "N": [dv.shape[0]],
            "mean": [dv.mean()],
            "sd": [dv.std(ddof=1)],
            "min": [dv.min()],
            "max": [dv.max()],
        }
    )
    dv_desc.to_csv("./output/dv_music_exclusive_descriptives.csv", index=False)
    with open("./output/dv_music_exclusive_descriptives.txt", "w", encoding="utf-8") as f:
        f.write(dv_desc.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    # Combined table for convenience
    combined = pd.concat(
        {k: v[["beta_std", "se", "t", "p", "sig"]] for k, v in results.items()},
        axis=1
    )
    combined.to_csv("./output/table1_like_combined.csv")
    with open("./output/table1_like_combined.txt", "w", encoding="utf-8") as f:
        f.write(combined.to_string(float_format=lambda x: f"{x: .6f}"))
        f.write("\n")

    return {"tables": results, "model_info": pd.DataFrame(infos).T, "dv_desc": dv_desc}