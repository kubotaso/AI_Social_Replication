def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter
    import statsmodels.api as sm

    def find_col(df, preferred_names, contains_any=None):
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}

        for name in preferred_names:
            if name is None:
                continue
            key = str(name).strip().lower()
            if key in lower_map:
                return lower_map[key]

        if contains_any:
            tokens = [str(t).strip().lower() for t in contains_any if t is not None and str(t).strip() != ""]
            hits = []
            for c in cols:
                cl = c.lower()
                if any(tok in cl for tok in tokens):
                    hits.append(c)
            if len(hits) == 1:
                return hits[0]

        # unique substring match
        for name in preferred_names:
            if name is None:
                continue
            n = str(name).strip().lower()
            matches = [c for c in cols if n in c.lower()]
            if len(matches) == 1:
                return matches[0]

        return None

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def recode_dislike(series):
        # 1 if 4/5, 0 if 1/2/3, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x >= 1) & (x <= 3)] = 0.0
        out[(x == 4) | (x == 5)] = 1.0
        return out

    def recode_like(series):
        # 1 if 1/2, 0 if 3/4/5, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x == 1) | (x == 2)] = 1.0
        out[(x >= 3) & (x <= 5)] = 0.0
        return out

    def no_leading_zero(x, pos):
        s = f"{x:.1f}"
        return s.replace("-0.", "-.").replace("0.", ".")

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(data_source)

    # YEAR filter
    year_col = find_col(df, ["YEAR", "year"])
    if year_col is not None:
        df[year_col] = to_num(df[year_col])
        df = df.loc[df[year_col] == 1993].copy()

    # EDUC
    educ_col = find_col(df, ["EDUC", "educ", "education", "years_education", "schooling"])
    if educ_col is None:
        raise ValueError("Could not find education column (EDUC/educ).")
    df[educ_col] = to_num(df[educ_col])

    # Drop missing education up-front (used in all models and the sample mean line)
    df = df.dropna(subset=[educ_col]).copy()

    # ----------------------------
    # Exact Figure-1 genre order (as provided)
    # ----------------------------
    genre_order = [
        "Latin/Salsa", "Jazz", "Blues/R&B", "Show Tunes", "Oldies", "Classical",
        "Reggae", "Swing", "New Age/Space", "Opera", "Bluegrass", "Folk",
        "Easy Listening", "Pop/Rock", "Rap", "Heavy Metal", "Country", "Gospel"
    ]

    genre_candidates = {
        "Latin/Salsa": ["LATIN", "latin"],
        "Jazz": ["JAZZ", "jazz"],
        "Blues/R&B": ["BLUES", "blues"],
        "Show Tunes": ["MUSICALS", "musicals", "SHOWTUNES", "showtunes"],
        "Oldies": ["OLDIES", "oldies"],
        "Classical": ["CLASSICL", "classicl", "CLASSICAL", "classical"],
        "Reggae": ["REGGAE", "reggae"],
        "Swing": ["BIGBAND", "bigband", "SWING", "swing"],
        "New Age/Space": ["NEWAGE", "newage", "NEW_AGE", "new_age"],
        "Opera": ["OPERA", "opera"],
        "Bluegrass": ["BLUGRASS", "blugrass", "BLUEGRASS", "bluegrass"],
        "Folk": ["FOLK", "folk"],
        "Easy Listening": ["MOODEASY", "moodeasy", "EASYLIST", "easylist", "EASY", "easy"],
        "Pop/Rock": ["CONROCK", "conrock", "POPROCK", "poprock", "ROCK", "rock"],
        "Rap": ["RAP", "rap", "HIPHOP", "hiphop", "HIP_HOP", "hip_hop"],
        "Heavy Metal": ["HVYMETAL", "hvymetal", "HEAVYMETAL", "heavymetal", "HEAVY_METAL", "heavy_metal"],
        "Country": ["COUNTRY", "country", "COUNTRYWESTERN", "countrywestern", "COUNTRY_WESTERN", "country_western"],
        "Gospel": ["GOSPEL", "gospel"],
    }

    resolved_cols = {}
    for label in genre_order:
        col = find_col(df, genre_candidates.get(label, []))
        if col is None:
            tokens = label.lower().replace("/", " ").replace("&", " ").replace("-", " ").split()
            col = find_col(df, [], contains_any=tokens)
        if col is None:
            raise ValueError(f"Could not find column for genre '{label}'. Available columns: {list(df.columns)}")
        resolved_cols[label] = col

    # ----------------------------
    # Recode dislikes/likes
    # ----------------------------
    dislike = pd.DataFrame(index=df.index)
    like = pd.DataFrame(index=df.index)
    for g in genre_order:
        src = df[resolved_cols[g]]
        dislike[g] = recode_dislike(src)
        like[g] = recode_like(src)

    # Compute sample mean education from the analysis sample (education non-missing)
    sample_mean_edu = float(df[educ_col].mean()) if len(df) else np.nan

    # ----------------------------
    # Fit per-genre logistic models and compute mean edu among likers
    # ----------------------------
    rows = []
    for g in genre_order:
        y = dislike[g]

        others = [h for h in genre_order if h != g]
        others_dislike = dislike[others]

        # Tolerance_-g: number of OTHER genres not disliked
        # Require all 17 other responses to be present; else NaN
        tol_minus_g = (1.0 - others_dislike).sum(axis=1, min_count=len(others))

        model_df = pd.DataFrame(
            {"y": y, "tolerance": tol_minus_g, "educ": df[educ_col]}
        ).dropna()

        coef_tol = np.nan
        pval_tol = np.nan
        n_used = int(len(model_df))

        if n_used > 0 and model_df["y"].nunique() >= 2 and model_df["tolerance"].nunique() >= 2:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=500)
                coef_tol = float(fit.params["tolerance"])
                pval_tol = float(fit.pvalues["tolerance"])
            except Exception:
                # fallback in case of separation; still produces a coefficient (no p-value)
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=5000)
                    coef_tol = float(fit.params["tolerance"])
                    pval_tol = np.nan
                except Exception:
                    coef_tol = np.nan
                    pval_tol = np.nan

        # Mean education among likers (like very much / like it = 1/2)
        aud_df = pd.DataFrame({"like": like[g], "educ": df[educ_col]}).dropna()
        mean_edu = float(aud_df.loc[aud_df["like"] == 1.0, "educ"].mean()) if (len(aud_df) and (aud_df["like"] == 1.0).any()) else np.nan

        rows.append(
            {"genre": g, "coef_tolerance": coef_tol, "pval": pval_tol, "mean_edu": mean_edu, "n": n_used}
        )

    res = pd.DataFrame(rows)

    # Force exact plotting order via ordered categorical, then sort
    res["genre"] = pd.Categorical(res["genre"], categories=genre_order, ordered=True)
    res = res.sort_values("genre").reset_index(drop=True)

    # ----------------------------
    # Plot (matplotlib only)
    # ----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(genre_order), dtype=int)
    coef_vals = res["coef_tolerance"].to_numpy(dtype="float64")
    edu_vals = res["mean_edu"].to_numpy(dtype="float64")

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
    ax2 = ax.twinx()

    # LEFT axis: coefficient (solid)
    ax.plot(x, coef_vals, color="black", lw=1.4, ls="-", zorder=3)

    # RIGHT axis: mean education (dash-dot)
    edu_line = ax2.plot(x, edu_vals, color="black", lw=1.4, ls="-.", zorder=2)[0]
    edu_line.set_dashes([8, 3, 2, 3])  # dash, gap, dot, gap (print-like)

    # Sample mean education reference line (RIGHT axis)
    if np.isfinite(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="black", lw=1.0, ls=":", zorder=1)

    # Axes labels (match requested wording)
    ax.set_xlabel("Type of Music", fontweight="bold")
    ax.set_ylabel(
        "Coefficients for Musical Tolerance as It Affects\n"
        "One’s Probability of Disliking Each Music Genre",
        fontweight="bold"
    )
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\n"
        "Reported Liking Each Music Genre",
        fontweight="bold",
        rotation=270,
        labelpad=28
    )

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(genre_order, rotation=45, ha="right")

    # Left y-axis (coefficients)
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

    # Right y-axis (education)
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # Styling (no gridlines; clean spines)
    ax.grid(False)
    ax2.grid(False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1)
    ax2.tick_params(direction="out", length=6, width=1)

    # ----------------------------
    # Annotations (leader lines point to correct series/axes)
    # ----------------------------
    def idx_of(label, fallback=2):
        try:
            return genre_order.index(label)
        except Exception:
            return min(max(fallback, 0), len(genre_order) - 1)

    def first_finite(arr, start_idx=0, default=2):
        for i in range(start_idx, len(arr)):
            if np.isfinite(arr[i]):
                return i
        for i in range(0, len(arr)):
            if np.isfinite(arr[i]):
                return i
        return min(max(default, 0), len(arr) - 1)

    anchor = idx_of("Blues/R&B", fallback=2)
    i_edu = anchor if np.isfinite(edu_vals[anchor]) else first_finite(edu_vals, default=2)
    i_coef = anchor if np.isfinite(coef_vals[anchor]) else first_finite(coef_vals, default=2)

    # Education callout (targets dash-dot series on right axis)
    ax2.annotate(
        "Mean Education of Genre Audience",
        xy=(x[i_edu], float(edu_vals[i_edu]) if np.isfinite(edu_vals[i_edu]) else 14.0),
        xycoords="data",
        xytext=(x[i_edu] - 1.2, 14.2),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Coefficient callout (targets solid series on left axis)
    ax.annotate(
        "Coefficient for Musical Tolerance",
        xy=(x[i_coef], float(coef_vals[i_coef]) if np.isfinite(coef_vals[i_coef]) else -0.35),
        xycoords="data",
        xytext=(x[i_coef] + 0.6, -0.47),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Sample mean education label with upward arrow (right axis)
    if np.isfinite(sample_mean_edu):
        mean_x = idx_of("Reggae", fallback=6)
        ax2.annotate(
            "Sample Mean Education",
            xy=(x[mean_x], sample_mean_edu),
            xycoords="data",
            xytext=(x[mean_x], sample_mean_edu - 0.30),
            textcoords="data",
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)