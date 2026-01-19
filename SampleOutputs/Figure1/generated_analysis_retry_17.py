def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter
    import statsmodels.api as sm

    # ----------------------------
    # Helpers
    # ----------------------------
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
        # 1 if 1/2 (like/like very much), 0 if 3/4/5, NaN otherwise
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

    # Filter to YEAR == 1993 if present
    year_col = find_col(df, ["YEAR", "year"])
    if year_col is not None:
        df[year_col] = to_num(df[year_col])
        df = df.loc[df[year_col] == 1993].copy()

    # EDUC (years)
    educ_col = find_col(df, ["EDUC", "educ", "education", "years_education", "schooling"])
    if educ_col is None:
        raise ValueError("Could not find education column (EDUC/educ).")
    df[educ_col] = to_num(df[educ_col])

    # Drop missing education up-front (required for all models and sample mean line)
    df = df.dropna(subset=[educ_col]).copy()

    # ----------------------------
    # Canonical Figure-1 order / labels (as printed in Figure 1)
    # ----------------------------
    genre_order = [
        "Latin/Salsa",
        "Jazz",
        "Blues/R&B",
        "Show Tunes",
        "Oldies",
        "Classical/Chamber",
        "Swing/Big Band",
        "New Age/Space",
        "Opera",
        "Bluegrass",
        "Folk",
        "Easy Listening",
        "Pop/Contemporary Rock",
        "Reggae",
        "Rap",
        "Heavy Metal",
        "Country/Western",
        "Gospel",
    ]

    # Resolve columns (case-insensitive + fallbacks)
    genre_candidates = {
        "Latin/Salsa": ["LATIN", "latin"],
        "Jazz": ["JAZZ", "jazz"],
        "Blues/R&B": ["BLUES", "blues"],
        "Show Tunes": ["MUSICALS", "musicals", "SHOWTUNES", "showtunes"],
        "Oldies": ["OLDIES", "oldies"],
        "Classical/Chamber": ["CLASSICL", "classicl", "CLASSICAL", "classical"],
        "Swing/Big Band": ["BIGBAND", "bigband", "SWING", "swing"],
        "New Age/Space": ["NEWAGE", "newage", "NEW_AGE", "new_age"],
        "Opera": ["OPERA", "opera"],
        "Bluegrass": ["BLUGRASS", "blugrass", "BLUEGRASS", "bluegrass"],
        "Folk": ["FOLK", "folk"],
        "Easy Listening": ["MOODEASY", "moodeasy", "EASYLIST", "easylist", "EASY", "easy"],
        "Pop/Contemporary Rock": ["CONROCK", "conrock", "POPROCK", "poprock", "ROCK", "rock"],
        "Reggae": ["REGGAE", "reggae"],
        "Rap": ["RAP", "rap", "HIPHOP", "hiphop", "HIP_HOP", "hip_hop"],
        "Heavy Metal": ["HVYMETAL", "hvymetal", "HEAVYMETAL", "heavymetal", "HEAVY_METAL", "heavy_metal"],
        "Country/Western": ["COUNTRY", "country", "COUNTRYWESTERN", "countrywestern", "COUNTRY_WESTERN", "country_western"],
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
    # Recode dislikes/likes for all genres
    # ----------------------------
    dislike = pd.DataFrame(index=df.index)
    like = pd.DataFrame(index=df.index)
    for g in genre_order:
        col = resolved_cols[g]
        dislike[g] = recode_dislike(df[col])
        like[g] = recode_like(df[col])

    # Sample mean education (respondent-level; education non-missing)
    sample_mean_edu = float(df[educ_col].mean()) if len(df) else np.nan

    # ----------------------------
    # Per-genre logistic regression:
    #   y = Dislike_g
    #   x = Tolerance_-g (# of other genres not disliked)
    #   + education
    # ----------------------------
    results = []
    for g in genre_order:
        y = dislike[g]

        others = [h for h in genre_order if h != g]
        others_dislike = dislike[others]

        # Count number of other genres not disliked: sum(1 - Dislike_h) across other 17
        # Require all 17 other genre responses observed (no imputation)
        tol_minus_g = (1.0 - others_dislike).sum(axis=1, min_count=len(others))

        model_df = pd.DataFrame(
            {"y": y, "tolerance": tol_minus_g, "educ": df[educ_col]}
        ).dropna()

        coef_tol = np.nan
        pval_tol = np.nan
        if len(model_df) > 0 and model_df["y"].nunique() >= 2 and model_df["tolerance"].nunique() >= 2:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=500)
                coef_tol = float(fit.params["tolerance"])
                pval_tol = float(fit.pvalues["tolerance"])
            except Exception:
                # fallback for separation/instability
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=5000)
                    coef_tol = float(fit.params["tolerance"])
                    pval_tol = np.nan
                except Exception:
                    coef_tol = np.nan
                    pval_tol = np.nan

        # Mean education among "likers" (1/2 only)
        aud_df = pd.DataFrame({"like": like[g], "educ": df[educ_col]}).dropna()
        mean_edu = np.nan
        if len(aud_df) and (aud_df["like"] == 1.0).any():
            mean_edu = float(aud_df.loc[aud_df["like"] == 1.0, "educ"].mean())

        results.append(
            {"genre": g, "coef_tolerance": coef_tol, "pval": pval_tol, "mean_edu": mean_edu}
        )

    res = pd.DataFrame(results)

    # Align to canonical Figure-1 x order (NO sorting by values)
    res["genre"] = pd.Categorical(res["genre"], categories=genre_order, ordered=True)
    res = res.sort_values("genre").reset_index(drop=True)

    coef = res.set_index("genre").reindex(genre_order)["coef_tolerance"].to_numpy(dtype="float64")
    edu = res.set_index("genre").reindex(genre_order)["mean_edu"].to_numpy(dtype="float64")

    # ----------------------------
    # Plot (match Figure 1 mapping/styles)
    # ----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(genre_order), dtype=int)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)
    ax2 = ax.twinx()

    # LEFT axis: coefficient (SOLID)
    ax.plot(x, coef, color="black", lw=2.0, ls="-", zorder=3)

    # RIGHT axis: mean education (DASH-DOT)
    ax2.plot(x, edu, color="black", lw=2.0, ls="-.", zorder=2)

    # Sample mean education (RIGHT axis), dotted
    if np.isfinite(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="black", lw=1.2, ls=":", zorder=1)

    # Labels
    ax.set_xlabel("Type of Music", fontweight="bold")
    ax.set_ylabel(
        "Coefficients for Musical Tolerance as It Affects\n"
        "One’s Probability of Disliking Each Music Genre"
    )
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\n"
        "Reported Liking Each Music Genre",
        rotation=270,
        labelpad=28
    )

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(genre_order, rotation=45, ha="right")

    # Axis ranges/ticks/formatting
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # Styling similar to printed figure
    ax.grid(False)
    ax2.grid(False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(direction="out")
    ax2.tick_params(direction="out")

    # ----------------------------
    # Annotations (ensure axis-correct targeting)
    # ----------------------------
    def first_finite_idx(arr, default=2):
        for i, v in enumerate(arr):
            if np.isfinite(v):
                return i
        return min(max(default, 0), len(arr) - 1)

    i_anchor = min(2, len(x) - 1)

    # Education callout: must be on ax2 coordinates
    i_edu = first_finite_idx(edu, default=i_anchor)
    edu_xy = (int(i_edu), float(edu[i_edu]) if np.isfinite(edu[i_edu]) else 14.0)
    ax2.annotate(
        "Mean Education of Genre Audience",
        xy=edu_xy,
        xycoords=("data", "data"),
        xytext=(max(0.5, edu_xy[0] + 0.8), 14.2),
        textcoords=("data", "data"),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Coefficient callout: must be on ax coordinates
    i_coef = first_finite_idx(coef, default=i_anchor)
    coef_xy = (int(i_coef), float(coef[i_coef]) if np.isfinite(coef[i_coef]) else -0.35)
    ax.annotate(
        "Coefficient for Musical Tolerance",
        xy=coef_xy,
        xycoords=("data", "data"),
        xytext=(max(0.5, coef_xy[0] + 0.8), -0.47),
        textcoords=("data", "data"),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Sample mean education label with upward arrow (ax2)
    if np.isfinite(sample_mean_edu):
        mean_x = int(min(6, len(genre_order) - 1))
        ax2.annotate(
            "Sample Mean Education",
            xy=(mean_x, sample_mean_edu),
            xycoords=("data", "data"),
            xytext=(mean_x, sample_mean_edu - 0.35),
            textcoords=("data", "data"),
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