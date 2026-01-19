def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter
    import statsmodels.api as sm

    # ----------------------------
    # Global style to mimic printed figure
    # ----------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
    })

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

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

    # Drop missing education up-front (used in all models and sample mean)
    df = df.dropna(subset=[educ_col]).copy()

    # ----------------------------
    # Figure 1 canonical genre order + exact tick labels used for plotting
    # (Keep these fixed; DO NOT sort by coefficients/education.)
    # ----------------------------
    genre_order = [
        "Latin/Salsa", "Jazz", "Blues/R&B", "Show Tunes", "Oldies", "Classical",
        "Reggae", "Swing", "New Age/Space", "Opera", "Bluegrass", "Folk",
        "Easy Listening", "Pop/Rock", "Rap", "Heavy Metal", "Country", "Gospel"
    ]

    # Map tick labels -> dataset column candidates
    # (Use provided 1993 GSS culture-module column names; add fallbacks.)
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
        "Easy Listening": ["MOODEASY", "moodeasy", "EASYLIST", "easylist", "EASY_LISTENING", "easy_listening", "EASY", "easy"],
        "Pop/Rock": ["CONROCK", "conrock", "POPROCK", "poprock", "POP_ROCK", "pop_rock", "ROCK", "rock"],
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
        col = resolved_cols[g]
        dislike[g] = recode_dislike(df[col])
        like[g] = recode_like(df[col])

    # Sample mean education (computed from the analysis sample with non-missing EDUC)
    sample_mean_edu = float(df[educ_col].mean()) if len(df) else np.nan

    # ----------------------------
    # Per-genre: fit logit + mean edu among likers
    # ----------------------------
    rows = []
    for g in genre_order:
        y = dislike[g]
        educ = df[educ_col]

        others = [h for h in genre_order if h != g]
        others_dislike = dislike[others]

        # Tolerance_-g = number of other genres not disliked (requires all 17 observed)
        tol_minus_g = (1.0 - others_dislike).sum(axis=1, min_count=len(others))  # NaN if any missing among the 17

        # CRITICAL: drop missing values before model
        model_df = pd.DataFrame({"y": y, "tolerance": tol_minus_g, "educ": educ}).dropna()

        coef_tol = np.nan
        pval_tol = np.nan
        if len(model_df) > 0 and model_df["y"].nunique() >= 2 and model_df["tolerance"].nunique() >= 2:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=500)
                coef_tol = float(fit.params["tolerance"])
                pval_tol = float(fit.pvalues["tolerance"])
            except Exception:
                # fallback (rare) if separation causes failure
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=5000)
                    coef_tol = float(fit.params["tolerance"])
                    pval_tol = np.nan
                except Exception:
                    coef_tol = np.nan
                    pval_tol = np.nan

        # Mean education among likers (like very much / like it = 1/2)
        aud_df = pd.DataFrame({"like": like[g], "educ": educ}).dropna()
        if len(aud_df) and (aud_df["like"] == 1.0).any():
            mean_edu = float(aud_df.loc[aud_df["like"] == 1.0, "educ"].mean())
        else:
            mean_edu = np.nan

        rows.append({"genre": g, "coef_tolerance": coef_tol, "pval": pval_tol, "mean_edu": mean_edu})

    res = pd.DataFrame(rows)

    # Enforce exact x-order (no sorting)
    res = res.set_index("genre").reindex(genre_order).reset_index()

    # ----------------------------
    # Plot
    # ----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(genre_order), dtype=int)
    coef = res["coef_tolerance"].to_numpy(dtype="float64")
    edu = res["mean_edu"].to_numpy(dtype="float64")

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)
    ax2 = ax.twinx()

    # Solid line = coefficient (LEFT axis)
    ax.plot(x, coef, color="black", lw=2.0, ls="-", zorder=3)

    # Dash-dot line = mean education (RIGHT axis)
    ax2.plot(x, edu, color="black", lw=2.0, ls="-.", zorder=2)

    # Dotted reference line = sample mean education (RIGHT axis)
    if np.isfinite(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="black", lw=1.0, ls=":", zorder=1)

    # Axis labels/titles (comparable to article)
    ax.set_xlabel("Type of Music", fontweight="bold", fontsize=16)
    ax.set_ylabel(
        "Coefficients for Musical Tolerance as It Affects\n"
        "One’s Probability of Disliking Each Music Genre",
        fontsize=12
    )
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\n"
        "Reported Liking Each Music Genre",
        rotation=270,
        labelpad=28,
        fontsize=12
    )

    # X ticks/labels
    ax.set_xticks(x)
    ax.set_xticklabels(genre_order, rotation=45, ha="right")

    # Left axis range/ticks + formatter (no leading zero)
    ax.set_ylim(-0.5, 0.0)
    ax.set_yticks([0.0, -0.1, -0.2, -0.3, -0.4, -0.5])
    ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

    # Right axis range/ticks
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # Spines/ticks
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=8, width=1.5)
    ax2.tick_params(direction="out", length=8, width=1.5)

    ax.grid(False)
    ax2.grid(False)

    # ----------------------------
    # Annotations (axis-specific, fixed positions; tune as needed)
    # ----------------------------
    # Use index 2 (Blues/R&B) as anchor if finite; otherwise find first finite.
    def first_finite_idx(arr, fallback=2):
        if len(arr) == 0:
            return 0
        if 0 <= fallback < len(arr) and np.isfinite(arr[fallback]):
            return fallback
        for i, v in enumerate(arr):
            if np.isfinite(v):
                return i
        return 0

    i = first_finite_idx(edu, fallback=2)
    j = first_finite_idx(coef, fallback=2)

    # Mean education label pointing to dash-dot line (right axis)
    if np.isfinite(edu[i]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(2, float(edu[2]) if len(edu) > 2 and np.isfinite(edu[2]) else float(edu[i])),
            xycoords="data",
            xytext=(1.5, 14.2),
            textcoords="data",
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="center"
        )

    # Coefficient label pointing to solid line (left axis)
    if np.isfinite(coef[j]):
        ax.annotate(
            "Coefficient for Musical Tolerance",
            xy=(2, float(coef[2]) if len(coef) > 2 and np.isfinite(coef[2]) else float(coef[j])),
            xycoords="data",
            xytext=(2.5, -0.47),
            textcoords="data",
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="center"
        )

    # Sample mean education arrow (right axis)
    if np.isfinite(sample_mean_edu):
        ax2.annotate(
            "Sample Mean Education",
            xy=(6, sample_mean_edu),
            xycoords="data",
            xytext=(6, 12.85),
            textcoords="data",
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
            fontsize=16,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)