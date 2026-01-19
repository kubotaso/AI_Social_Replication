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
            key = str(name).lower()
            if key in lower_map:
                return lower_map[key]

        if contains_any:
            contains_any = [str(x).lower() for x in contains_any]
            hits = []
            for c in cols:
                cl = c.lower()
                if any(s in cl for s in contains_any):
                    hits.append(c)
            if len(hits) == 1:
                return hits[0]

        for name in preferred_names:
            if name is None:
                continue
            n = str(name).lower()
            matches = [c for c in cols if n in c.lower()]
            if len(matches) == 1:
                return matches[0]

        return None

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def recode_dislike(series):
        # 1 if response is 4/5, 0 if 1/2/3, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x >= 1) & (x <= 3)] = 0.0
        out[(x == 4) | (x == 5)] = 1.0
        return out

    def recode_like(series):
        # 1 if response is 1/2, 0 if 3/4/5, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x == 1) | (x == 2)] = 1.0
        out[(x >= 3) & (x <= 5)] = 0.0
        return out

    def no_leading_zero(x, pos):
        s = f"{x:.1f}"
        s = s.replace("-0.", "-.").replace("0.", ".")
        return s

    # ----- Load -----
    df = pd.read_csv(data_source)

    # ----- Filter to 1993 -----
    year_col = find_col(df, ["YEAR", "year"])
    if year_col is not None:
        df[year_col] = to_num(df[year_col])
        df = df.loc[df[year_col] == 1993].copy()

    # ----- Education -----
    educ_col = find_col(df, ["EDUC", "educ", "education", "years_education", "schooling"])
    if educ_col is None:
        raise ValueError("Could not find education column (EDUC/educ).")
    df[educ_col] = to_num(df[educ_col])

    # CRITICAL: drop missing education immediately for computations relying on it
    df = df.dropna(subset=[educ_col]).copy()

    # ----- Exact GT order and labels (use these exactly; do not sort) -----
    order = [
        "Latin/Salsa", "Jazz", "Blues/R&B", "Show Tunes", "Oldies", "Classical",
        "Reggae", "Swing", "New Age/Space", "Opera", "Bluegrass", "Folk",
        "Easy Listening", "Pop/Rock", "Rap", "Heavy Metal", "Country", "Gospel"
    ]

    # ----- Map labels to dataset columns (robust fallbacks) -----
    genre_map = {
        "Latin/Salsa": ["LATIN", "latin"],
        "Jazz": ["JAZZ", "jazz"],
        "Blues/R&B": ["BLUES", "blues"],
        "Show Tunes": ["MUSICALS", "musicals", "SHOWTUNES", "showtunes"],
        "Oldies": ["OLDIES", "oldies"],
        "Classical": ["CLASSICL", "classicl", "CLASSICAL", "classical"],
        "Swing": ["BIGBAND", "bigband", "SWING", "swing"],
        "New Age/Space": ["NEWAGE", "newage", "NEW_AGE", "new_age"],
        "Opera": ["OPERA", "opera"],
        "Bluegrass": ["BLUGRASS", "blugrass", "BLUEGRASS", "bluegrass"],
        "Folk": ["FOLK", "folk"],
        "Easy Listening": ["MOODEASY", "moodeasy", "EASYLIST", "easylist", "EASY", "easy"],
        "Pop/Rock": ["CONROCK", "conrock", "POPROCK", "poprock", "ROCK", "rock"],
        "Reggae": ["REGGAE", "reggae"],
        "Rap": ["RAP", "rap", "HIPHOP", "hiphop", "HIP_HOP", "hip_hop"],
        "Heavy Metal": ["HVYMETAL", "hvymetal", "HEAVYMETAL", "heavymetal", "HEAVY_METAL", "heavy_metal"],
        "Country": ["COUNTRY", "country", "COUNTRYWESTERN", "countrywestern", "COUNTRY_WESTERN", "country_western"],
        "Gospel": ["GOSPEL", "gospel"],
    }

    resolved = {}
    for label in order:
        col = find_col(df, genre_map.get(label, [label]))
        if col is None:
            # fallback: contains tokens of label
            tokens = [t for t in label.lower().replace("/", " ").replace("&", " ").replace("-", " ").split() if t]
            col = find_col(df, [], contains_any=tokens)
        if col is None:
            raise ValueError(f"Could not find column for genre '{label}'. Tried: {genre_map.get(label)}")
        resolved[label] = col

    # ----- Build dislike/like matrices -----
    dislike = pd.DataFrame(index=df.index)
    like = pd.DataFrame(index=df.index)

    for label in order:
        col = resolved[label]
        dislike[label] = recode_dislike(df[col])
        like[label] = recode_like(df[col])

    # CRITICAL: for tolerance computation, we need valid genre responses; keep as NaN where missing.
    # Sample mean education: respondent-level mean (1993, non-missing EDUC)
    sample_mean_edu = float(df[educ_col].mean()) if len(df) else np.nan

    # ----- Fit per-genre logits and compute mean edu of audience -----
    rows = []
    for g in order:
        y = dislike[g]

        others = [h for h in order if h != g]
        others_dislike = dislike[others]

        # Tolerance_-g = number of other genres NOT disliked (0/1), requiring all 17 observed
        not_disliked = 1.0 - others_dislike
        tol_minus_g = not_disliked.sum(axis=1, min_count=len(others))

        model_df = pd.DataFrame(
            {"y": y, "tolerance": tol_minus_g, "educ": df[educ_col]}
        ).dropna()

        coef_tol = np.nan
        pval_tol = np.nan
        if len(model_df) > 0 and model_df["y"].nunique() >= 2 and model_df["tolerance"].nunique() >= 2:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=300)
                coef_tol = float(fit.params["tolerance"])
                pval_tol = float(fit.pvalues["tolerance"])
            except Exception:
                # separation fallback; still returns a coefficient
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=2000)
                    coef_tol = float(fit.params["tolerance"])
                    pval_tol = np.nan
                except Exception:
                    coef_tol = np.nan
                    pval_tol = np.nan

        # Mean education among "likers" (like very much / like it => 1/2)
        aud_df = pd.DataFrame({"like": like[g], "educ": df[educ_col]}).dropna()
        if len(aud_df) and (aud_df["like"] == 1.0).any():
            mean_edu = float(aud_df.loc[aud_df["like"] == 1.0, "educ"].mean())
        else:
            mean_edu = np.nan

        rows.append({"genre": g, "coef_tolerance": coef_tol, "pval": pval_tol, "mean_edu": mean_edu})

    res = pd.DataFrame(rows)

    # CRITICAL: enforce GT x-order alignment (no sorting by values)
    res = res.set_index("genre").reindex(order).reset_index()

    # ----- Plot -----
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)
    ax2 = ax.twinx()

    # LEFT axis: coefficient (solid)
    coef_vals = res["coef_tolerance"].to_numpy(dtype="float64")
    ax.plot(x, coef_vals, color="black", lw=1.6, ls="-", zorder=3)

    # RIGHT axis: mean education (dash-dot)
    edu_vals = res["mean_edu"].to_numpy(dtype="float64")
    edu_line = ax2.plot(x, edu_vals, color="black", lw=1.6, ls="-.", zorder=2)[0]
    # Optional: tighten dash-dot cadence closer to print look
    try:
        edu_line.set_dashes([8, 3, 2, 3])  # dash, gap, dot, gap
    except Exception:
        pass

    # Sample mean education (RIGHT axis) dotted
    if np.isfinite(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="black", lw=1.0, ls=":", zorder=1)

    # Labels (match paper wording closely)
    ax.set_xlabel("Type of Music", fontweight="bold")
    ax.set_ylabel(
        "Coefficients for Musical Tolerance as It Affects\n"
        "One’s Probability of Disliking Each Music Genre",
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\n"
        "Reported Liking Each Music Genre",
        fontweight="bold",
        rotation=270,
        labelpad=28,
    )

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")

    # Left axis (coefficients): -0.5 to -0.1 with specified ticks, no leading zero
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

    # Right axis (education): 12 to 15 with integer ticks
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # No gridlines
    ax.grid(False)
    ax2.grid(False)

    # Classic look
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1)
    ax2.tick_params(direction="out", length=6, width=1)

    # ----- Annotations / callouts with leader lines -----
    # choose a stable anchor index with finite values
    def first_finite_idx(arr):
        for i, v in enumerate(arr):
            if np.isfinite(v):
                return i
        return 0

    i_edu = first_finite_idx(edu_vals)
    i_coef = first_finite_idx(coef_vals)

    # Education label (targets dash-dot line on RIGHT axis)
    if np.isfinite(edu_vals[i_edu]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(x[i_edu], float(edu_vals[i_edu])),
            xycoords=("data", "data"),
            xytext=(x[max(0, i_edu - 1)] + 0.2, 14.2),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Coefficient label (targets solid line on LEFT axis)
    if np.isfinite(coef_vals[i_coef]):
        ax.annotate(
            "Coefficient for Musical Tolerance",
            xy=(x[i_coef], float(coef_vals[i_coef])),
            xycoords=("data", "data"),
            xytext=(x[max(0, i_coef - 1)] + 0.6, -0.47),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Sample mean education label with upward arrow (RIGHT axis)
    if np.isfinite(sample_mean_edu):
        mean_x = int(min(6, len(order) - 1))
        ax2.annotate(
            "Sample Mean Education",
            xy=(x[mean_x], sample_mean_edu),
            xycoords=("data", "data"),
            xytext=(x[mean_x], 12.7),
            textcoords=("data", "data"),
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)