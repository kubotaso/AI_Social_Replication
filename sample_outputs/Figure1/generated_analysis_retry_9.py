def run_analysis(data_source, sep=None, na_values=None):
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    from sklearn.linear_model import LogisticRegression

    # 1) Read data
    if sep is None:
        sep = ','
    df = pd.read_csv(data_source, sep=sep, na_values=na_values)

    # 2) Optional filter to 1993 if present
    if 'year' in df.columns:
        try:
            df = df[df['year'].astype(float) == 1993]
        except Exception:
            pass  # keep as-is if coercion fails

    # 3) Helper: robust column finder
    def find_column(possible_names, columns):
        # Return the first matching column name (original case) by exact (case-insensitive) or substring match
        cols = list(columns)
        cols_lc = {c.lower(): c for c in cols}
        # exact, case-insensitive
        for cand in possible_names:
            c = cand.lower()
            if c in cols_lc:
                return cols_lc[c]
        # substring search
        for cand in possible_names:
            c = cand.lower()
            for col in cols:
                if c in col.lower():
                    return col
        # reverse substring search (column name inside candidate)
        for col in cols:
            cl = col.lower()
            for cand in possible_names:
                if cl in cand.lower():
                    return col
        return None

    # 4) Define genre mapping (display -> candidate variable names)
    genre_to_candidates = {
        "Latin/Salsa": ["latinsa", "latin", "latsalsa", "mlatinsa", "mlatin"],
        "Jazz": ["jazz", "mjazz"],
        "Blues/R&B": ["blues", "bluesrb", "mblues"],
        "Show Tunes": ["showtune", "showtun", "shows", "musicals", "mshowtu"],
        "Oldies": ["oldies", "moldies"],
        "Classical": ["classmus", "classicl", "classical", "clschmbr", "classchm", "mclass", "classic"],
        "Reggae": ["reggae", "mreggae"],
        "Swing": ["swing", "bigband", "mswing"],
        "New Age/Space": ["newage", "mnewage"],
        "Opera": ["opera", "mopera"],
        "Bluegrass": ["bluegras", "bluegrss", "mbluegra", "blugrass", "bluegrass"],
        "Folk": ["folk", "mfolk"],
        "Easy Listening": ["pop", "moodeasy", "easylist", "easylis", "mpop", "easylistening"],
        "Pop/Rock": ["conrock", "controck", "rockcont", "mconrock", "poprock", "pop_rock", "contemporaryrock", "contemporary_rock"],
        "Rap": ["rap", "mrap"],
        "Heavy Metal": ["heavymtl", "heavymet", "mheavym", "mheavymt", "hvymetal", "heavy_metal", "hvymetl", "hvymet"],
        "Country": ["country", "mcountry", "country_western", "country/western", "countrywestern"],
        "Gospel": ["gospel", "mgospel"],
    }

    # 5) Find columns for all genres; raise clear error if any missing
    missing = []
    found_col_for_genre = {}
    for disp, cands in genre_to_candidates.items():
        col = find_column(cands, df.columns)
        if col is None:
            missing.append((disp, cands))
        else:
            found_col_for_genre[disp] = col
    if missing:
        parts = []
        for disp, cands in missing:
            parts.append(f"{disp}: searched {cands}")
        raise ValueError(
            "Could not locate the following required music preference variables.\n"
            + "\n".join(parts) +
            f"\nAvailable columns: {list(df.columns)}"
        )

    # 6) Education column (robust search)
    educ_col = find_column(["educ", "education", "eduyrs", "years_school", "years of school"], df.columns)
    if educ_col is None:
        raise ValueError(
            "Could not find an education (years) variable. Searched ['educ','education','eduyrs','years_school'].\n"
            f"Available columns: {list(df.columns)}"
        )
    # Coerce education to numeric
    edu = pd.to_numeric(df[educ_col], errors='coerce')

    # 7) Build tidy DataFrame of the 18 genre items (converted to numeric; values outside 1–5 as NaN)
    music_df = df[[found_col_for_genre[g] for g in genre_to_candidates.keys()]].copy()
    music_df.columns = list(genre_to_candidates.keys())
    # Coerce and sanitize values
    for g in music_df.columns:
        music_df[g] = pd.to_numeric(music_df[g], errors='coerce')
        invalid_mask = ~music_df[g].isna() & ((music_df[g] < 1) | (music_df[g] > 5))
        if invalid_mask.any():
            music_df.loc[invalid_mask, g] = np.nan

    # 8) Compute overall sample mean education (exclude missing)
    sample_mean_edu = float(edu.dropna().mean()) if edu.notna().any() else np.nan

    # 9) Prepare outputs
    coef_by_genre = {}
    mean_edu_likers = {}

    # 10) Precompute not-disliked mask for all genres (1–3)
    not_disliked = (music_df >= 1) & (music_df <= 3)

    # 11) For each genre: compute dependent Y (dislike indicator), TOL_g, and fit logit(Y ~ TOL + educ)
    for g in music_df.columns:
        # Y_g: 1 if 4 or 5, 0 if 1-3, NaN otherwise
        y = np.where(music_df[g].isin([4, 5]), 1,
                     np.where(music_df[g].isin([1, 2, 3]), 0, np.nan)).astype(float)

        # TOL_g: count of not-disliked among other 17 genres; require all other 17 to be non-missing
        others = [c for c in music_df.columns if c != g]
        valid_others = music_df[others].notna().all(axis=1)
        tol = not_disliked[others].sum(axis=1).astype(float)
        tol[~valid_others] = np.nan

        # Modeling DataFrame; drop rows with missing in y, tol, or educ
        mod = pd.DataFrame({
            "y": y,
            "TOL": tol,
            "educ": edu
        }).dropna()

        coef_val = np.nan
        if len(mod) > 0 and mod['y'].nunique() == 2:
            # statsmodels Logit
            try:
                X = sm.add_constant(mod[["TOL", "educ"]].astype(float), has_constant='add')
                y_vec = mod["y"].astype(float)
                fit = sm.Logit(y_vec, X).fit(disp=False, maxiter=1000)
                if "TOL" in fit.params.index:
                    coef_val = float(fit.params["TOL"])
                else:
                    coef_val = np.nan
            except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
                # Fallback to scikit-learn LogisticRegression (L2-regularized)
                try:
                    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
                    X_lr = mod[["TOL", "educ"]].to_numpy(dtype=float)
                    y_lr = mod["y"].to_numpy(dtype=int)
                    # Ensure both classes present
                    if len(np.unique(y_lr)) == 2:
                        lr.fit(X_lr, y_lr)
                        # coefs correspond to order in X_lr columns
                        coef_dict = {"TOL": lr.coef_[0][0], "educ": lr.coef_[0][1]}
                        coef_val = float(coef_dict["TOL"])
                except Exception:
                    coef_val = np.nan
        coef_by_genre[g] = coef_val

        # Mean education among likers (responses 1 or 2 on g)
        likers_mask = music_df[g].isin([1, 2]) & edu.notna()
        mean_edu_likers[g] = float(edu[likers_mask].mean()) if likers_mask.any() else np.nan

    coef_series = pd.Series(coef_by_genre)
    meanedu_series = pd.Series(mean_edu_likers)

    # 12) Fixed category order for plotting (as in ground-truth)
    genre_order = [
        "Latin/Salsa", "Jazz", "Blues/R&B", "Show Tunes", "Oldies", "Classical",
        "Reggae", "Swing", "New Age/Space", "Opera", "Bluegrass", "Folk",
        "Easy Listening", "Pop/Rock", "Rap", "Heavy Metal", "Country", "Gospel"
    ]

    # Verify availability for all ordered labels
    missing_for_plot = [g for g in genre_order if g not in coef_series.index or g not in meanedu_series.index]
    if missing_for_plot:
        raise ValueError(f"Missing series values for these genres (cannot plot): {missing_for_plot}")

    coef_plot = coef_series.reindex(genre_order)
    edu_plot = meanedu_series.reindex(genre_order)

    # 13) Build the plot exactly as specified
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax2 = ax.twinx()

    x = np.arange(len(genre_order))

    # Left axis (coefficients): solid black line
    line_coef, = ax.plot(x, coef_plot.values, color="k", linestyle="-", linewidth=2.0, zorder=3)

    # Right axis (mean education among likers): dash-dot black line with custom dash pattern
    line_edu, = ax2.plot(x, edu_plot.values, color="k", linestyle="-.", linewidth=2.0, zorder=2)
    try:
        line_edu.set_dashes([7, 3, 1.8, 3])  # long dash – gap – dot – gap
    except Exception:
        pass

    # Sample mean education: horizontal dotted line on right axis
    if not np.isnan(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="0.35", linestyle=(0, (1, 3)), linewidth=1.0, zorder=1)

    # Axis limits and ticks
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    minus = "\u2212"
    def yfmt(v, _):
        return (minus + f"{abs(v):.1f}").replace("0.", ".")
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt))

    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])

    # X axis categories
    ax.set_xticks(x)
    ax.set_xticklabels(genre_order, rotation=45, ha="right", rotation_mode="anchor")

    # Labels (exact wording)
    ax.set_xlabel("Type of Music")
    ax.set_ylabel("Coefficients for Musical Tolerance as It Affects One’s Probability of Disliking Each Music Genre")
    ax2.set_ylabel("Mean Educational Level of Respondents Who Reported Liking Each Music Genre")

    # Spines and ticks styling (black)
    for a in (ax, ax2):
        a.tick_params(colors='k')
        for spine in a.spines.values():
            spine.set_color('k')

    # No legend; add in-plot annotations with arrows
    # Coefficient callout (lower-left area)
    try:
        idx_coef = 3 if not np.isnan(coef_plot.values[3]) else int(np.nanargmin(np.abs(coef_plot.values + 0.3)))
    except Exception:
        idx_coef = 3
    ax.annotate(
        "Coefficient for Musical Tolerance",
        xy=(x[idx_coef], coef_plot.values[idx_coef]),
        xytext=(-40, -25), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="k", lw=1),
        ha="right", va="top", color="k"
    )

    # Mean education callout (upper-middle area)
    try:
        idx_edu = 9 if not np.isnan(edu_plot.values[9]) else int(np.nanargmax(edu_plot.values))
    except Exception:
        idx_edu = 9
    ax2.annotate(
        "Mean Education of Genre Audience",
        xy=(x[idx_edu], edu_plot.values[idx_edu]),
        xytext=(-20, 25), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="k", lw=1),
        ha="right", va="bottom", color="k"
    )

    # Sample mean callout (near center)
    if not np.isnan(sample_mean_edu):
        ax2.annotate(
            "Sample Mean Education",
            xy=(x[len(x)//2], sample_mean_edu),
            xytext=(10, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="k", lw=1),
            ha="left", va="top", color="k"
        )

    # Figure caption and note (below the plot)
    fig.text(
        0.5, -0.02,
        "Figure 1. The Effect of Being Musically Tolerant on Disliking Each Music Genre Compared to the Educational Composition of Genre Audiences",
        ha="center", va="top", fontsize=10
    )
    fig.text(
        0.5, -0.065,
        "Note: Coefficients represent the effects of being musically tolerant (number of genres not disliked, excluding the one involved in the dependent variable) from logistic regression equations where disliking each genre is the dependent variable and education is controlled. All coefficients are significant at p < .0001.",
        ha="center", va="top", fontsize=9, style="italic"
    )

    # Layout and save
    fig.subplots_adjust(left=0.17, right=0.86, bottom=0.28, top=0.95)
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "generated_results.jpg"
    fig.savefig(out_path, dpi=300, format="jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return str(out_path.resolve())