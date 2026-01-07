def run_analysis(data_source, sep=None, na_values=None):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    from matplotlib.ticker import FuncFormatter

    # Read data
    if sep is None:
        sep = ','
    df = pd.read_csv(data_source, sep=sep, na_values=na_values)
    df.columns = [c.strip() for c in df.columns]

    # Helper: find a column by candidates (case-insensitive)
    def find_column(df_cols, candidates):
        cols_lower = {c.lower(): c for c in df_cols}
        # exact matches
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        # substring matches
        for cand in candidates:
            for c in df_cols:
                cl = c.lower()
                if cand.lower() in cl:
                    return c
        return None

    # Core columns: year and education
    year_col = find_column(df.columns, ['year'])
    if year_col is None:
        raise ValueError("Required column 'year' not found in the data.")
    educ_col = find_column(df.columns, ['educ', 'education', 'educyrs', 'educyr', 'years_of_education', 'education_years'])
    if educ_col is None:
        raise ValueError("Required column for years of education not found (tried: 'educ', 'education', etc.).")

    # Restrict to 1993
    df = df.copy()
    # Coerce year numeric
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df = df[df[year_col] == 1993].copy()
    if df.empty:
        raise ValueError("No rows found for year 1993 after filtering.")

    # Desired display order and labels
    categories = ['Latin/Salsa','Jazz','Blues/R&B','Show Tunes','Oldies','Classical','Reggae','Swing','New Age/Space','Opera','Bluegrass','Folk','Easy Listening','Pop/Rock','Rap','Heavy Metal','Country','Gospel']

    # Map display labels to plausible column names in file
    pattern_map = {
        'Latin/Salsa': ['muslatin', 'latin'],
        'Jazz': ['musjazz', 'jazz'],
        'Blues/R&B': ['musblues', 'blues', 'r&b', 'rnb', 'rhythm and blues'],
        'Show Tunes': ['musshow', 'showtune', 'show tunes', 'broadway', 'musicals', 'show'],
        'Oldies': ['musoldie', 'musoldies', 'oldies'],
        'Classical': ['musclass', 'musclas', 'classicl', 'classical', 'chamber'],
        'Reggae': ['musregga', 'musreggae', 'reggae'],
        'Swing': ['musswing', 'swing', 'bigband', 'big band', 'big_band', 'big-band'],
        'New Age/Space': ['musnewage', 'musnewag', 'newage', 'new age', 'space'],
        'Opera': ['musopera', 'opera'],
        'Bluegrass': ['musblueg', 'musbluegr', 'bluegrass', 'blugrass', 'blueg'],
        'Folk': ['musfolk', 'folk'],
        'Easy Listening': ['museasy', 'moodeasy', 'easy listening', 'easy', 'easylis', 'easylistening'],
        'Pop/Rock': ['conrock', 'contemporary rock', 'pop/rock', 'poprock', 'pop rock', 'muspop', 'pop'],
        'Rap': ['musrap', 'rap'],
        'Heavy Metal': ['mushemet', 'musheavy', 'mushmetl', 'hvymetal', 'heavy metal', 'heavy'],
        'Country': ['muscoun', 'muscw', 'muscwest', 'country', 'western', 'cw'],
        'Gospel': ['musgospel', 'musgospl', 'gospel'],
    }
    # Known aliases from provided sample columns
    alias_pref = {
        'Swing': 'bigband',
        'Bluegrass': 'blugrass',
        'Classical': 'classicl',
        'Easy Listening': 'moodeasy',
        'Pop/Rock': 'conrock',
        'Heavy Metal': 'hvymetal'
    }

    used_cols = set()
    music_map = {}
    cols_lower_map = {c.lower(): c for c in df.columns}

    for disp in categories:
        # Prefer alias if present
        alias = alias_pref.get(disp)
        if alias and alias.lower() in cols_lower_map and cols_lower_map[alias.lower()] not in used_cols:
            music_map[disp] = cols_lower_map[alias.lower()]
            used_cols.add(music_map[disp])
            continue
        # Try exact candidates
        cand_list = pattern_map.get(disp, [])
        found = None
        # First, exact case-insensitive matches
        for cand in cand_list:
            c = cols_lower_map.get(cand.lower())
            if c and c not in used_cols:
                found = c
                break
        # Then, substring matches with simple guards for ambiguous ones
        if found is None:
            for c in df.columns:
                cl = c.lower()
                for cand in cand_list:
                    candl = cand.lower()
                    if candl in cl:
                        # guard against 'blues' matching 'bluegrass'
                        if disp == 'Blues/R&B' and ('grass' in cl):
                            continue
                        # guard against 'heavy' matching 'contemporary rock' etc.
                        if disp == 'Heavy Metal' and ('heavy' in candl or 'heavy metal' in candl):
                            pass
                        # avoid 'pop' matching 'moodeasy'
                        if disp == 'Pop/Rock' and ('moodeasy' in cl or 'easy' in cl):
                            continue
                        if c not in used_cols:
                            found = c
                            break
                if found is not None:
                    break
        if found is None:
            raise ValueError(f"Could not find a column for genre '{disp}'. Available columns: {list(df.columns)}")
        music_map[disp] = found
        used_cols.add(found)

    # Subset to columns of interest
    music_cols = [music_map[g] for g in categories]
    work = df[[educ_col] + music_cols].copy()

    # Coerce music responses to numeric and set invalids (non 1..5) to NaN
    for c in music_cols:
        work[c] = pd.to_numeric(work[c], errors='coerce')
        mask_valid = work[c].between(1, 5)
        work.loc[~mask_valid, c] = np.nan

    # Drop respondents with any "don't know" (NaN after coercion) on the music battery
    work = work.dropna(subset=music_cols, how='any').copy()

    # Coerce education to numeric and drop missing education
    work[educ_col] = pd.to_numeric(work[educ_col], errors='coerce')
    work = work.dropna(subset=[educ_col]).copy()

    if work.empty:
        raise ValueError("No valid cases remain after dropping 'don't know' responses and missing education.")

    # Build dislike indicators (1 if 4 or 5; else 0)
    music_arr = work[music_cols].to_numpy()
    dislike_all = ((music_arr >= 4) & (music_arr <= 5)).astype(int)

    # Education vector
    edu_vec = work[educ_col].to_numpy()

    # Compute tolerance excluding each genre and fit models
    coef_map = {}
    mean_edu_likers = {}

    total_genres = len(categories)
    # Precompute dislike counts across all 18 for tolerance calculations
    dislikes_count_all = dislike_all.sum(axis=1)

    for j, genre in enumerate(categories):
        # Outcome: dislike of genre j
        y = dislike_all[:, j].astype(float)

        # Tolerance excluding j: 17 - dislikes among other 17
        dislikes_excl_j = dislikes_count_all - dislike_all[:, j]
        tol_excl = (total_genres - 1) - dislikes_excl_j  # 17 - dislikes among other 17

        # Prepare X with constant, tolerance, and education
        X = pd.DataFrame({
            'const': 1.0,
            'tolerance': tol_excl.astype(float),
            'education': edu_vec.astype(float)
        })

        # Drop any rows with missing (shouldn't have after filtering, but safe)
        valid_mask = np.isfinite(y) & np.isfinite(X['tolerance'].to_numpy()) & np.isfinite(X['education'].to_numpy())
        y_valid = y[valid_mask]
        X_valid = X.loc[valid_mask, ['const', 'tolerance', 'education']]

        if y_valid.size == 0 or y_valid.min() == y_valid.max():
            # Degenerate case: all 0/1; set coef to NaN
            coef_map[genre] = np.nan
        else:
            try:
                model = sm.Logit(y_valid, X_valid).fit(disp=0)
                coef_map[genre] = float(model.params['tolerance'])
            except PerfectSeparationError:
                # Fall back to regularized fit
                model = sm.Logit(y_valid, X_valid).fit_regularized(disp=0)
                coef_map[genre] = float(model.params['tolerance'])
            except Exception:
                # Generic fallback to regularized
                model = sm.Logit(y_valid, X_valid).fit_regularized(disp=0)
                coef_map[genre] = float(model.params['tolerance'])

        # Mean education among likers (responses 1 or 2)
        likers_mask = (music_arr[:, j] <= 2)
        if np.any(likers_mask):
            mean_edu_likers[genre] = float(np.nanmean(edu_vec[likers_mask]))
        else:
            mean_edu_likers[genre] = np.nan

    # Convert to aligned arrays in the specified order
    coef_series = pd.Series(coef_map).reindex(categories)
    edu_like_series = pd.Series(mean_edu_likers).reindex(categories)

    # If any mean edu is NaN (no likers), impute with sample mean to avoid plotting gaps
    sample_mean_edu = float(work[educ_col].mean())
    edu_like_series = edu_like_series.fillna(sample_mean_edu)

    # Sanity check lengths
    if not (len(coef_series) == len(edu_like_series) == len(categories)):
        raise ValueError("Series lengths mismatch; cannot plot.")

    # Build plot
    x = np.arange(len(categories))
    coef_vals = coef_series.to_numpy(dtype=float)
    edu_vals = edu_like_series.to_numpy(dtype=float)

    # Create output directory
    out_dir = os.path.join('.', 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'generated_results.jpg')

    # Plotting
    fig, ax = plt.subplots(figsize=(11, 6))
    ax2 = ax.twinx()

    # Lines: solid for coefficients (left axis), dash-dot for mean education (right axis)
    ax.plot(x, coef_vals, color='k', lw=2.2, ls='-')
    line2, = ax2.plot(x, edu_vals, color='k', lw=2.2, ls='-.')
    # Optionally lengthen dash-dot pattern to mimic print style
    try:
        line2.set_dashes([9, 4, 2, 4])
    except Exception:
        pass

    # Axes labels and limits
    ax.set_xlabel('Type of Music')
    ax.set_ylabel("Coefficients for Musical Tolerance as It Affects One's Probability of Disliking Each Music Genre")
    ax2.set_ylabel("Mean Educational Level of Respondents Who Reported Liking Each Music Genre")

    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])

    # Left-axis tick formatter to suppress leading zero (e.g., -.1)
    def _fmt(v, p):
        s = f"{v:.1f}"
        s = s.replace("-0.", "-.").replace("0.", ".")
        return s
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))

    # X tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=60, ha='right')

    # Sample mean education line on RIGHT axis
    ax2.axhline(sample_mean_edu, color='0.35', lw=1.2, ls=':', zorder=0)

    # Remove grid; adjust spines
    ax.grid(False)
    ax2.grid(False)
    for a in (ax, ax2):
        for sp in a.spines.values():
            sp.set_linewidth(1.2)
        a.tick_params(width=1.2, length=5, direction='out')

    # No title
    ax.set_title('')

    # Annotations with arrows (positions tuned but data-driven)
    # Coefficient annotation pointing to solid line
    try:
        idx_coef_annot = int(np.nanargmin(np.abs(x - len(categories)//4)))
        ax.annotate('Coefficient for Musical Tolerance',
                    xy=(idx_coef_annot, coef_vals[idx_coef_annot]),
                    xytext=(max(0, idx_coef_annot - 2), -0.44),
                    arrowprops=dict(arrowstyle='->', color='k'),
                    color='k', ha='left', va='center')
    except Exception:
        pass

    # Mean education annotation pointing to dash-dot line near middle
    try:
        mid_idx = len(categories)//2
        ax2.annotate('Mean Education of Genre Audience',
                     xy=(mid_idx, edu_vals[mid_idx]),
                     xytext=(max(0, mid_idx - 4), 14.25),
                     arrowprops=dict(arrowstyle='->', color='k'),
                     color='k', ha='left', va='center')
    except Exception:
        pass

    # Sample mean education annotation pointing to dotted line
    try:
        ax2.annotate('Sample Mean Education',
                     xy=(len(categories)*0.55, sample_mean_edu),
                     xytext=(len(categories)*0.62, sample_mean_edu + 0.22),
                     arrowprops=dict(arrowstyle='->', color='k'),
                     color='k', ha='left', va='bottom')
    except Exception:
        pass

    # Layout adjustments to prevent clipping
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.28, top=0.96)

    # Save and close
    fig.savefig(out_path, dpi=300, format='jpg')
    plt.close(fig)

    return os.path.abspath(out_path)