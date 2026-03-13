import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run_analysis(data_source):
    df = pd.read_csv(data_source)

    genre_cols = ['latin', 'jazz', 'blues', 'musicals', 'oldies', 'classicl',
                  'reggae', 'bigband', 'newage', 'opera', 'blugrass', 'folk',
                  'moodeasy', 'conrock', 'rap', 'hvymetal', 'country', 'gospel']

    display_labels = {
        'latin': 'Latin/Salsa', 'jazz': 'Jazz', 'blues': 'Blues/R&B',
        'musicals': 'Show Tunes', 'oldies': 'Oldies', 'classicl': 'Classical',
        'reggae': 'Reggae', 'bigband': 'Swing', 'newage': 'New Age/Space',
        'opera': 'Opera', 'blugrass': 'Bluegrass', 'folk': 'Folk',
        'moodeasy': 'Easy Listening', 'conrock': 'Pop/Rock',
        'rap': 'Rap', 'hvymetal': 'Heavy Metal', 'country': 'Country',
        'gospel': 'Gospel'
    }

    # Step 1: Listwise deletion - keep only respondents with all 18 genres rated 1-5
    df_analysis = df.copy()
    for col in genre_cols:
        df_analysis = df_analysis[df_analysis[col].between(1, 5)]
    df_analysis = df_analysis[df_analysis['educ'].notna()]
    df_analysis = df_analysis.reset_index(drop=True)

    print(f"Analysis sample size: {len(df_analysis)}")

    # Step 2: For each genre, run logistic regression
    tolerance_coeffs = {}
    tolerance_pvals = {}

    for genre in genre_cols:
        dv = (df_analysis[genre] >= 4).astype(int)
        other_genres = [g for g in genre_cols if g != genre]
        tolerance = sum((df_analysis[g] < 4).astype(int) for g in other_genres)
        educ = df_analysis['educ']
        X = pd.DataFrame({'tolerance': tolerance, 'educ': educ})
        X = sm.add_constant(X)
        model = sm.Logit(dv, X)
        result = model.fit(disp=0)
        tolerance_coeffs[genre] = result.params['tolerance']
        tolerance_pvals[genre] = result.pvalues['tolerance']

    # Step 3: Mean education of fans (rating 1 or 2) - use full sample
    df_fans = df.copy()
    df_fans = df_fans[df_fans['educ'].notna()]
    mean_educ_fans = {}
    for genre in genre_cols:
        fans = df_fans[df_fans[genre].isin([1, 2])]
        if len(fans) > 0:
            mean_educ_fans[genre] = fans['educ'].mean()
        else:
            mean_educ_fans[genre] = np.nan

    # Sample mean education (from analysis sample)
    sample_mean_educ = df_analysis['educ'].mean()

    # Step 4: Sort genres by tolerance coefficient (ascending = most negative first)
    sorted_genres = sorted(genre_cols, key=lambda g: tolerance_coeffs[g])

    # Print results
    print(f"\nSample mean education: {sample_mean_educ:.2f}")
    print("\n=== TOLERANCE COEFFICIENTS (sorted, most negative first) ===")
    print(f"{'Genre':<25} {'Coeff':>8} {'p-value':>12}")
    print("-" * 47)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {tolerance_coeffs[g]:>8.4f} {tolerance_pvals[g]:>12.6f}")

    print("\n=== MEAN EDUCATION OF FANS (in sorted order) ===")
    print(f"{'Genre':<25} {'Mean Educ':>10}")
    print("-" * 37)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {mean_educ_fans[g]:>10.2f}")

    # Step 5: Create figure matching the paper's style
    fig, ax1 = plt.subplots(figsize=(12, 7))

    x_pos = range(len(sorted_genres))
    labels = [display_labels[g] for g in sorted_genres]
    coeffs = [tolerance_coeffs[g] for g in sorted_genres]
    educ_vals = [mean_educ_fans[g] for g in sorted_genres]

    # Left axis: tolerance coefficients (solid line - matching paper)
    ax1.set_xlabel('Type of Music', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coefficients for Musical Tolerance as It Affects\nOne\'s Probability of Disliking Each Music Genre',
                    fontsize=10)
    line1, = ax1.plot(x_pos, coeffs, '-', color='black', linewidth=2, label='Coefficient for Musical Tolerance')
    ax1.tick_params(axis='y')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=60, ha='right', fontsize=9)

    # Set left Y-axis range to match paper: -0.5 to -0.1
    ax1.set_ylim(-0.50, -0.10)
    ax1.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax1.set_yticklabels(['-.5', '-.4', '-.3', '-.2', '-.1'])

    # Right axis: mean education of fans (dash-dot line - matching paper)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Educational Level of Respondents Who\nReported Liking Each Music Genre',
                    fontsize=10)
    line2, = ax2.plot(x_pos, educ_vals, '--', color='black', linewidth=2,
                       dashes=(8, 4), label='Mean Education of Genre Audience')
    ax2.tick_params(axis='y')

    # Set right Y-axis range to match paper: 12 to 15
    ax2.set_ylim(12, 15)

    # Horizontal reference line at sample mean education
    hline = ax2.axhline(y=sample_mean_educ, color='black', linestyle=':', linewidth=1,
                         label='Sample Mean Education')

    # Add annotation for Sample Mean Education (matching paper placement)
    ax2.annotate('Sample Mean Education', xy=(7, sample_mean_educ),
                 xytext=(7, sample_mean_educ - 0.3),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    # Add text labels for the lines (matching paper style)
    ax1.text(1.5, -0.47, 'Coefficient for Musical Tolerance', fontsize=9, fontweight='bold')
    ax2.text(2, 14.5, 'Mean Education of Genre Audience', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, 'generated_results_attempt_2.jpg')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', format='jpeg')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return {
        'tolerance_coeffs': {display_labels[g]: tolerance_coeffs[g] for g in sorted_genres},
        'mean_educ_fans': {display_labels[g]: mean_educ_fans[g] for g in sorted_genres},
        'genre_order': [display_labels[g] for g in sorted_genres],
        'sample_mean_educ': sample_mean_educ,
        'n': len(df_analysis)
    }


def score_against_ground_truth(result):
    """Compare generated results against paper's figure values."""
    # Ground truth read from original figure (corrected after viewing image)
    paper_order = [
        'Latin/Salsa', 'Jazz', 'Blues/R&B', 'Show Tunes', 'Oldies',
        'Classical', 'Reggae', 'New Age/Space', 'Swing', 'Opera',
        'Bluegrass', 'Folk', 'Easy Listening', 'Pop/Rock',
        'Rap', 'Heavy Metal', 'Country', 'Gospel'
    ]

    # Approximate coefficients read from the original figure image
    paper_coeffs = {
        'Latin/Salsa': -0.43, 'Jazz': -0.40, 'Blues/R&B': -0.36,
        'Show Tunes': -0.35, 'Oldies': -0.35, 'Classical': -0.34,
        'Reggae': -0.32, 'New Age/Space': -0.23, 'Swing': -0.31,
        'Opera': -0.29, 'Bluegrass': -0.29, 'Folk': -0.27,
        'Easy Listening': -0.27, 'Pop/Rock': -0.25,
        'Rap': -0.19, 'Heavy Metal': -0.16, 'Country': -0.16,
        'Gospel': -0.13
    }

    paper_educ = {
        'Latin/Salsa': 14.5, 'Jazz': 14.3, 'Blues/R&B': 13.5,
        'Show Tunes': 14.0, 'Oldies': 13.2, 'Classical': 14.5,
        'Reggae': 13.8, 'New Age/Space': 14.3, 'Swing': 13.7,
        'Opera': 12.7, 'Bluegrass': 13.2, 'Folk': 13.1,
        'Easy Listening': 13.0, 'Pop/Rock': 13.3,
        'Rap': 13.0, 'Heavy Metal': 13.2, 'Country': 14.5,
        'Gospel': 14.8
    }

    print("\n=== SCORING AGAINST GROUND TRUTH ===")

    # Check genre order match
    my_order = result['genre_order']
    order_matches = sum(1 for a, b in zip(my_order, paper_order) if a == b)
    print(f"\nGenre order matches: {order_matches}/18")
    print(f"My order:    {my_order}")
    print(f"Paper order: {paper_order}")

    # Check coefficient values
    print(f"\n{'Genre':<25} {'My Coeff':>10} {'Paper Coeff':>12} {'Diff':>8}")
    print("-" * 57)
    total_coeff_diff = 0
    for genre in my_order:
        my_val = result['tolerance_coeffs'][genre]
        paper_val = paper_coeffs.get(genre, None)
        if paper_val is not None:
            diff = abs(my_val - paper_val)
            total_coeff_diff += diff
            print(f"{genre:<25} {my_val:>10.4f} {paper_val:>12.2f} {diff:>8.4f}")

    avg_coeff_diff = total_coeff_diff / 18
    print(f"\nAverage absolute coefficient difference: {avg_coeff_diff:.4f}")

    # Check education values
    print(f"\n{'Genre':<25} {'My Educ':>10} {'Paper Educ':>12} {'Diff':>8}")
    print("-" * 57)
    total_educ_diff = 0
    for genre in my_order:
        my_val = result['mean_educ_fans'][genre]
        paper_val = paper_educ.get(genre, None)
        if paper_val is not None:
            diff = abs(my_val - paper_val)
            total_educ_diff += diff
            print(f"{genre:<25} {my_val:>10.2f} {paper_val:>12.1f} {diff:>8.2f}")

    avg_educ_diff = total_educ_diff / 18
    print(f"\nAverage absolute education difference: {avg_educ_diff:.2f}")


if __name__ == "__main__":
    result = run_analysis("gss1993_clean.csv")

    print("\n=== SUMMARY ===")
    print(f"N = {result['n']}")
    print(f"Sample mean education = {result['sample_mean_educ']:.2f}")
    print(f"Genre order: {result['genre_order']}")

    score_against_ground_truth(result)
