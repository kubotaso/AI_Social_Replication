"""Test score impact of using full PID dummies as IV instruments."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

GROUND_TRUTH = {
    '1960': {
        'N': 911,
        'current': {'strong': (1.358, 0.094), 'weak': (1.028, 0.083), 'lean': (0.855, 0.131), 'intercept': (0.035, 0.053), 'llf': -372.7, 'r2': 0.41},
        'lagged': {'strong': (1.363, 0.092), 'weak': (0.842, 0.078), 'lean': (0.564, 0.125), 'intercept': (0.068, 0.051), 'llf': -403.9, 'r2': 0.36},
        'iv': {'strong': (1.715, 0.173), 'weak': (0.728, 0.239), 'lean': (1.081, 0.696), 'intercept': (0.032, 0.057), 'llf': -403.9, 'r2': 0.36},
    },
    '1976': {
        'N': 682,
        'current': {'strong': (1.087, 0.105), 'weak': (0.624, 0.086), 'lean': (0.622, 0.110), 'intercept': (-0.123, 0.054), 'llf': -358.2, 'r2': 0.24},
        'lagged': {'strong': (0.966, 0.104), 'weak': (0.738, 0.089), 'lean': (0.486, 0.109), 'intercept': (-0.063, 0.053), 'llf': -371.3, 'r2': 0.21},
        'iv': {'strong': (1.123, 0.178), 'weak': (0.745, 0.251), 'lean': (0.725, 0.438), 'intercept': (-0.102, 0.055), 'llf': -371.3, 'r2': 0.21},
    },
    '1992': {
        'N': 760,
        'current': {'strong': (0.975, 0.094), 'weak': (0.627, 0.084), 'lean': (0.472, 0.098), 'intercept': (-0.211, 0.051), 'llf': -408.2, 'r2': 0.20},
        'lagged': {'strong': (1.061, 0.100), 'weak': (0.404, 0.077), 'lean': (0.519, 0.101), 'intercept': (-0.168, 0.051), 'llf': -416.2, 'r2': 0.19},
        'iv': {'strong': (1.516, 0.180), 'weak': (-0.225, 0.268), 'lean': (1.824, 0.513), 'intercept': (-0.125, 0.053), 'llf': -416.2, 'r2': 0.19},
    }
}

# Score IV coefficients for standard vs full PID dummies approaches

# Standard approach (current attempt 4 results):
iv_standard = {
    '1960': {'strong': 1.827, 'weak': 0.155, 'lean': 1.888, 'intercept': 0.063},
    '1976': {'strong': 1.351, 'weak': 0.602, 'lean': 0.833, 'intercept': -0.087},
    '1992': {'strong': 1.514, 'weak': -0.060, 'lean': 1.493, 'intercept': -0.148},
}

# Full PID dummies approach:
iv_fullpid = {
    '1960': {'strong': 1.853, 'weak': 0.149, 'lean': 1.804, 'intercept': 0.073},
    '1976': {'strong': 1.315, 'weak': 0.709, 'lean': 0.720, 'intercept': -0.085},
    '1992': {'strong': 1.511, 'weak': -0.088, 'lean': 1.610, 'intercept': -0.135},
}

for approach_name, iv_vals in [('Standard 2SLS', iv_standard), ('Full PID dummies', iv_fullpid)]:
    total_score = 0
    n_coefs = 0
    print(f'\n=== {approach_name} ===')
    for year in ['1960', '1976', '1992']:
        gt = GROUND_TRUTH[year]['iv']
        for vk in ['strong', 'weak', 'lean', 'intercept']:
            gc, gs = gt[vk]
            gen = iv_vals[year][vk]
            diff = abs(gen - gc)
            score = max(0, 1.0 - diff / 0.05) if diff <= 0.15 else 0.0
            total_score += score
            n_coefs += 1
            if score > 0:
                print(f'  {year} {vk}: gen={gen:.3f} vs {gc:.3f}, diff={diff:.3f}, score={score:.3f}')
            else:
                print(f'  {year} {vk}: gen={gen:.3f} vs {gc:.3f}, diff={diff:.3f}, ZERO')

    print(f'  IV coef subtotal: {total_score:.3f}/{n_coefs}')
    # This is 12 out of 36 total coefficients
    print(f'  Impact on 30-point coef score: {total_score/36*30:.2f} contribution from IV')
