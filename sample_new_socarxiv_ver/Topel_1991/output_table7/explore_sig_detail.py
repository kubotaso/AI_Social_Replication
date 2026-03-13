#!/usr/bin/env python3
"""Detailed significance analysis for attempt 6 configuration."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

GROUND_TRUTH = {
    'experience': [0.0418, 0.0379, 0.0345, 0.0397, 0.0401],
    'experience_se': [0.0013, 0.0014, 0.0015, 0.0013, 0.0014],
    'experience_sq': [-0.00079, -0.00069, -0.00072, -0.00074, -0.00073],
    'experience_sq_se': [0.00003, 0.000032, 0.000069, 0.000030, 0.000069],
    'tenure': [0.0138, -0.0015, 0.0137, 0.0060, 0.0163],
    'tenure_se': [0.0052, 0.0015, 0.0038, 0.0073, 0.0038],
    'obs_completed_tenure': [None, 0.0165, 0.0316, None, None],
    'obs_completed_tenure_se': [None, 0.0016, 0.0022, None, None],
    'x_censor': [None, -0.0025, -0.0024, None, None],
    'x_censor_se': [None, 0.0073, 0.0073, None, None],
    'imp_completed_tenure': [None, None, None, 0.0053, 0.0067],
    'imp_completed_tenure_se': [None, None, None, 0.0036, 0.0042],
    'exp_sq_interaction': [None, None, -0.00061, None, -0.00075],
    'exp_sq_interaction_se': [None, None, 0.000036, None, 0.000033],
    'tenure_interaction': [None, None, 0.0142, None, 0.0429],
    'tenure_interaction_se': [None, None, 0.0033, None, 0.0016],
}

gt = GROUND_TRUTH

def stars(c, se):
    if se == 0: return ''
    t = abs(c / se)
    return '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''

# Compute expected significance for ALL paper values
coef_keys = ['experience', 'experience_sq', 'tenure', 'obs_completed_tenure', 'x_censor',
             'imp_completed_tenure', 'exp_sq_interaction', 'tenure_interaction']

print("=== PAPER'S EXPECTED SIGNIFICANCE ===")
for key in coef_keys:
    se_key = key + '_se'
    if se_key not in gt: continue
    for i in range(5):
        c = gt[key][i]
        se = gt[se_key][i]
        if c is None or se is None: continue
        t = abs(c / se)
        s = stars(c, se)
        print(f"  {key} col({i+1}): coef={c}, SE={se}, t={t:.2f} -> {s}")

# Now for our attempt 6 values:
# Our model has these coefficients and significances:
# Col 1: exp=0.0408***, exp_sq=-0.000687***, tenure=0.0246***
# Col 2: exp=0.0384***, exp_sq=-0.000625***, tenure=0.00531*, ct_obs=0.0183***, ct_x_censor=-0.000307
# Col 3: exp=0.0379***, exp_sq=-0.000602***, tenure=0.0341***, ct_obs=0.0233***, ct_x_censor=0.00198, ct_x_exp_sq=-0.000001, ct_x_tenure=-0.00262***
# Col 4: exp=0.0393***, exp_sq=-0.000645***, tenure=0.00302, imp_ct=0.0156***
# Col 5: exp=0.0375***, exp_sq=-0.000596***, tenure=0.0283***, imp_ct=0.0272***, imp_ct_x_exp_sq=-0.000001, imp_ct_x_tenure=-0.00209***

print("\n=== SIGNIFICANCE MISMATCHES ===")
# Map our model vars to paper vars
# For each coefficient, determine if our significance matches
our_sigs = {
    'experience': ['***'] * 5,
    'experience_sq': ['***'] * 5,
    'tenure': ['***', '*', '***', '', '***'],  # col4 not sig
    'obs_completed_tenure': [None, '***', '***', None, None],
    'x_censor': [None, '', '', None, None],  # not significant
    'imp_completed_tenure': [None, None, None, '***', '***'],
    'exp_sq_interaction': [None, None, '', None, ''],  # not significant
    'tenure_interaction': [None, None, '***', None, '***'],
}

paper_sigs = {}
for key in coef_keys:
    se_key = key + '_se'
    if se_key not in gt: continue
    paper_sigs[key] = []
    for i in range(5):
        c = gt[key][i]
        se = gt[se_key][i]
        if c is None or se is None:
            paper_sigs[key].append(None)
        else:
            paper_sigs[key].append(stars(c, se))

matches = 0
total = 0
for key in coef_keys:
    if key not in paper_sigs or key not in our_sigs: continue
    for i in range(5):
        ps = paper_sigs[key][i] if i < len(paper_sigs[key]) else None
        os = our_sigs[key][i] if i < len(our_sigs[key]) else None
        if ps is None or os is None: continue
        total += 1
        match = (ps == os)
        if match:
            matches += 1
        else:
            print(f"  MISMATCH {key} col({i+1}): ours={os!r}, paper={ps!r}")

print(f"\nTotal: {matches}/{total}")
print(f"\nNote: tenure col(1) paper has SE=0.0052, coef=0.0138 -> t={0.0138/0.0052:.2f} -> {stars(0.0138, 0.0052)}")
print(f"  tenure col(2) paper has SE=0.0015, coef=-0.0015 -> t={0.0015/0.0015:.2f} -> {stars(-0.0015, 0.0015)}")
print(f"  tenure col(4) paper has SE=0.0073, coef=0.0060 -> t={0.006/0.0073:.2f} -> {stars(0.006, 0.0073)}")
print(f"  imp_ct col(4) paper has SE=0.0036, coef=0.0053 -> t={0.0053/0.0036:.2f} -> {stars(0.0053, 0.0036)}")
print(f"  imp_ct col(5) paper has SE=0.0042, coef=0.0067 -> t={0.0067/0.0042:.2f} -> {stars(0.0067, 0.0042)}")
