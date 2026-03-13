import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("gss1993_clean.csv")

# The paper says: "leaving 912 valid cases. The exclusiveness scale has a range
# of 0 to 5, a mean of 2.65, a standard deviation of 1.56, and an alpha of .54"
#
# Wait -- the paper says "the exclusiveness scale" not "the racism scale"!
# Let me re-read: "I create the racism scale by collecting all the 1993 GSS
# questions about racial attitudes..."
# "leaving 912 valid cases"
# Then: "The exclusiveness scale has a range of 0 to 5, a mean of 2.65"
#
# Wait, maybe the 912 refers to something else? Let me check the
# musical exclusiveness scale (the DV) descriptives.

# The DV is "number of genres disliked" which has range 0-18
# The racism scale has range 0-5
# The paper says: "The exclusiveness scale has a range of 0 to 15"
# No wait -- re-reading: page 889 top, "ranges from 0 to 15, has a mean of 5.24"
# That's the political intolerance scale.
# Then: "the racism scale has a range of 0 to 5, a mean of 2.65"

# So the 912 refers to the racism scale having 912 valid cases.
# But we only get 708 with all 5 items valid...

# The key question: does person-mean imputation with min 4 give 912?
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

# Check various scales for getting N=912
items_sets = {
    'paper5_racdif3eq2': (['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3'],
                           {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 2}),
    'paper5_racdif3eq1': (['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3'],
                           {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 1}),
    'alt5_with_racdif4':  (['racmost', 'busing', 'racdif1', 'racdif3', 'racdif4'],
                            {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif3': 2, 'racdif4': 1}),
}

for scale_name, (items, rv) in items_sets.items():
    coded = []
    for item in items:
        col = f'r_{item}'
        df[col] = (df[item] == rv[item]).astype(float).where(df[item].isin([1,2]))
        coded.append(col)

    for min_valid in [3, 4, 5]:
        mask = df[coded].notna().sum(axis=1) >= min_valid
        valid_vals = []
        for idx in df.index:
            if not mask[idx]:
                valid_vals.append(np.nan)
                continue
            vals = [df.loc[idx, c] for c in coded]
            vv = [v for v in vals if not np.isnan(v)]
            if min_valid == len(items):  # all required, simple sum
                valid_vals.append(sum(vv))
            else:
                pm = np.mean(vv)
                total = sum(v if not np.isnan(v) else pm for v in vals)
                valid_vals.append(total)

        arr = np.array(valid_vals, dtype=float)
        n_valid = np.sum(~np.isnan(arr))
        if n_valid > 0:
            mean_v = np.nanmean(arr)
            sd_v = np.nanstd(arr, ddof=1)
            print(f"{scale_name:25s} min{min_valid}: N={int(n_valid):4d} mean={mean_v:.2f} SD={sd_v:.2f}")

# Maybe the paper used ALL the racdif items (1-4) plus racmost and busing = 6 items?
# And removed one via factor analysis leaving 5?
# If racdif4 was removed, that leaves racmost+busing+racdif1+racdif2+racdif3
# If racdif2 was removed, that leaves racmost+busing+racdif1+racdif3+racdif4
# etc.

# The paper says: "I remove questions with extremely small variances"
# Let's check variances of all racism items
print("\n=== Variance of racism items ===")
all_racism = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']
for item in all_racism:
    vals = pd.to_numeric(df[item], errors='coerce')
    valid = vals[vals.isin([1,2])]
    if len(valid) > 0:
        # Variance of dichotomous coding
        racist_map = {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2,
                      'racdif3': 2, 'racdif4': 1}
        coded_vals = (valid == racist_map.get(item, 1)).astype(float)
        p = coded_vals.mean()
        var = p * (1-p)
        print(f"  {item}: N={len(valid)}, p(racist)={p:.3f}, var={var:.4f}")

# What if there are MORE racism items in the GSS beyond racdif1-4?
# Let's check for racwork, racmar, racpush, racopen, etc.
print("\n=== Other potential racism variables ===")
for col in sorted(df.columns):
    if any(col.startswith(prefix) for prefix in ['rac', 'affirm', 'wrkway', 'closeblk', 'closewht']):
        vals = pd.to_numeric(df[col], errors='coerce')
        n = vals.notna().sum()
        if n > 100:
            print(f"  {col}: N={n}, unique={sorted(vals.dropna().unique())[:10]}")

# Maybe racpres or other items are also in the scale
print("\n=== Testing 6-item scale (all racdif + racmost + busing) ===")
items6 = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']
rv6 = {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 2, 'racdif4': 1}
coded = []
for item in items6:
    col = f'r_{item}'
    df[col] = (df[item] == rv6[item]).astype(float).where(df[item].isin([1,2]))
    coded.append(col)

for min_valid in [4, 5, 6]:
    mask = df[coded].notna().sum(axis=1) >= min_valid
    valid_vals = []
    for idx in df.index:
        if not mask[idx]:
            valid_vals.append(np.nan)
            continue
        vals = [df.loc[idx, c] for c in coded]
        vv = [v for v in vals if not np.isnan(v)]
        if min_valid == 6:
            valid_vals.append(sum(vv))
        else:
            pm = np.mean(vv)
            total = sum(v if not np.isnan(v) else pm for v in vals)
            valid_vals.append(total)

    arr = np.array(valid_vals, dtype=float)
    n_valid = np.sum(~np.isnan(arr))
    if n_valid > 0:
        mean_v = np.nanmean(arr)
        sd_v = np.nanstd(arr, ddof=1)
        print(f"  6-item min{min_valid}: N={int(n_valid):4d} mean={mean_v:.2f} SD={sd_v:.2f}")

# NEW IDEA: What about using racdif3==1 as racist but with PM imputation?
# The inter-item correlations were NEGATIVE for racdif3==1, but the paper
# explicitly says racdif3 is about "motivation" and "yes" (==1) is racist.
# Maybe the paper computed it this way despite negative correlations.
# Let's see what mean/SD we get:
print("\n=== 5-item with racdif3==1, PM imputation ===")
items5 = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']
rv5 = {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 1}
coded = []
for item in items5:
    col = f'r_{item}'
    df[col] = (df[item] == rv5[item]).astype(float).where(df[item].isin([1,2]))
    coded.append(col)

for min_valid in [3, 4, 5]:
    valid_vals = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded]
        n = sum(1 for v in vals if not np.isnan(v))
        if n >= min_valid:
            vv = [v for v in vals if not np.isnan(v)]
            if n == 5:
                valid_vals.append(sum(vv))
            else:
                pm = np.mean(vv)
                valid_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            valid_vals.append(np.nan)
    arr = np.array(valid_vals, dtype=float)
    n_valid = np.sum(~np.isnan(arr))
    mean_v = np.nanmean(arr)
    sd_v = np.nanstd(arr, ddof=1)
    print(f"  min{min_valid}: N={int(n_valid)} mean={mean_v:.2f} SD={sd_v:.2f}")
