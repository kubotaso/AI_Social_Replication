"""Explore alternative columns in the dataset."""
import pandas as pd
df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')

# Check alternative rate columns
for col in ['tbill_3m', 'tbill_6m', 'treasury_1y', 'treasury_10y', 'ffbond', 'cpbill', 'drbond', 'cpbill_long', 'discount_rate']:
    if col in df.columns:
        vals = df.loc['1970-01':'1970-06', col]
        print(f"{col}: {vals.values[:6]}")
    else:
        print(f"{col}: NOT FOUND")

# Check correlation between tbill_3m and tbill_6m
sub = df.loc['1959-07':'1989-12']
print(f"\nCorrelation tbill_3m vs tbill_6m: {sub['tbill_3m'].corr(sub['tbill_6m']):.4f}")
print(f"Correlation treasury_10y vs treasury_1y: {sub['treasury_10y'].corr(sub['treasury_1y']):.4f}")

# Check if ffbond and drbond are spreads or levels
print(f"\nffbond sample: {df.loc['1980-01':'1980-06', 'ffbond'].values}")
print(f"drbond sample: {df.loc['1980-01':'1980-06', 'drbond'].values}")
print(f"funds_rate sample: {df.loc['1980-01':'1980-06', 'funds_rate'].values}")
print(f"treasury_10y sample: {df.loc['1980-01':'1980-06', 'treasury_10y'].values}")
print(f"discount_rate sample: {df.loc['1980-01':'1980-06', 'discount_rate'].values if 'discount_rate' in df.columns else 'N/A'}")

# Check unemp_rate vs unemp_male_2554
print(f"\nunemp_rate sample: {df.loc['1970-01':'1970-06', 'unemp_rate'].values}")
print(f"unemp_male_2554 sample: {df.loc['1970-01':'1970-06', 'unemp_male_2554'].values}")
