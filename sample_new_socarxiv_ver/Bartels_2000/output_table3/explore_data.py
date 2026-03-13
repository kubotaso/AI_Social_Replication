import pandas as pd

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2/anes_cumulative.csv',
                 usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)

years = [1970, 1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996]
paper_n = [683, 798, 1079, 1009, 859, 712, 1185, 981, 1054, 801, 1370, 942, 1031]

dem_inc = [12, 13, 14, 19]
rep_inc = [21, 23, 24, 29]
open_seat = list(range(40, 60))
valid_inc = dem_inc + rep_inc + open_seat

print("Checking N values under different filtering strategies:")
print(f"{'Year':>4} {'Paper':>6} {'pid+validinc':>13} {'pid+notnull':>12} {'nopid+validinc':>15} {'nopid+notnull':>14}")
for i, y in enumerate(years):
    sub = df[df['VCF0004'] == y].copy()
    sub = sub[sub['VCF0707'].isin([1, 2])]
    sub_pid = sub[sub['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]
    sub_inc = sub_pid[sub_pid['VCF0902'].isin(valid_inc)]
    sub_inc2 = sub_pid[sub_pid['VCF0902'].notna()]
    sub_inc3 = sub[sub['VCF0902'].isin(valid_inc)]
    sub_inc4 = sub[sub['VCF0902'].notna()]
    print(f'{y:>4} {paper_n[i]:>6} {len(sub_inc):>13} {len(sub_inc2):>12} {len(sub_inc3):>15} {len(sub_inc4):>14}')

# Check what non-standard VCF0902 codes exist per year for voters
print("\nNon-standard VCF0902 codes (not in dem_inc, rep_inc, open_seat 40-59) per year:")
for i, y in enumerate(years):
    sub = df[(df['VCF0004'] == y) & (df['VCF0707'].isin([1, 2]))].copy()
    all_codes = sub['VCF0902'].dropna().unique()
    non_standard = [c for c in sorted(all_codes) if int(c) not in valid_inc]
    if non_standard:
        print(f"  {y}: {non_standard}")

# For years where nopid+notnull matches paper N best
print("\nDetailed check for years with discrepancy:")
for i, y in enumerate(years):
    sub = df[(df['VCF0004'] == y) & (df['VCF0707'].isin([1, 2]))].copy()
    n_vote = len(sub)
    sub_pid = sub[sub['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]
    n_pid = len(sub_pid)
    sub_notnull = sub[sub['VCF0902'].notna()]
    n_notnull = len(sub_notnull)
    sub_pid_notnull = sub_pid[sub_pid['VCF0902'].notna()]
    n_pid_notnull = len(sub_pid_notnull)

    if n_vote != paper_n[i]:
        print(f"  {y}: vote={n_vote}, paper={paper_n[i]}, diff={n_vote - paper_n[i]}")
        print(f"    After PID filter: {n_pid}")
        print(f"    After notnull inc: {n_notnull}")
        print(f"    After both: {n_pid_notnull}")
