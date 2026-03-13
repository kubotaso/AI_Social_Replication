import pandas as pd
import numpy as np

df = pd.read_csv('anes_cumulative.csv', low_memory=False)

# Check VCF0113 coding
print("VCF0113 value counts (all years):")
print(df['VCF0113'].value_counts().sort_index())
print()

# Check for a specific year
for year in [1952, 1988]:
    year_df = df[df['VCF0004'] == year]
    voters = year_df[year_df['VCF0704a'].isin([1, 2])]
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]
    white_voters = voters[voters['VCF0105a'] == 1]

    nonsouth = white_voters[white_voters['VCF0113'] == 2]
    south = white_voters[white_voters['VCF0113'] == 1]

    print(f"\n{year}:")
    print(f"  White voters: {len(white_voters)}")
    print(f"  White Non-South (VCF0113=2): {len(nonsouth)}")
    print(f"  White South (VCF0113=1): {len(south)}")
    print(f"  VCF0113 values: {white_voters['VCF0113'].value_counts().sort_index().to_dict()}")

    # Check party ID distribution for each group
    if len(nonsouth) > 0:
        ns_pid = nonsouth['VCF0301'].value_counts().sort_index()
        print(f"  Non-South PID dist: {ns_pid.to_dict()}")
    if len(south) > 0:
        s_pid = south['VCF0301'].value_counts().sort_index()
        print(f"  South PID dist: {s_pid.to_dict()}")

    # Check vote distribution for each group
    if len(nonsouth) > 0:
        ns_vote = nonsouth['VCF0704a'].value_counts().sort_index()
        print(f"  Non-South vote: {ns_vote.to_dict()}")
    if len(south) > 0:
        s_vote = south['VCF0704a'].value_counts().sort_index()
        print(f"  South vote: {s_vote.to_dict()}")
