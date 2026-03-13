import pandas as pd
import math
evs = pd.read_stata('/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta',
                     convert_categoricals=False, columns=['S002EVS','S003','S003A','F001','S017','S020'])
w1 = evs[evs['S002EVS']==1]

# Tricky cases: Canada, Ireland, Netherlands, United States, West Germany
tricky = {124:'Canada', 372:'Ireland', 528:'Netherlands', 840:'United States', 276:'West Germany'}
gt = {'Canada':38, 'Ireland':25, 'Netherlands':21, 'United States':48, 'West Germany':29}

for s003, name in tricky.items():
    sub = w1[w1['S003']==s003]
    f = sub['F001']; valid = f[f>0]
    mask = f>0; f2=f[mask]; w2=sub['S017'][mask].fillna(1)
    uw = (valid==1).mean()*100
    wt = ((f2==1)*w2).sum()/w2.sum()*100
    paper = gt[name]
    print(f'{name}: uw={uw:.4f} wt={wt:.4f} paper={paper}')
    print(f'  round(uw)={round(uw)} int(uw)={int(uw)} round(wt)={round(wt)} int(wt)={int(wt)}')
    print(f'  ceil(uw)={math.ceil(uw)} ceil(wt)={math.ceil(wt)}')
    # Try S003A for sub-regions
    print(f'  S003A values: {sorted(sub["S003A"].unique())}')
    for sa in sorted(sub['S003A'].unique()):
        if sa > 0:
            ss = sub[sub['S003A']==sa]
            fv = ss['F001'][ss['F001']>0]
            if len(fv) > 0:
                print(f'    S003A={sa}: n={len(fv)}, uw={100*(fv==1).mean():.1f}')
    print()
