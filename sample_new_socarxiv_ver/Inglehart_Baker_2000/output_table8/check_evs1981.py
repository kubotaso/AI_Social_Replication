import pandas as pd
evs = pd.read_stata('/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta',
                     convert_categoricals=False, columns=['S002EVS','S003','S003A','F001','S017','S020'])
w1 = evs[evs['S002EVS']==1]

code_map = {56:'Belgium',124:'Canada',208:'Denmark',250:'France',276:'West Germany',
            352:'Iceland',372:'Ireland',380:'Italy',470:'Malta',528:'Netherlands',
            578:'Norway',724:'Spain',752:'Sweden',826:'Great Britain',840:'United States',
            909:'Northern Ireland'}

gt = {'Belgium':22,'Canada':38,'France':36,'West Germany':29,'Great Britain':34,
      'Iceland':39,'Ireland':25,'Northern Ireland':29,'Italy':36,'Netherlands':21,
      'Norway':26,'Spain':24,'Sweden':20,'United States':48}

print(f'{"Country":<22s} {"S003":>5s} {"n":>5s} {"uw":>7s} {"S017":>7s} {"paper":>6s} {"uw_r":>5s} {"wt_r":>5s} {"match":>8s}')
for s003, name in sorted(code_map.items(), key=lambda x:x[1]):
    sub = w1[w1['S003']==s003]
    f = sub['F001']
    valid = f[f>0]
    if len(valid)==0:
        print(f'{name:<22s} {s003:>5d} {len(sub):>5d} no F001 data')
        continue
    uw = (valid==1).mean()*100
    mask = f>0; f2=f[mask]; w2=sub['S017'][mask].fillna(1)
    wt = ((f2==1)*w2).sum()/w2.sum()*100
    paper = gt.get(name)
    uw_r = round(uw); wt_r = round(wt)
    best = 'NONE'
    if uw_r == paper: best = 'uw_round'
    elif wt_r == paper: best = 'wt_round'
    elif int(uw) == paper: best = 'uw_floor'
    elif int(wt) == paper: best = 'wt_floor'
    elif paper and abs(uw_r-paper) <= 2: best = f'uw_partial({uw_r-paper:+d})'
    elif paper and abs(wt_r-paper) <= 2: best = f'wt_partial({wt_r-paper:+d})'
    print(f'{name:<22s} {s003:>5d} {len(valid):>5d} {uw:>7.2f} {wt:>7.2f} {str(paper):>6s} {uw_r:>5d} {wt_r:>5d} {best:>12s}')
