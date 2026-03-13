#!/usr/bin/env python3
import pandas as pd

# The figure shows 38 societies. Let me list what I can see from the figure:
# From the original figure, I can identify these country-timepoints:
#
# Societies with arrows (initial -> latest):
# China 90 -> China 95
# Bulgaria 90 -> Bulg. 97
# Estonia 90 -> Estonia 96
# Russia 90 -> Russia 96
# Belarus 90 -> Belarus 96
# Latvia 90 -> Latvia 96
# Lithuania 90 -> Lithuania (not labeled clearly)
# Slovenia 90 -> Slovenia 95
# Hungary 81 -> Hungary (98?)
# Poland 90 -> Poland 97
# Japan 81 -> Japan 95
# S. Korea 81 -> S. Korea 96
# East Germany 90 -> East Germany 97
# West Germany 81 -> West Germany 97
# Sweden 81 -> Sweden 96 (but we have SWE in wave 2 from 1990)
# Norway 81 -> Norway 96
# Finland 81 -> Finland 96
# Netherlands 81 -> Netherlands 90
# Switzerland 90 -> Switzerland 96
# France 81 -> France 90
# Belgium 81 -> Belgium 90
# Italy 81 -> Italy 90
# Spain 81 -> Spain 95
# Britain 81 -> Britain 98
# Ireland 81 -> Ireland 90
# N. Ireland 81 -> N. Ireland 90
# Iceland 81 -> Iceland 90
# Canada 81 -> Canada 90
# Australia 81 -> Australia 95
# U.S.A 81 -> U.S.A 95
# Argentina 81 -> Argentina 95
# Brazil 90 -> Brazil 97
# Mexico 81 -> Mexico 96
# Chile 90 -> Chile 96
# Turkey 90 -> Turkey 97
# South Africa 81 -> South Africa 96
# India 90 -> India 96
# Nigeria 90 -> Nigeria 95

# That's 38 societies. Note many have "81" as initial, coming from EVS 1981 or WVS wave 1.
# But we only have EVS 1990 data, not EVS 1981.
# Our available multi-wave countries from WVS+EVS:
# Wave 1 (1981): ARG, AUS, FIN, HUN, JPN, KOR, MEX, ZAF (8 countries)
# Wave 2 (1990): many countries from WVS + EVS
# Wave 3 (1995-98): many countries from WVS

# For the "81" countries: Norway 81, Sweden 81, Netherlands 81, France 81, Belgium 81, Italy 81,
# Britain 81, Ireland 81, N.Ireland 81, Iceland 81, Canada 81, Spain 81 all have "81" as initial.
# These come from EVS 1981 wave, which we DON'T have.
# However, we do have EVS 1990 data for: NOR, SWE, NLD, FRA, BEL, ITA, GBR, IRL, NIR, ISL, CAN, ESP

# For the paper's figure:
# Countries with "81" initial that we have wave 1 WVS: ARG(84), AUS(81), FIN(81), HUN(82), JPN(81), KOR(82), MEX(81), ZAF(82)
# Countries with "81" initial from EVS 1981 (not available): NOR, SWE, NLD, FRA, BEL, ITA, GBR, IRL, NIR, ISL, CAN, ESP, West Germany

# Plan: For countries where we have wave 1 WVS data, use wave 1 as initial.
#        For EVS-only countries, we only have wave 2 (1990), so we can't show an arrow.
#        UNLESS we use wave 2 (1990) as initial and wave 3 as final.
#        But for France, Belgium, Italy, Netherlands, Iceland, Ireland, N.Ireland, Canada:
#        these don't have WVS wave 3 data!

# Wait - let me re-read the figure. Countries like "France 81" -> "France 90" means:
# the arrow goes FROM the 1981 position TO the 1990 position. Both points are shown.
# For France, the initial is EVS 1981 and final is EVS 1990.
# Since we only have EVS 1990, we can only show the 1990 endpoint, not the arrow.

# Similarly "Netherlands 81" -> "Netherlands 90": EVS 1981 -> EVS 1990.

# So with our available data, we can only show arrows for countries that have:
# - WVS wave 1 + WVS wave 2 or 3
# - WVS wave 2 + WVS wave 3
# - EVS wave 2 + WVS wave 3

# That gives us far fewer than 38 countries with arrows.
# BUT: the instruction says to compute factor scores at each time point.
# Let's work with what we have and show arrows where possible.

# Let me check: for the EVS countries, do any also have WVS wave 3 data?
print("Checking EVS-only countries for WVS wave 3:")
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','S020'], low_memory=False)
wvs3 = wvs[wvs['S002VS']==3]
evs_countries = ['NOR','SWE','FRA','BEL','ITA','GBR','IRL','NIR','ISL','CAN','NLD','ESP','DEU','FIN','DNK']
for ca in evs_countries:
    w3 = wvs3[wvs3['COUNTRY_ALPHA']==ca]
    if len(w3) > 0:
        print(f'  {ca}: WVS wave 3, years={sorted(w3.S020.unique())}')
    else:
        print(f'  {ca}: NO WVS wave 3')

# What about Sweden and Norway in WVS?
print("\nSweden and Norway in all WVS waves:")
for ca in ['SWE', 'NOR']:
    sub = wvs[wvs['COUNTRY_ALPHA']==ca]
    for w in sorted(sub['S002VS'].unique()):
        years = sorted(sub[sub['S002VS']==w]['S020'].unique())
        print(f'  {ca} wave {int(w)}: {[int(y) for y in years]}, n={len(sub[sub["S002VS"]==w])}')
