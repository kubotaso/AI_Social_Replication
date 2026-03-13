#!/usr/bin/env python3
"""Analyze zone membership vs data availability."""

zones = {
    'Ex-Communist': ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                     'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                     'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB'],
    'Protestant Europe': ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE', 'DEU'],
    'English-speaking': ['AUS', 'CAN', 'GBR', 'NIR', 'IRL', 'NZL', 'USA'],
    'Latin America': ['ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN'],
    'Africa': ['GHA', 'NGA', 'ZAF'],
    'South Asia': ['BGD', 'IND', 'PAK', 'PHL', 'TUR'],
    'Orthodox': ['ARM', 'AZE', 'BLR', 'BGR', 'GEO', 'MDA', 'ROU', 'RUS', 'UKR', 'SRB', 'MKD'],
    'Confucian': ['CHN', 'JPN', 'KOR', 'TWN'],
}

all_zoned = set()
for z, countries in zones.items():
    all_zoned.update(countries)
print(f'Total unique countries in zones: {len(all_zoned)}')
print(sorted(all_zoned))

# Complete data countries (57)
complete = ['ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BEL', 'BGD', 'BGR', 'BIH', 'BLR', 'BRA',
            'CAN', 'CHE', 'CHL', 'CHN', 'COL', 'CZE', 'DEU', 'DNK', 'DOM', 'ESP', 'EST',
            'FIN', 'FRA', 'GBR', 'GEO', 'HUN', 'IND', 'IRL', 'ISL', 'ITA', 'JPN', 'KOR',
            'LTU', 'LVA', 'MDA', 'MEX', 'MKD', 'NLD', 'NOR', 'NZL', 'PER', 'PHL', 'POL',
            'PRI', 'PRT', 'ROU', 'RUS', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR', 'URY', 'USA',
            'VEN', 'ZAF']

not_zoned = [c for c in complete if c not in all_zoned]
print(f'\nNot in any zone ({len(not_zoned)}): {not_zoned}')

in_zones_missing = [c for c in all_zoned if c not in complete]
print(f'In zones but missing data ({len(in_zones_missing)}): {in_zones_missing}')

# If we drop non-zoned countries
remaining = [c for c in complete if c in all_zoned]
print(f'\nCountries in zones with complete data: {len(remaining)}')

# NIR likely doesn't have its own WB data
no_wb = ['NIR', 'TWN', 'PAK', 'NGA']
# Actually the factor scores already showed NIR, TWN are in factor data but
# maybe not in WB. Let's check: TWN = not in WB (not UN member).
# NIR = part of GBR in WB. PAK, NGA = in WB.
# SRB = may not be in WB as separate (was Yugoslavia)
print(f'\nLikely missing from WB: TWN (not UN member), NIR (part of GBR)')
print(f'SRB may also be missing from WB')

# The 57 complete list doesn't include: NIR, TWN, PAK, NGA, SRB, BGD
# Wait, PAK is in the list. Let me recheck.
# From factor output: PAK not in complete list?
# Let me check again...
# Complete list from the previous output: PAK is NOT in the list
# BGD IS in the list
# NGA IS NOT in the list... wait, NGA is at position 36...
# Let me just check which zone countries are missing
for z, countries in zones.items():
    missing = [c for c in countries if c not in complete]
    if missing:
        print(f'{z}: missing {missing}')
