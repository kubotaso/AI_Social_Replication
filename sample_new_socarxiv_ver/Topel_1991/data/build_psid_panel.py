#!/usr/bin/env python3
"""
Build PSID Panel Dataset for Topel (1991) Replication
=====================================================

Reads raw fixed-width PSID family files (1968-1983) and the cross-year
individual file, extracts relevant variables, merges across years to
create a person-year panel, and applies sample restrictions.

Column positions come from the .do files in each year's ZIP archive.
All positions are 1-based, inclusive.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/psid_raw"
OUTDIR = "/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data"

# CPS Real Wage Index from Table A1 of Topel (1991)
# Used to deflate nominal wages to a common real basis
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

###############################################################################
# VARIABLE DEFINITIONS BY YEAR
# Each entry: (variable_name, start_col, end_col)
# Columns are 1-based and inclusive as in the .do files.
# Python's pd.read_fwf uses 0-based tuples (start, end) where end is exclusive,
# so we convert: python_start = start_col - 1, python_end = end_col
###############################################################################

FAMILY_VARS = {
    1968: {
        'interview_number': ('V2', 2, 5),  # 4-digit ID to match individual file ER30001
        'age_head': ('V117', 283, 284),
        'sex_head': ('V119', 287, 287),
        'race': ('V181', 362, 362),
        'education': ('V313', 521, 521),
        'marital_status': ('V239', 438, 438),
        'labor_income': ('V74', 183, 187),
        'hourly_earnings': ('V337', 608, 612),  # in dollars (99.99 = top-code)
        'annual_hours': ('V47', 114, 117),
        'self_employed': ('V198', 388, 388),
        'occupation': ('V197_A', 382, 384),
        'industry': ('V197_B', 385, 387),
        'union': ('V294', 501, 501),
        'state': ('V93', 245, 246),
        'region': ('V361', 642, 642),
        'smsa': ('V188', 369, 371),
        'disability': ('V216', 409, 409),
        # No explicit employment status; use self_employed
        # No govt employee variable in 1968
        # SRC/SEO: interview numbers < 3000 are SRC, >= 5001 are SEO
    },
    1969: {
        'interview_number': ('V442', 2, 5),
        'age_head': ('V1008', 1060, 1061),
        'sex_head': ('V1010', 1063, 1063),
        'race': ('V801', 577, 577),
        'education': ('V794', 570, 570),
        'marital_status': ('V607', 347, 347),
        'labor_income': ('V514', 185, 189),
        'hourly_earnings': ('V871', 725, 729),  # in dollars (99.99 = top-code)
        'annual_hours': ('V465', 62, 65),
        'self_employed': ('V641', 393, 393),
        'occupation': ('V640_A', 387, 389),
        'industry': ('V640_B', 390, 392),
        'union': ('V766', 537, 537),
        'state': ('V537', 258, 259),
        'region': ('V876', 738, 738),
        'smsa': ('V808', 584, 586),
        'disability': ('V743', 514, 514),
        'employment_status': ('V639', 385, 385),
    },
    1970: {
        'interview_number': ('V1102', 2, 5),
        'age_head': ('V1239', 343, 344),
        'sex_head': ('V1240', 345, 345),
        'race': ('V1490', 678, 678),
        'education': ('V1485', 673, 673),
        'marital_status': ('V1365', 524, 524),
        'labor_income': ('V1196', 214, 218),
        'wages': ('V1191', 205, 209),
        'hourly_earnings': ('V1567', 848, 852),  # in dollars (99.99 = top-code)
        'annual_hours': ('V1138', 76, 79),
        'self_employed': ('V1280', 394, 394),
        'occupation': ('V1279_A', 388, 390),
        'industry': ('V1279_B', 391, 393),
        'union': ('V1434', 607, 607),
        'state': ('V1103', 6, 7),
        'region': ('V1572', 861, 861),
        'smsa': ('V1497', 685, 687),
        'disability': ('V1409', 579, 579),
        'employment_status': ('V1278', 386, 386),
    },
    1971: {
        'interview_number': ('V1802', 2, 5),
        'age_head': ('V1942', 347, 348),
        'sex_head': ('V1943', 349, 349),
        'race': ('V2202', 695, 695),
        'education': ('V2197', 690, 690),
        'marital_status': ('V2072', 535, 535),
        'labor_income': ('V1897', 213, 217),
        'wages': ('V1892', 204, 208),
        'hourly_earnings': ('V2279', 866, 870),  # in dollars (99.99 = top-code)
        'annual_hours': ('V1839', 76, 79),
        'self_employed': ('V1986', 402, 402),
        'occupation': ('V1984_A', 394, 396),
        'industry': ('V1985_A', 399, 401),
        'union': ('V2145', 623, 623),
        'state': ('V1803', 6, 7),
        'region': ('V2284', 879, 879),
        'smsa': ('V2209', 702, 704),
        'disability': ('V2121', 596, 596),
        'employment_status': ('V1983', 392, 392),
    },
    1972: {
        'interview_number': ('V2402', 2, 5),
        'age_head': ('V2542', 350, 351),
        'sex_head': ('V2543', 352, 352),
        'race': ('V2828', 728, 728),
        'education': ('V2823', 723, 723),
        'marital_status': ('V2670', 538, 538),
        'labor_income': ('V2498', 216, 220),
        'wages': ('V2493', 207, 211),
        'hourly_earnings': ('V2906', 899, 903),  # in dollars (99.99 = top-code)
        'annual_hours': ('V2439', 77, 80),
        'self_employed': ('V2584', 404, 404),
        'occupation': ('V2582_A', 396, 398),
        'industry': ('V2583_A', 401, 403),
        'union': ('V2787', 671, 671),
        'state': ('V2403', 6, 7),
        'region': ('V2911', 912, 912),
        'smsa': ('V2835', 735, 737),
        'disability': ('V2718', 599, 599),
        'employment_status': ('V2581', 394, 394),
    },
    1973: {
        'interview_number': ('V3002', 2, 5),
        'age_head': ('V3095', 235, 236),
        'sex_head': ('V3096', 237, 237),
        'race': ('V3300', 562, 562),
        'education': ('V3241', 466, 466),
        'marital_status': ('V3181', 373, 373),
        'labor_income': ('V3051', 101, 105),
        'hourly_earnings': ('V3275', 520, 524),  # in dollars (99.99 = top-code)
        'annual_hours': ('V3027', 50, 53),
        'occupation': ('V3115_A', 260, 262),
        'industry': ('V3116_A', 265, 267),
        'state': ('V3003', 6, 7),
        'region': ('V3279', 532, 532),
        'smsa': ('V3250', 475, 477),
        'employment_status': ('V3114', 258, 258),
    },
    1974: {
        'interview_number': ('V3402', 2, 5),
        'age_head': ('V3508', 269, 270),
        'sex_head': ('V3509', 271, 271),
        'race': ('V3720', 613, 613),
        'education': ('V3663', 519, 519),
        'marital_status': ('V3598', 421, 421),
        'labor_income': ('V3463', 131, 135),
        'wages': ('V3458', 122, 126),
        'hourly_earnings': ('V3695', 571, 575),  # in dollars (99.99 = top-code)
        'annual_hours': ('V3423', 37, 40),
        'self_employed': ('V3532', 310, 310),
        'occupation': ('V3530_A', 302, 304),
        'industry': ('V3531_A', 307, 309),
        'state': ('V3403', 6, 7),
        'region': ('V3699', 583, 583),
        'smsa': ('V3672', 528, 530),
        'employment_status': ('V3528', 297, 297),
    },
    1975: {
        'interview_number': ('V3802', 2, 5),
        'age_head': ('V3921', 317, 318),
        'sex_head': ('V3922', 319, 319),
        'race': ('V4204', 730, 730),
        'education': ('V4198', 722, 722),
        'labor_income': ('V3863', 128, 132),
        'wages': ('V3858', 119, 123),
        'hourly_earnings': ('V4174', 684, 688),  # in dollars (99.99 = top-code)
        'annual_hours': ('V3823', 38, 41),
        'occupation': ('V3968_A', 377, 379),
        'industry': ('V3969_A', 382, 384),
        'union': ('V4087', 558, 558),
        'state': ('V3803', 6, 7),
        'region': ('V4178', 696, 696),
        'smsa': ('V3933', 337, 339),
        'employment_status': ('V3967', 375, 375),
        'govt_employee': ('V3971', 386, 386),
    },
    1976: {
        'interview_number': ('V4302', 2, 5),
        'age_head': ('V4436', 409, 410),
        'sex_head': ('V4437', 411, 411),
        'race': ('V5096', 1472, 1472),
        'education': ('V5074', 1438, 1438),
        'marital_status': ('V4603', 662, 662),
        'labor_income': ('V5031', 1366, 1370),
        'wages': ('V4373', 162, 166),
        'hourly_earnings': ('V5050', 1400, 1404),  # in dollars (99.99 = top-code)
        'annual_hours': ('V4332', 60, 63),
        'occupation': ('V4459_A', 443, 445),
        'industry': ('V4460_A', 448, 450),
        'union': ('V4478', 472, 472),
        'state': ('V4303', 6, 7),
        'region': ('V5054', 1412, 1412),
        'employment_status': ('V4458', 440, 440),
        'tenure_months': ('V4871', 1120, 1121),
        'mo_started_job': ('V4489', 487, 488),
        'same_job': ('V4652', 739, 739),
    },
    1977: {
        'interview_number': ('V5202', 2, 5),
        'age_head': ('V5350', 448, 449),
        'sex_head': ('V5351', 450, 450),
        'race': ('V5662', 934, 934),
        'education': ('V5647', 917, 917),
        'marital_status': ('V5650', 922, 922),
        'labor_income': ('V5627', 868, 872),
        'wages': ('V5283', 192, 196),
        # No explicit hourly_earnings in 1977; compute from wages/hours
        'annual_hours': ('V5232', 60, 63),
        'occupation': ('V5374_A', 482, 484),
        'industry': ('V5375_A', 487, 489),
        'union': ('V5559', 770, 770),
        'state': ('V5203', 6, 7),  # "CURRENT STATE 1977"
        'smsa': ('V5206', 10, 10),
        'employment_status': ('V5373', 479, 479),
        'mo_started_job': ('V5398', 522, 523),
        'govt_employee': ('V5377', 491, 491),
    },
    1978: {
        'interview_number': ('V5702', 2, 5),
        'age_head': ('V5850', 427, 428),
        'sex_head': ('V5851', 429, 429),
        'race': ('V6209', 967, 967),
        'education': ('V6194', 950, 950),
        'marital_status': ('V6197', 955, 955),
        'labor_income': ('V6174', 901, 905),
        'wages': ('V5782', 182, 186),
        'hourly_earnings': ('V6178', 920, 924),  # in dollars (99.99 = top-code)
        'annual_hours': ('V5731', 56, 59),
        'occupation': ('V5873_A', 460, 462),
        'industry': ('V5874_A', 465, 467),
        'union': ('V6101', 798, 798),
        'state': ('V5703', 6, 7),  # "CURRENT STATE 1978"
        'smsa': ('V5706', 10, 10),
        'employment_status': ('V5872', 457, 457),
        'same_employer': ('V5940', 578, 578),
        'mo_started_job': ('V5889', 486, 487),
        'govt_employee': ('V5876', 469, 469),
    },
    1979: {
        'interview_number': ('V6302', 2, 5),
        'age_head': ('V6462', 491, 492),
        'sex_head': ('V6463', 493, 493),
        'race': ('V6802', 1038, 1038),
        'education': ('V6787', 1021, 1021),
        'marital_status': ('V6790', 1026, 1026),
        'labor_income': ('V6767', 984, 988),
        'wages': ('V6391', 222, 226),
        'hourly_earnings': ('V6771', 991, 995),  # in dollars (99.99 = top-code)
        'annual_hours': ('V6336', 77, 80),
        'occupation': ('V6497_A', 537, 539),
        'industry': ('V6498_A', 542, 544),
        'union': ('V6707', 887, 887),
        'state': ('V6303', 6, 7),
        'smsa': ('V6306', 10, 10),
        'mo_started_job': ('V6500', 548, 549),
        'govt_employee': ('V6494', 532, 532),
    },
    1980: {
        'interview_number': ('V6902', 2, 5),
        'age_head': ('V7067', 491, 492),
        'sex_head': ('V7068', 493, 493),
        'race': ('V7447', 1102, 1102),
        'education': ('V7433', 1088, 1088),
        'marital_status': ('V7435', 1090, 1090),
        'labor_income': ('V7413', 1039, 1043),
        'wages': ('V6981', 185, 189),
        'hourly_earnings': ('V7417', 1058, 1062),  # in dollars (99.99 = top-code)
        'annual_hours': ('V6934', 65, 68),
        'occupation': ('V7100_A', 535, 537),
        'industry': ('V7101_A', 540, 542),
        'union': ('V7340', 910, 910),
        'state': ('V6903', 6, 7),
        'smsa': ('V6906', 10, 10),
        'mo_started_job': ('V7103', 546, 547),
        'govt_employee': ('V7097', 530, 530),
    },
    1981: {
        'interview_number': ('V7502', 2, 5),
        'age_head': ('V7658', 478, 479),
        'sex_head': ('V7659', 480, 480),
        'race': ('V8099', 1219, 1219),
        'education': ('V8085', 1205, 1205),
        'marital_status': ('V8087', 1207, 1207),
        'labor_income': ('V8066', 1161, 1165),
        'wages': ('V7573', 169, 173),
        'hourly_earnings': ('V8069', 1175, 1179),  # in dollars (99.99 = top-code)
        'annual_hours': ('V7530', 58, 61),
        'occupation': ('V7712', 561, 563),
        'industry': ('V7713', 564, 566),
        'union': ('V7971', 1001, 1001),
        'state': ('V7503', 6, 7),
        'smsa': ('V7506', 10, 10),
        'employment_status': ('V7706', 553, 553),
        'self_employed_check': ('V7707', 554, 554),  # C2 WORK SELF/OTR?
        'same_employer': ('V7779', 689, 689),
        'same_job': ('V7780', 690, 690),
        'mo_started_job': ('V7726', 601, 602),
        'govt_employee': ('V7708', 555, 555),
        'tenure_months_emp': ('V7711', 558, 560),  # C6 # MOS THIS EMP
    },
    1982: {
        'interview_number': ('V8202', 2, 5),
        'age_head': ('V8352', 455, 456),
        'sex_head': ('V8353', 457, 457),
        'race': ('V8723', 1076, 1076),
        'education': ('V8709', 1062, 1062),
        'marital_status': ('V8711', 1064, 1064),
        'labor_income': ('V8690', 1018, 1022),
        'wages': ('V8265', 146, 150),
        'hourly_earnings': ('V8693', 1032, 1036),  # in dollars (99.99 = top-code)
        'annual_hours': ('V8228', 52, 55),
        'occupation': ('V8380', 494, 496),
        'industry': ('V8381', 497, 499),
        'union': ('V8377', 489, 489),
        'state': ('V8203', 6, 7),
        'smsa': ('V8206', 10, 10),
        'employment_status': ('V8374', 486, 486),
        'self_employed_check': ('V8375', 487, 487),
        'same_employer': ('V8444', 615, 615),
        'same_job': ('V8445', 616, 616),
        'govt_employee': ('V8376', 488, 488),
        'tenure_months_emp': ('V8379', 491, 493),  # C6 # MOS THIS EMP
    },
    1983: {
        'interview_number': ('V8802', 2, 5),
        'age_head': ('V8961', 502, 503),
        'sex_head': ('V8962', 504, 504),
        'race': ('V9408', 1259, 1259),
        'education': ('V9395', 1246, 1246),
        'marital_status': ('V9419', 1281, 1281),
        'labor_income': ('V9376', 1200, 1205),
        'wages': ('V8873', 172, 177),
        'hourly_earnings': ('V9379', 1215, 1219),  # in dollars (99.99 = top-code)
        'annual_hours': ('V8830', 62, 65),
        'occupation': ('V9011', 583, 585),
        'industry': ('V9012', 586, 588),
        'union': ('V9008', 578, 578),
        'state': ('V8803', 6, 7),
        'smsa': ('V8806', 10, 10),
        'employment_status': ('V9005', 575, 575),
        'self_employed_check': ('V9006', 576, 576),
        'same_employer': ('V9075', 704, 704),
        'same_job': ('V9076', 705, 705),
        'govt_employee': ('V9007', 577, 577),
        'tenure_months_emp': ('V9010', 580, 582),  # C6 # MOS THIS EMP(HD-E)
    },
}

# Individual file variables for linking
INDIVIDUAL_VARS = {
    'id_68': ('ER30001', 2, 5),      # 1968 interview number (person ID part 1)
    'pn': ('ER30002', 6, 8),          # Person number (person ID part 2)
    'sex': ('ER32000', 2057, 2057),   # Sex (1=Male, 2=Female)
}

# Interview number variables in individual file, by year
IND_INTERVIEW = {
    1968: ('ER30001', 2, 5),
    1969: ('ER30020', 44, 47),
    1970: ('ER30043', 97, 100),
    1971: ('ER30067', 152, 155),
    1972: ('ER30091', 207, 210),
    1973: ('ER30117', 265, 268),
    1974: ('ER30138', 317, 320),
    1975: ('ER30160', 370, 373),
    1976: ('ER30188', 436, 439),
    1977: ('ER30217', 503, 506),
    1978: ('ER30246', 571, 574),
    1979: ('ER30283', 648, 651),
    1980: ('ER30313', 718, 721),
    1981: ('ER30343', 788, 791),
    1982: ('ER30373', 858, 861),
    1983: ('ER30399', 919, 922),
}

# Relationship to head in individual file, by year
IND_RELHEAD = {
    1968: ('ER30003', 9, 9),
    1969: ('ER30022', 50, 50),
    1970: ('ER30045', 103, 103),
    1971: ('ER30069', 158, 158),
    1972: ('ER30093', 213, 213),
    1973: ('ER30119', 271, 271),
    1974: ('ER30140', 323, 323),
    1975: ('ER30162', 376, 376),
    1976: ('ER30190', 442, 442),
    1977: ('ER30219', 509, 509),
    1978: ('ER30248', 577, 577),
    1979: ('ER30285', 654, 654),
    1980: ('ER30315', 724, 724),
    1981: ('ER30345', 794, 794),
    1982: ('ER30375', 864, 864),
    1983: ('ER30401', 925, 926),
}


def read_fixed_width(filepath, var_specs):
    """Read specific variables from a fixed-width ASCII file.

    var_specs: dict of {name: (orig_name, start_col, end_col)}
    where start_col and end_col are 1-based inclusive.
    """
    # Convert to 0-based (start, end_exclusive) for pd.read_fwf
    colspecs = []
    names = []
    for name, (orig, start, end) in var_specs.items():
        colspecs.append((start - 1, end))  # 0-based start, exclusive end
        names.append(name)

    df = pd.read_fwf(filepath, colspecs=colspecs, names=names, header=None)
    return df


def read_individual_file():
    """Read the cross-year individual file and extract key linking variables."""
    filepath = os.path.join(BASE, "ind2023er", "IND2023ER.txt")

    # Build colspecs for all needed variables
    all_specs = {}

    # Person ID components
    all_specs['id_68'] = INDIVIDUAL_VARS['id_68']
    all_specs['pn'] = INDIVIDUAL_VARS['pn']
    all_specs['sex'] = INDIVIDUAL_VARS['sex']

    # Interview numbers for each year
    for year in range(1968, 1984):
        name = f'interview_{year}'
        all_specs[name] = IND_INTERVIEW[year]

    # Relationship to head for each year
    for year in range(1968, 1984):
        name = f'relhead_{year}'
        all_specs[name] = IND_RELHEAD[year]

    print(f"Reading individual file: {filepath}")
    df = read_fixed_width(filepath, all_specs)
    print(f"  Read {len(df)} individuals")

    # Create unique person ID
    df['person_id'] = df['id_68'] * 1000 + df['pn']

    return df


def read_family_file(year):
    """Read a single year's family file."""
    filepath = os.path.join(BASE, f"fam{year}", f"FAM{year}.txt")

    var_specs = FAMILY_VARS[year]

    print(f"Reading family file for {year}: {filepath}")
    df = read_fixed_width(filepath, var_specs)
    print(f"  Read {len(df)} families, columns: {list(df.columns)}")

    # Add year column
    df['year'] = year

    return df


def recode_education(edu_code, year):
    """Convert education bracket codes to approximate years of schooling.

    The PSID education coding differs slightly across years but generally:
    Pre-1975: 0=0-5, 1=6-8, 2=9-11, 3=12, 4=some coll, 5=college deg, 6=adv deg
    1975+:    0=0-5, 1=6-8, 2=9-11, 3=12, 4=12+nonacad, 5=some coll, 6=BA, 7=adv, 9=NA

    We use midpoints to approximate years.
    """
    if pd.isna(edu_code) or edu_code == 9 or edu_code == 99:
        return np.nan

    edu_code = int(edu_code)

    if year < 1975:
        edu_map = {
            0: 3,    # 0-5 grades -> midpoint 3
            1: 7,    # 6-8 grades -> midpoint 7
            2: 10,   # 9-11 grades -> midpoint 10
            3: 12,   # 12 grades
            4: 14,   # some college -> ~14
            5: 16,   # college degree
            6: 17,   # advanced degree
        }
    else:
        edu_map = {
            0: 3,    # 0-5 grades
            1: 7,    # 6-8 grades
            2: 10,   # 9-11 grades
            3: 12,   # 12 grades
            4: 12,   # 12 grades + nonacademic training
            5: 14,   # some college
            6: 16,   # college degree (BA)
            7: 17,   # advanced degree
            8: 17,   # advanced degree (if coded)
        }

    return edu_map.get(edu_code, np.nan)


def determine_self_employed(row, year):
    """Determine if the head is self-employed.

    Returns True if self-employed, False if not, np.nan if unknown.

    All years use the same coding:
      0 = Inapplicable (unemployed, retired, housewife, student)
      1 = Someone else (NOT self-employed)
      2 = Both someone else and self (IS self-employed)
      3 = Self only (IS self-employed)
      9 = NA/DK
    """
    # Check self_employed variable if available
    if 'self_employed' in row.index and not pd.isna(row.get('self_employed')):
        val = row['self_employed']
        if val in (2, 3):
            return True   # Self-employed (both or self only)
        elif val == 1:
            return False  # Someone else (not self-employed)

    # Check self_employed_check (C2 variable in later years) - same coding
    if 'self_employed_check' in row.index and not pd.isna(row.get('self_employed_check')):
        val = row['self_employed_check']
        if val in (2, 3):
            return True   # Self-employed
        elif val == 1:
            return False  # Someone else

    # Check employment status - cannot determine self-employment from this alone
    if 'employment_status' in row.index and not pd.isna(row.get('employment_status')):
        val = row['employment_status']
        # 1=Working now (includes both employed and self-employed)
        # Need separate check for self-employment
        pass

    return np.nan


def determine_govt_employee(row, year):
    """Determine if head works for government."""
    if 'govt_employee' in row.index and not pd.isna(row.get('govt_employee')):
        val = row['govt_employee']
        # Typically: 1=Federal, 2=State, 3=Local, 4=Other (not govt), 5=No (not govt)
        # Or: 1=Yes (govt), 5=No
        if val in [1, 2, 3]:
            return True
        elif val in [4, 5]:
            return False
    return np.nan


def is_agriculture(occ_code, ind_code):
    """Check if the occupation/industry is agriculture.

    Agriculture industry codes (Census 3-digit):
      017-029 = Agriculture, forestry, fishing
    Agriculture occupation codes:
      100-199 = Farmers, farm managers, farm laborers
    """
    is_ag = False

    if not pd.isna(ind_code):
        ind = int(ind_code)
        if 17 <= ind <= 29:
            is_ag = True

    if not pd.isna(occ_code):
        occ = int(occ_code)
        # Farm workers in Census occupation codes
        if 100 <= occ <= 199:
            is_ag = True
        # Also check for 3-digit codes used differently in some years
        if 600 <= occ <= 699:  # Farm occupations in 1970 Census coding
            is_ag = True

    return is_ag


def build_panel():
    """Build the person-year panel dataset."""

    # Step 1: Read individual file
    print("="*60)
    print("STEP 1: Reading individual file")
    print("="*60)
    ind = read_individual_file()

    # Step 2: Read all family files
    print("\n" + "="*60)
    print("STEP 2: Reading family files")
    print("="*60)

    family_dfs = {}
    for year in range(1968, 1984):
        family_dfs[year] = read_family_file(year)

    # Step 3: Create person-year panel by merging individual and family data
    print("\n" + "="*60)
    print("STEP 3: Creating person-year panel")
    print("="*60)

    all_person_years = []

    for year in range(1968, 1984):
        print(f"\nProcessing year {year}...")

        fam = family_dfs[year].copy()

        # Get individuals who are heads in this year
        # From individual file, get interview number and relationship to head
        interview_col = f'interview_{year}'
        relhead_col = f'relhead_{year}'

        # Filter individual file to those present in this year (interview > 0)
        ind_year = ind[ind[interview_col] > 0][['person_id', 'id_68', 'pn', 'sex',
                                                  interview_col, relhead_col]].copy()
        ind_year.rename(columns={interview_col: 'fam_interview',
                                  relhead_col: 'relhead'}, inplace=True)

        # Keep only heads (relationship = 1 for pre-1983, = 10 for 1983+)
        if year <= 1982:
            heads = ind_year[ind_year['relhead'] == 1].copy()
        else:
            heads = ind_year[ind_year['relhead'].isin([1, 10])].copy()

        print(f"  Heads in individual file for {year}: {len(heads)}")

        # Merge heads with family data on interview number
        merged = heads.merge(fam, left_on='fam_interview', right_on='interview_number',
                            how='inner')

        print(f"  Merged records: {len(merged)}")

        # Standardize columns
        result = pd.DataFrame()
        result['person_id'] = merged['person_id']
        result['year'] = year
        result['fam_interview'] = merged['fam_interview']
        result['age'] = pd.to_numeric(merged.get('age_head'), errors='coerce')
        result['sex_head'] = pd.to_numeric(merged.get('sex_head'), errors='coerce')
        result['sex_ind'] = pd.to_numeric(merged.get('sex'), errors='coerce')
        result['race'] = pd.to_numeric(merged.get('race'), errors='coerce')
        result['education_code'] = pd.to_numeric(merged.get('education'), errors='coerce')
        result['marital_status'] = pd.to_numeric(merged.get('marital_status'), errors='coerce')
        result['labor_income'] = pd.to_numeric(merged.get('labor_income'), errors='coerce')

        # Wages
        if 'wages' in merged.columns:
            result['wages'] = pd.to_numeric(merged['wages'], errors='coerce')
        else:
            result['wages'] = np.nan

        # Hourly earnings (in cents for most years)
        if 'hourly_earnings' in merged.columns:
            result['hourly_earnings_raw'] = pd.to_numeric(merged['hourly_earnings'], errors='coerce')
        else:
            result['hourly_earnings_raw'] = np.nan

        result['annual_hours'] = pd.to_numeric(merged.get('annual_hours'), errors='coerce')
        result['occupation'] = pd.to_numeric(merged.get('occupation'), errors='coerce')
        result['industry'] = pd.to_numeric(merged.get('industry'), errors='coerce')
        result['state'] = pd.to_numeric(merged.get('state'), errors='coerce')
        result['region'] = pd.to_numeric(merged.get('region'), errors='coerce')
        result['smsa'] = pd.to_numeric(merged.get('smsa'), errors='coerce')

        # Self-employed
        if 'self_employed' in merged.columns:
            result['self_employed_raw'] = pd.to_numeric(merged['self_employed'], errors='coerce')
        else:
            result['self_employed_raw'] = np.nan

        if 'self_employed_check' in merged.columns:
            result['self_employed_check'] = pd.to_numeric(merged['self_employed_check'], errors='coerce')
        else:
            result['self_employed_check'] = np.nan

        # Employment status
        if 'employment_status' in merged.columns:
            result['employment_status'] = pd.to_numeric(merged['employment_status'], errors='coerce')
        else:
            result['employment_status'] = np.nan

        # Union
        if 'union' in merged.columns:
            result['union_raw'] = pd.to_numeric(merged['union'], errors='coerce')
        else:
            result['union_raw'] = np.nan

        # Disability
        if 'disability' in merged.columns:
            result['disability'] = pd.to_numeric(merged['disability'], errors='coerce')
        else:
            result['disability'] = np.nan

        # Government employee
        if 'govt_employee' in merged.columns:
            result['govt_employee_raw'] = pd.to_numeric(merged['govt_employee'], errors='coerce')
        else:
            result['govt_employee_raw'] = np.nan

        # Tenure / same employer
        if 'same_employer' in merged.columns:
            result['same_employer'] = pd.to_numeric(merged['same_employer'], errors='coerce')
        else:
            result['same_employer'] = np.nan

        if 'same_job' in merged.columns:
            result['same_job'] = pd.to_numeric(merged['same_job'], errors='coerce')
        else:
            result['same_job'] = np.nan

        if 'mo_started_job' in merged.columns:
            result['mo_started_job'] = pd.to_numeric(merged['mo_started_job'], errors='coerce')
        else:
            result['mo_started_job'] = np.nan

        if 'tenure_months' in merged.columns:
            result['tenure_months'] = pd.to_numeric(merged['tenure_months'], errors='coerce')
        elif 'tenure_months_emp' in merged.columns:
            result['tenure_months'] = pd.to_numeric(merged['tenure_months_emp'], errors='coerce')
        else:
            result['tenure_months'] = np.nan

        all_person_years.append(result)

    # Combine all years
    panel = pd.concat(all_person_years, ignore_index=True)
    print(f"\nCombined panel: {len(panel)} person-years")

    # Step 4: Construct derived variables
    print("\n" + "="*60)
    print("STEP 4: Constructing derived variables")
    print("="*60)

    # Convert education codes to years
    panel['education_years'] = panel.apply(
        lambda row: recode_education(row['education_code'], row['year']), axis=1)

    # Experience = Age - Education - 6
    panel['experience'] = panel['age'] - panel['education_years'] - 6

    # Self-employed indicator
    # All years use the same coding for self_employed / self_employed_check:
    #   0 = Inapplicable (unemployed, retired, housewife, student)
    #   1 = Someone else (NOT self-employed)
    #   2 = Both someone else and self (IS self-employed)
    #   3 = Self only (IS self-employed)
    #   9 = NA/DK
    # self_employed: 1968 V198, 1969 V641, 1970 V1280, 1971 V1986, 1972 V2584, 1974 V3532
    # self_employed_check: 1981 V7707, 1982 V8375, 1983 V9006
    panel['is_self_employed'] = False
    mask_se = panel['self_employed_raw'].isin([2, 3])
    panel.loc[mask_se, 'is_self_employed'] = True
    mask_se_check = panel['self_employed_check'].isin([2, 3])
    panel.loc[mask_se_check, 'is_self_employed'] = True

    # For years without explicit self-employment variable, use employment status
    # Employment status: 1=Working, 2=Temp layoff, 3=Unemployed, etc.
    # Self-employment can only be confirmed where a separate variable exists

    # Government employee indicator
    panel['is_govt_employee'] = False
    mask_govt = panel['govt_employee_raw'].isin([1, 2, 3])
    panel.loc[mask_govt, 'is_govt_employee'] = True

    # Agriculture indicator
    panel['is_agriculture'] = panel.apply(
        lambda row: is_agriculture(row['occupation'], row['industry']), axis=1)

    # White indicator (race=1)
    panel['is_white'] = (panel['race'] == 1)

    # Male indicator
    # Use sex from individual file if available, otherwise from family head variable
    panel['is_male'] = False
    # From individual file: 1=Male
    mask_male_ind = (panel['sex_ind'] == 1)
    panel.loc[mask_male_ind, 'is_male'] = True
    # Fallback to family head sex: 1=Male
    mask_male_head = (panel['sex_ind'].isna()) & (panel['sex_head'] == 1)
    panel.loc[mask_male_head, 'is_male'] = True

    # Hourly earnings computation
    # PSID hourly earnings are in DOLLARS (not cents)
    # Values of 99.99 are top-codes (treat as missing)
    # For years where hourly_earnings_raw is available, use it
    # Otherwise compute from wages/hours
    panel['hourly_earnings_dollars'] = np.nan

    # Handle top-coded values (99.99)
    panel.loc[panel['hourly_earnings_raw'] >= 99.98, 'hourly_earnings_raw'] = np.nan

    # Use the PSID-provided hourly earnings (already in dollars)
    mask_has_hrly = panel['hourly_earnings_raw'].notna() & (panel['hourly_earnings_raw'] > 0)
    panel.loc[mask_has_hrly, 'hourly_earnings_dollars'] = panel.loc[mask_has_hrly, 'hourly_earnings_raw']

    # For observations without PSID hourly earnings, compute from labor income / hours
    mask_needs_compute = panel['hourly_earnings_dollars'].isna() & (panel['annual_hours'] > 0)

    # Prefer wages over labor_income for computing hourly rate
    mask_has_wages = mask_needs_compute & panel['wages'].notna() & (panel['wages'] > 0)
    panel.loc[mask_has_wages, 'hourly_earnings_dollars'] = (
        panel.loc[mask_has_wages, 'wages'] / panel.loc[mask_has_wages, 'annual_hours'])

    mask_has_labor = mask_needs_compute & ~mask_has_wages & panel['labor_income'].notna() & (panel['labor_income'] > 0)
    panel.loc[mask_has_labor, 'hourly_earnings_dollars'] = (
        panel.loc[mask_has_labor, 'labor_income'] / panel.loc[mask_has_labor, 'annual_hours'])

    # Deflate hourly earnings by CPS wage index
    # The index values are for the year the income was earned (which is year-1 for most PSID files)
    # PSID year Y contains income data for year Y-1
    # For 1968: income is from 1967 (no index) -- but Topel uses 1968 interview year data
    # Note: Topel uses INTERVIEW year as the observation year. The income data in year Y's
    # interview refers to the prior year (Y-1). But Topel's wage index is indexed to match.
    # We deflate using the INTERVIEW YEAR index.
    panel['cps_wage_index'] = panel['year'].map(CPS_WAGE_INDEX)
    panel['log_hourly_real'] = np.nan
    mask_pos = panel['hourly_earnings_dollars'].notna() & (panel['hourly_earnings_dollars'] > 0)
    panel.loc[mask_pos, 'log_hourly_real'] = (
        np.log(panel.loc[mask_pos, 'hourly_earnings_dollars']) -
        np.log(panel.loc[mask_pos, 'cps_wage_index']))

    # Step 5: Construct tenure from "same employer" variable
    print("\n" + "="*60)
    print("STEP 5: Constructing tenure")
    print("="*60)

    # Sort by person and year
    panel.sort_values(['person_id', 'year'], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # Initialize tenure
    panel['tenure'] = np.nan

    # For each person, track tenure across years
    # same_employer: 1=Yes (same employer as last year), 5=No, 0=NA/inap
    # When same_employer is missing, we can use mo_started_job to infer

    persons = panel.groupby('person_id')

    tenure_values = np.full(len(panel), np.nan)

    for pid, group in persons:
        idx = group.index.tolist()
        years = group['year'].values

        # First observation for this person: tenure = 0 (new entrant)
        current_tenure = 0
        tenure_values[idx[0]] = current_tenure

        for i in range(1, len(idx)):
            prev_year = years[i-1]
            curr_year = years[i]
            row = panel.loc[idx[i]]
            prev_row = panel.loc[idx[i-1]]

            # Check if consecutive years
            if curr_year != prev_year + 1:
                # Gap in panel - reset tenure
                current_tenure = 0
                tenure_values[idx[i]] = current_tenure
                continue

            # Check "same employer" variable
            same_emp = row.get('same_employer')
            if not pd.isna(same_emp) and same_emp != 0:
                if same_emp == 1:
                    # Same employer - increment tenure
                    current_tenure += 1
                elif same_emp == 5:
                    # Different employer - reset tenure
                    current_tenure = 0
                tenure_values[idx[i]] = current_tenure
                continue

            # Check "same job" variable
            same_j = row.get('same_job')
            if not pd.isna(same_j) and same_j != 0:
                if same_j == 1:
                    current_tenure += 1
                elif same_j == 5:
                    current_tenure = 0
                tenure_values[idx[i]] = current_tenure
                continue

            # Check month started job - if it changed substantially from prior year,
            # that might indicate a new job, but this is unreliable
            # For years without same_employer variable, use the "mo_started_job"
            # If the month started job in the current year is recent (within last year),
            # it indicates a new job
            mo_start = row.get('mo_started_job')
            if not pd.isna(mo_start) and mo_start > 0:
                # If they started a new job recently (in the current or prior year),
                # reset tenure. Otherwise, they're with the same employer.
                # However, mo_started_job reports when current employment spell began
                # If mo_started_job is different from previous year's value, new job
                prev_mo = prev_row.get('mo_started_job')
                if not pd.isna(prev_mo) and prev_mo > 0:
                    if mo_start != prev_mo:
                        # Different start month -> probably new job
                        current_tenure = 0
                    else:
                        current_tenure += 1
                    tenure_values[idx[i]] = current_tenure
                    continue

            # Default: assume same employer (increment tenure)
            # This is a simplification; Topel uses occupation/industry matching
            # and other indicators when same_employer is unavailable
            current_tenure += 1
            tenure_values[idx[i]] = current_tenure

    panel['tenure'] = tenure_values

    print(f"Tenure distribution:")
    print(panel['tenure'].describe())

    # Step 6: Apply sample restrictions
    print("\n" + "="*60)
    print("STEP 6: Applying sample restrictions")
    print("="*60)

    n_start = len(panel)
    print(f"Starting observations: {n_start}")

    # Restriction 1: White males only
    panel_r = panel[panel['is_white'] & panel['is_male']].copy()
    print(f"After white males only: {len(panel_r)} (dropped {n_start - len(panel_r)})")

    # Restriction 2: Ages 18-60
    n_before = len(panel_r)
    panel_r = panel_r[(panel_r['age'] >= 18) & (panel_r['age'] <= 60)]
    print(f"After age 18-60: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Restriction 3: Not self-employed
    n_before = len(panel_r)
    panel_r = panel_r[~panel_r['is_self_employed']]
    print(f"After excluding self-employed: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Restriction 4: Not in agriculture
    n_before = len(panel_r)
    panel_r = panel_r[~panel_r['is_agriculture']]
    print(f"After excluding agriculture: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Restriction 5: Not government employees (where variable available)
    n_before = len(panel_r)
    panel_r = panel_r[~panel_r['is_govt_employee']]
    print(f"After excluding govt employees: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Restriction 6: SRC sample only (exclude SEO poverty oversample)
    # In 1968, SRC families have interview numbers 1-2930 (approximately)
    # SEO families have numbers 5001-6872
    # For later years, the "split sample" variable identifies SRC vs SEO
    # But since we're tracking the same people, the 1968 interview number
    # determines SRC/SEO membership. SRC sample = id_68 < 3000
    # Actually, the standard PSID documentation says:
    #   SRC: 1968 IDs 1-2930
    #   SEO (Census): 1968 IDs 5001-6872
    n_before = len(panel_r)
    # Get the 1968 family ID from person_id
    panel_r['id_68_from_pid'] = panel_r['person_id'] // 1000
    panel_r = panel_r[panel_r['id_68_from_pid'] <= 2930]
    print(f"After SRC sample only (id_68 <= 2930): {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Restriction 7: Positive earnings
    n_before = len(panel_r)
    panel_r = panel_r[panel_r['hourly_earnings_dollars'].notna() & (panel_r['hourly_earnings_dollars'] > 0)]
    print(f"After positive hourly earnings: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Restriction 8: Positive annual hours
    n_before = len(panel_r)
    panel_r = panel_r[panel_r['annual_hours'].notna() & (panel_r['annual_hours'] > 0)]
    print(f"After positive annual hours: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Additional: require valid age and education
    n_before = len(panel_r)
    panel_r = panel_r[panel_r['age'].notna() & panel_r['education_years'].notna()]
    print(f"After valid age & education: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Additional: require positive experience
    n_before = len(panel_r)
    panel_r = panel_r[panel_r['experience'] > 0]
    print(f"After positive experience: {len(panel_r)} (dropped {n_before - len(panel_r)})")

    # Trim extreme hourly wages (likely coding errors)
    # Topel doesn't explicitly state trim bounds but researchers typically use $1-$100/hr
    n_before = len(panel_r)
    panel_r = panel_r[(panel_r['hourly_earnings_dollars'] >= 1.0) &
                       (panel_r['hourly_earnings_dollars'] <= 200.0)]
    print(f"After trimming extreme wages ($1-$200): {len(panel_r)} (dropped {n_before - len(panel_r)})")

    print(f"\nFinal panel size: {len(panel_r)} person-years")
    print(f"Unique persons: {panel_r['person_id'].nunique()}")
    print(f"Years covered: {sorted(panel_r['year'].unique())}")

    # Step 7: Reconstruct tenure for restricted sample
    print("\n" + "="*60)
    print("STEP 7: Reconstructing tenure for restricted sample")
    print("="*60)

    # Re-sort
    panel_r.sort_values(['person_id', 'year'], inplace=True)
    panel_r.reset_index(drop=True, inplace=True)

    # Recalculate tenure within the restricted sample
    # Since some observations may have been dropped, tenure needs rechecking
    new_tenure = np.full(len(panel_r), np.nan)

    for pid, group in panel_r.groupby('person_id'):
        idx = group.index.tolist()
        years = group['year'].values

        # Use the existing tenure logic but verify continuity
        prev_tenure = 0
        new_tenure[idx[0]] = panel_r.loc[idx[0], 'tenure']
        if pd.isna(new_tenure[idx[0]]):
            new_tenure[idx[0]] = 0

        for i in range(1, len(idx)):
            if years[i] == years[i-1] + 1:
                # Consecutive year - use existing tenure if available
                if not pd.isna(panel_r.loc[idx[i], 'tenure']):
                    new_tenure[idx[i]] = panel_r.loc[idx[i], 'tenure']
                else:
                    new_tenure[idx[i]] = new_tenure[idx[i-1]] + 1
            else:
                # Gap - reset
                new_tenure[idx[i]] = 0

    panel_r['tenure'] = new_tenure

    print(f"Tenure distribution (restricted sample):")
    print(panel_r['tenure'].describe())

    # Step 8: Select and save output columns
    print("\n" + "="*60)
    print("STEP 8: Saving output")
    print("="*60)

    output_cols = [
        'person_id', 'year', 'fam_interview',
        'age', 'sex_head', 'race',
        'education_code', 'education_years',
        'marital_status',
        'labor_income', 'wages',
        'hourly_earnings_dollars', 'log_hourly_real',
        'annual_hours',
        'experience', 'tenure',
        'occupation', 'industry',
        'state', 'region', 'smsa',
        'employment_status',
        'is_self_employed', 'is_govt_employee', 'is_agriculture',
        'union_raw',
        'same_employer', 'same_job', 'mo_started_job',
        'cps_wage_index',
    ]

    # Only include columns that exist
    output_cols = [c for c in output_cols if c in panel_r.columns]

    output_path = os.path.join(OUTDIR, "psid_panel.csv")
    panel_r[output_cols].to_csv(output_path, index=False)
    print(f"Saved panel to: {output_path}")
    print(f"Final dimensions: {panel_r[output_cols].shape}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nObservations by year:")
    print(panel_r.groupby('year').size().to_string())

    print(f"\nKey variable means:")
    for col in ['age', 'education_years', 'experience', 'tenure',
                 'hourly_earnings_dollars', 'log_hourly_real', 'annual_hours']:
        if col in panel_r.columns:
            print(f"  {col}: mean={panel_r[col].mean():.3f}, "
                  f"median={panel_r[col].median():.3f}, "
                  f"std={panel_r[col].std():.3f}")

    print(f"\nTenure distribution:")
    print(panel_r['tenure'].value_counts().sort_index().head(20).to_string())

    return panel_r


if __name__ == '__main__':
    panel = build_panel()
