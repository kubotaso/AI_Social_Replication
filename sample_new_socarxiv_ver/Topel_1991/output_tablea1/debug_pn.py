#!/usr/bin/env python3
"""Check how person number filter affects sample size."""
import pandas as pd

df = pd.read_csv('data/psid_panel.csv')
df['pn'] = df['person_id'] % 1000

print(f"All: {len(df)} obs, {df['person_id'].nunique()} persons")

# Filter: pn < 170 (original household members, not splitoffs)
# In PSID, persons 001-169 are original sample members,
# 170+ are born-in or moved-in members
df_orig = df[df['pn'] < 170]
print(f"pn < 170: {len(df_orig)} obs, {df_orig['person_id'].nunique()} persons")

df_head = df[df['pn'] == 1]
print(f"pn == 1 (heads only): {len(df_head)} obs, {df_head['person_id'].nunique()} persons")

# Heads or wives (pn <= 2)
df_hw = df[df['pn'] <= 2]
print(f"pn <= 2: {len(df_hw)} obs, {df_hw['person_id'].nunique()} persons")

# Heads or wives or first child (pn <= 3)
df_hwc = df[df['pn'] <= 10]
print(f"pn <= 10: {len(df_hwc)} obs, {df_hwc['person_id'].nunique()} persons")

# Check who the non-head persons are
non_heads = df[df['pn'] > 1]['pn'].value_counts().sort_index()
print(f"\nPersons by pn (non-head):")
print(non_heads)

# What about filtereing to original SRC families + original members
# SRC families: id_68 <= 2930
# Original members: pn < 170
df_src_orig = df[(df['person_id'] // 1000 <= 2930) & (df['pn'] < 170)]
print(f"\nSRC + pn < 170: {len(df_src_orig)} obs, {df_src_orig['person_id'].nunique()} persons")
