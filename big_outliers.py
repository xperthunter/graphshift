#!/usr/bin/python3

import sys
import csv
import numpy as np
import pandas as pd

import datalib as dl

df = pd.read_json(sys.argv[1],compression='xz').reset_index()

atoms = ['N', 'H', 'CA', 'HA', 'C']
"""
for a in atoms:
	
	stats = dl.stats_report(dl.clean_atom_set(df,a, 1.0), a)
	print(f"atom: {a} max: {stats['max']:.4f} min: {stats['min']:.4f}")
	
	max = stats['max']
	min = stats['min']
	
	mx = False
	mi = False
	for id, shifts, bool in zip(df['name'], df[a], df[a].isna()):
		if bool: continue
		
		if max in shifts:
			print(f'\tMAX atom: {a} bmrb_id: {id} val: {max}')
			mx = True
		
		if min in shifts:
			print(f'\tMIN atom: {a} bmrb_id: {id} val: {min}')
			mi = True
		
		if mx and mi:
			break
"""

new_frame = dl.strong_filter(df, ['HA'], 3.0)
print(new_frame.shape)
			 