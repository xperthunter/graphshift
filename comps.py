#!/usr/bin/python3

import itertools
import json
from multiprocessing import Pool
from multiprocessing import sharedctypes
from multiprocessing import Manager
import numpy as np
import pandas as pd
import sys

import sequence_filter

dfin = pd.read_json(sys.argv[1],compression='xz').reset_index()
print(f'Input shape: {dfin.shape}')

keep = sequence_filter.identity_filter(sys.argv[2], 50)
df = dfin[dfin['name'].isin(keep)]
print(f'After filtering: {df.shape}')

heldout_df = dfin[~dfin['name'].isin(keep)]
idx = heldout_df['index'].to_list()
idx = idx[:1000]
print(len(idx))
#print(heldout_df.loc[idx[1]]['seq'])
#print(sys.exit())
idxs = [(i, j) for i, j in itertools.product(idx, idx) 
		if i < j]
print(len(idxs))

def identical_data(x):
	x1, x2 = x
	r1 = heldout_df.loc[x1]
	r2 = heldout_df.loc[x2]
	matches = []
	for c1, c2 in zip(r1.keys(), r2.keys()):
		if c1 == 'index' or c1 == 'name' or c1 == 'seq': continue
		if c2 == 'index' or c2 == 'name' or c2 == 'seq': continue
		if type(r1[c1]) != list or type(r2[c2]) != list: continue
		if len(r1[c2]) != len(r2[c2]): continue
		
		diff = False
		for s1, s2 in zip(r1[c1], r2[c2]):
			if s1 != s2:
				diff = True
				break
				
		if diff is False:
			matches.append((x1, x2, c1, c2))
	
	if len(matches) > 0: return matches

with Pool(processes=6) as pool:
	res = pool.map(identical_data, idxs)

print(len(res))
print(res[:20])

"""
for p in idxs:
	r, c = p
	print(r, c)
"""