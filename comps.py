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

#keep = sequence_filter.identity_filter(sys.argv[2], 50)
#df = dfin[dfin['name'].isin(keep)]
#print(f'After filtering: {df.shape}')

#heldout_df = dfin[~dfin['name'].isin(keep)]
idx = dfin['index'].to_list()
print(f'indexes: {len(idx)}')
idxs = [(i, j) for i, j in itertools.product(idx, idx) 
		if i < j]
print(f'comparisons: {len(idxs)}')

def identical_data(x):
	x1, x2 = x
	r1 = dfin.loc[x1]
	r2 = dfin.loc[x2]
	
	matches = []
	for col, b1, b2 in zip(r1.keys(), r1.isnull().values, r2.isnull().values):
		if col == 'index' or col == 'name' or col == 'seq': continue
		if b1 or b2: continue
		#if len(r1[col]) != len(r2[col]): continue
		diff = False
		for s1, s2 in zip(r1[col], r2[col]):
			if s1 is None and s2 is None: continue
			if s1 != s2:
				diff = True
				break
		
		if ~diff:
			matches.append((x1, x2, col))
	
	if len(matches) > 0: return matches
	else: return matches

with Pool(processes=12) as pool:
	res = pool.map(identical_data, idxs)

print(len(res))
print(res[:20])

kill = dict()
for l in res:
	if len(l) > 0:
		for tup in l:
			i, j, atm = tup
			if j not in kill: kill[j] = dict()
			kill[j][atm] = True

with open('skips', 'w') as fo:
	json.dump(kill,fo,indent=2)
















