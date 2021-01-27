#!/usr/bin/python3

import argparse
import json
from math import sqrt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
import sys

import sequence_filter

parser = argparse.ArgumentParser(description=''.join(('Construct base models',
	' for chemical shift prediction. Base models are shift averages constructed',
	' from element, atom-type, and atom-type&residue splits of chemical shift',
	' data.')))
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--blast','-b', required=False, type=str,
	metavar='<str>', help='blast report for sequence filtering')
parser.add_argument('--identity','-i', required=False, type=int,
	metavar='<int>', help='maximum percent identity in dataset')

"""
which entries are identical
but i am getting at the atm level
"""

def identical_data(df1, df2):
	mask = dict()
	counter = 0
	for i, r1 in df1.iloc[0:].iterrows():
		#print(i)
		counter += 1
		if counter % 100 == 0: print(i, counter)
		for j, r2 in df2.iloc[counter:].iterrows():
			#if j <= i: continue
			#print(i, j)
			for c1, c2 in zip(r1.keys(), r2.keys()):
				if c1 == 'index' or c1 == 'name' or c1 == 'seq': continue
				if c2 == 'index' or c2 == 'name' or c2 == 'seq': continue
				if type(r1[c1]) != list or type(r2[c2]) != list: continue
				if len(r1[c2]) != len(r2[c2]): continue
				
				diff = False
				for s1, s2 in zip(r1[c1], r2[c2]):
					if s1 != s2: diff = True
				
				if diff is False:
					if j not in mask: mask[j] = dict()
					mask[j][c2] = True
	return mask				 

def performance(shifts):
	rmse = 0
	np.random.shuffle(shifts)
	kf = KFold(n_splits=10)
	kf.get_n_splits(shifts)
	
	for tr, te in kf.split(shifts):
		train = shifts[tr]
		test  = shifts[te]
		
		mu = np.mean(train)
		r = 0
		for t in test:
			r += (t - mu)**2
		
		r /= len(test)
		r = sqrt(r)
		rmse += r
	rmse /= 10
	return rmse

arg = parser.parse_args()

dfin = pd.read_json(arg.data,compression='xz').sample(frac=1).reset_index()
print(f'Input shape: {dfin.shape}')

if arg.blast:
	assert(type(arg.identity) == int)
	assert(arg.identity > 0)
	assert(arg.identity < 101)
	
	keep = sequence_filter.identity_filter(arg.blast, arg.identity)
	df = dfin[dfin['name'].isin(keep)]
	print(f'After filtering: {df.shape}')
	heldout_df = dfin[~dfin['name'].isin(keep)]
	
	self_heldout = identical_data(heldout_df, heldout_df)
	print(json.dumps(self_heldout,indent=2))
	"""
	held_elms = {'N':[], 'H':[], 'C':[]}
	held_set = {'N':[], 'H':[], 'C':[]}
	for indx, row in heldout_df.iloc[0:].iterrows():
		
		for col in row.keys():
			if col == 'index' or col == 'name' or col == 'seq': continue
			
			el = col[0]
			if type[row[col]] != list: continue
			held_elms[el].extend(row[col])
			for i, aa in enumerate(row[seq]):
				if row[col][i] == None: continue
				held_set[el].append((col, aa, row[col][i]))

	# in held set, find rows that are exactly the same




# make element model
#print('\n== ELEMENT MODEL ==')

elements  = {'N':[], 'H':[], 'C':[]}
shift_set = {'N':[], 'H':[], 'C':[]}
for indx, row in df.iloc[0:].iterrows():
	seq = row['seq']
	for col in row.keys():
		if col == 'index' or col == 'name' or col == 'seq': continue
		
		el = col[0]
		if type(row[col]) != list: continue
		elements[el].extend(row[col])
		for i, aa in enumerate(seq):
			if row[col][i] == None: continue
			shift_set[el].append((col, aa, row[col][i]))

print(f'{"model":<7} {"N rmse":>7} {"H rmse":>7} {"C rmse":>7} {"sum":>7}',end='')
print(f' | held_out')

print(f'{"element":<7}', end=' ')
sum = 0
for el in elements.keys():
	shifts = np.array(elements[el])
	shifts = shifts[np.where(shifts != None)]
	rmse = performance(shifts)
	sum += rmse
	print(f'{rmse:{" "}>7.4f}', end=' ')
	
	held_rmse = 0
	tot = np.mean(shifts)
	for t in held_elms[el]:
		held_rmse += (t - tot)**2
	held_rmse /= len(held_elms[el])
	held_rmse = sqrt(held_rmse)
	
	print 
	
	
print(f'{sum:{" "}>7.4f}')

print(f'{"atom":<7}', end=' ')
sum = 0
for el in elements.keys():
	shifts = shift_set[el]
	random.shuffle(shifts)
	
	kf = KFold(n_splits=10)
	kf.get_n_splits(shifts)
	val_rmse = 0
	
	train = []
	test  = []
	
	for tr, te in kf.split(shifts):
		train = [shifts[i] for i in tr]
		test  = [shifts[i] for i in te]
		model = dict()
		
		for s in train:
			if s[0] not in model: model[s[0]] = list()
			model[s[0]].append(s[2])
		
		for k in model.keys():
			cs = np.array(model[k])
			model[k] = np.mean(cs)
		
		#print(json.dumps(model,indent=2))
		
		rmse = 0
		for ts in test:
			rmse += (ts[2] - model[ts[0]])**2
		
		rmse /= len(test)
		rmse = sqrt(rmse)
		#print(rmse)
		val_rmse += rmse
	val_rmse /= 10
	sum += val_rmse
	print(f'{val_rmse:{" "}>7.4f}', end=' ')
print(f'{sum:{" "}>7.4f}')

print(f'{"atm-res":<7}', end=' ')
sum = 0
for el in elements.keys():
	shifts = shift_set[el]
	random.shuffle(shifts)
	
	kf = KFold(n_splits=10)
	kf.get_n_splits(shifts)
	val_rmse = 0
	
	train = []
	test  = []
	
	for tr, te in kf.split(shifts):
		train = [shifts[i] for i in tr]
		test  = [shifts[i] for i in te]
		model = dict()
		
		for s in train:
			if s[1] not in model: model[s[1]] = dict()
			if s[0] not in model[s[1]]: model[s[1]][s[0]] = list()
			model[s[1]][s[0]].append(s[2])
		
		for k1 in model.keys():
			for k2 in model[k1].keys():
				cs = np.array(model[k1][k2])
				model[k1][k2] = np.mean(cs)
		
		#print(json.dumps(model,indent=2))
		
		rmse = 0
		for ts in test:
			rmse += (ts[2] - model[ts[1]][ts[0]])**2
		
		rmse /= len(test)
		rmse = sqrt(rmse)
		#print(rmse)
		val_rmse += rmse
	val_rmse /= 10
	sum += val_rmse
	print(f'{val_rmse:{" "}>7.4f}', end=' ')
print(f'{sum:{" "}>7.4f}')
"""
	
	
		
		
				
		
		
		







