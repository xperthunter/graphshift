#!/usr/bin/python3

import argparse
import itertools
import json
from math import sqrt
from multiprocessing import Pool
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
parser.add_argument('--skips', '-k', required=False, type=str,
	metavar='<str>', help='kill list')

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

def make_set(df, kill):
	elements   = {'N':[], 'H':[], 'C':[]}
	shift_set  = {'N':[], 'H':[], 'C':[]}
	for indx, row in df.iloc[0:].iterrows():
		
		for col in row.keys():
			if col == 'index' or col =='name' or col == 'seq': continue
			if type(row[col]) != list: continue
			if kill:
				if indx in kill:
					if col in kill[indx]: continue
			
			el = col[0]
			elements[el].extend(row[col])
			seq = row['seq']
			for i, aa in enumerate(seq):
				if row[col][i] == None: continue
				if i-1 > 0 and i+1 < len(seq):
					set = {'elm':el, 'atm':col, 'res':aa,
						   'resl':seq[i-1], 'resr':seq[i+1],
						   'cs':row[col][i]}
				else:
					set = {'elm':el, 'atm':col, 'res':aa,
						   'resl':None, 'resr':None,
						   'cs':row[col][i]}
				shift_set[el].append(set)
	
	return elements, shift_set

arg = parser.parse_args()

dfin = pd.read_json(arg.data,compression='xz').reset_index()
print(f'Input shape: {dfin.shape}')

kill = {}
if arg.skips:
	with open(arg.skips) as fp:
		kill = json.load(fp)

if arg.blast:
	assert(type(arg.identity) == int)
	assert(arg.identity > 0)
	assert(arg.identity < 101)
	
	keep = sequence_filter.identity_filter(arg.blast, arg.identity)
	df = dfin[dfin['name'].isin(keep)]
	print(f'After filtering: {df.shape}')
	heldout_df = dfin[~dfin['name'].isin(keep)]
	
	held_elms, held_set = make_set(heldout_df, kill)

elms, sets = make_set(df, kill)

print(f'{"model":<7} {"Nrmse":>7} {"Nout":>7} {"Hrmse":>7} {"Hout":>7}',end='')
print(f' {"Crmse":>7} {"Cout":>7} {"Sumrmse":>7} {"Sumout":>7}')

print(f'{"element":<7}', end=' ')
sumin  = 0
sumout = 0
for el in elms.keys():
	shifts = np.array(elms[el])
	shifts = shifts[np.where(shifts != None)]
	rmse = performance(shifts)
	sumin += rmse
	print(f'{rmse:{" "}>7.4f}', end=' ')
	
	held_rmse = 0
	tot = np.mean(shifts)
	hs = np.array(held_elms[el])
	hs = hs[np.where(hs != None)]
	for t in hs: 
		held_rmse += (t - tot)**2
	held_rmse /= hs.shape[0]
	held_rmse = sqrt(held_rmse)
	print(f'{held_rmse:{" "}>7.4f}', end=' ')
	sumout += held_rmse

print(f'{sumin:{" "}>7.4f} {sumout:{" "}>7.4f}')

print(f'{"atom":<7}', end=' ')
sumin  = 0
sumout = 0
for el in elms.keys():
	shifts = sets[el]
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
			if s['atm'] not in model: model[s['atm']] = list()
			model[s['atm']].append(s['cs'])
		
		for k in model.keys():
			cs = np.array(model[k])
			model[k] = np.mean(cs)
		
		rmse = 0
		for ts in test:
			rmse += (ts['cs'] - model[ts['atm']])**2
		
		rmse /= len(test)
		rmse = sqrt(rmse)
		#print(rmse)
		val_rmse += rmse
	val_rmse /= 10
	sumin += val_rmse
	print(f'{val_rmse:{" "}>7.4f}', end=' ')
	
	overall = dict()
	for si in shifts:
		if si['atm'] not in overall: overall[si['atm']] = list()
		overall[si['atm']].append(si['cs'])
	
	for k in overall.keys():
		val = np.mean(np.array(overall[k]))
		overall[k] = val
	
	outrmse = 0
	for sh in held_set[el]:
		outrmse += (sh['cs'] - overall[sh['atm']])**2
	
	outrmse /= len(held_set[el])
	outrmse = sqrt(outrmse)
	print(f'{outrmse:{" "}>7.4f}', end=' ')
	sumout += outrmse

print(f'{sumin:{" "}>7.4f} {sumout:{" "}>7.4f}')

print(f'{"atm-res":<7}', end=' ')
sumin  = 0
sumout = 0
for el in elms.keys():
	shifts = sets[el]
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
			if s['res'] not in model: model[s['res']] = dict()
			if s['atm'] not in model[s['res']]: model[s['res']][s['atm']] = []
			model[s['res']][s['atm']].append(s['cs'])
		
		for k1 in model.keys():
			for k2 in model[k1].keys():
				cs = np.array(model[k1][k2])
				model[k1][k2] = np.mean(cs)
		
		rmse = 0
		for ts in test:
			rmse += (ts['cs'] - model[ts['res']][ts['atm']])**2
		
		rmse /= len(test)
		rmse = sqrt(rmse)
		#print(rmse)
		val_rmse += rmse
	val_rmse /= 10
	sumin += val_rmse
	print(f'{val_rmse:{" "}>7.4f}', end=' ')
	
	overall = dict()
	for s in shifts:
		if s['res'] not in overall: overall[s['res']] = dict()
		if s['atm'] not in overall[s['res']]: overall[s['res']][s['atm']] = []
		overall[s['res']][s['atm']].append(s['cs'])
	
	for k1 in overall.keys():
		for k2 in overall[k1].keys():
			val = np.mean(np.array(overall[k1][k2]))
			overall[k1][k2] = val
	
	outrmse = 0
	for s in held_set[el]:
		outrmse += (s['cs'] - overall[s['res']][s['atm']])**2
	
	outrmse /= len(held_set[el])
	outrmse = sqrt(outrmse)
	print(f'{outrmse:{" "}>7.4f}', end=' ')
	sumout += outrmse

print(f'{sumin:{" "}>7.4f} {sumout:{" "}>7.4f}')

print(f'{"3meratm":<7}', end=' ')
sumin  = 0
sumout = 0
for el in elms.keys():
	shifts = sets[el]
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
			if s['resl'] == None or s['resr'] == None: continue
			mer = s['resl']+s['res']+s['resr']
			if mer not in model: model[mer] = dict()
			if s['atm'] not in model[mer]: model[mer][s['atm']] = []
			model[mer][s['atm']].append(s['cs'])
		
		for k1 in model.keys():
			for k2 in model[k1].keys():
				val = np.mean(np.array(model[k1][k2]))
				model[k1][k2] = val
		
		rmse = 0
		n = 0
		for ts in test:
			if ts['resl'] == None or ts['resr'] == None: continue
			key = ts['resl']+ts['res']+ts['resr']
			if key in model:
				if ts['atm'] in model[key]:
					rmse += (ts['cs'] - model[key][ts['atm']])**2
					n += 1
		
		rmse /= n
		rmse = sqrt(rmse)
		#print(rmse)
		val_rmse += rmse
	val_rmse /= 10
	sumin += val_rmse
	print(f'{val_rmse:{" "}>7.4f}', end=' ')
	
	overall = dict()
	for s in shifts:
		if s['resl'] == None or s['resr'] == None: continue
		mer = s['resl']+s['res']+s['resr']
		if mer not in overall: overall[mer] = dict()
		if s['atm'] not in overall[mer]: overall[mer][s['atm']] = []
		overall[mer][s['atm']].append(s['cs'])
	
	for k1 in overall.keys():
		for k2 in overall[k1].keys():
			val = np.mean(np.array(overall[k1][k2]))
			overall[k1][k2] = val
	
	outrmse = 0
	n = 0
	for s in held_set[el]:
		if s['resl'] == None or s['resr'] == None: continue
		merk = s['resl']+s['res']+s['resr']		
		if merk in overall:
			if s['atm'] in overall[merk]:
				outrmse += (s['cs'] - overall[merk][s['atm']])**2
				n += 1
	
	outrmse /= n
	outrmse = sqrt(outrmse)
	print(f'{outrmse:{" "}>7.4f}', end=' ')
	sumout += outrmse

print(f'{sumin:{" "}>7.4f} {sumout:{" "}>7.4f}')