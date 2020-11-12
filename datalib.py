#!/usr/bin/python3

"""
Library for manipulation and analysis of BMRB Chemical Shift data
"""

import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
import scipy.stats

aa_encoding = {
	'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 
	'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 
	'M':11, 'N':12, 'P':13, 'Q':14, 'R':15,
	'S':16, 'T':17, 'V':18, 'W':19, 'Y':20,
	'X':21
}

def mean_std(df, atm_name):
	mask = ~df[atm_name].isna()
	
	shifts = np.concatenate(df[atm_name][mask].to_list())
	shifts = shifts[np.where(shifts != None)]
	
	return np.mean(shifts), np.std(shifts)

def zscore_pos(shifts, mean, std):
	zsum = 0
	size = 0
	for cs in shifts:
		if cs is None: continue
		
		size += 1
		zscore = abs((cs - mean) / std)
		zsum += zscore
	
	if zsum == 0 or size == 0:
		return 0.0
	else:
		return zsum / size

def make_zscores(df, atm_types):
	ztable = dict()
	
	for atm in atm_types:
		mean, std = mean_std(df, atm)
		
		for i, (id, seq, bool) in enumerate(zip(df.name, df.seq, df[atm].isna())):
		
			if i in ztable:
				ztable[i][atm] = 0.0
			else:
				ztable[i] = dict()
				ztable[i]['id'] = str(id)
				ztable[i][atm] = 0.0
				ztable[i]['sum'] = 0.0
			
			if bool: continue
			
			ztable[i][atm] = zscore_pos(df[atm][i], mean, std)
			ztable[i]['sum'] += ztable[i][atm]
	
	zreport = pd.DataFrame(ztable).transpose()
	return zreport

def filter_sigma(report, sigma, atom_types):
	
	#print(len(report['id'].to_list()))
	report = report[report['sum'] != 0]
	#print(len(report['id'].to_list()))
	for atm in atom_types:
		report = report[report[atm] < sigma]
		#print(len(report['id'].to_list()))
	
	print(report)
	return report['id'].to_list()

def clean_atom_set(dataset, name, q):
	assert((q >= 0) & (q <= 1.0))
	
	mask = ~dataset[name].isna()
	
	shifts = np.concatenate(dataset[name][mask].to_list())
	shifts = shifts[np.where(shifts != None)]
	
	lower = (1.0 - q)/2
	upper = 1 - lower
	
	lowq = np.quantile(shifts, lower)
	uppq = np.quantile(shifts, upper)
	
	shifts_q = shifts[(shifts >= lowq) & (shifts <= uppq)]
	
	return shifts_q

def stats_report(data,atom_name):
	report = {
		'name'   : atom_name,
		'mean'   : np.round(np.mean(data),4),
		'std'    : np.round(np.std(data),4),
		'median' : np.round(np.median(data),4),
		'min'    : np.round(np.amin(data),4),
		'max'    : np.round(np.amax(data),4),
		'number' : data.shape[0]
	}
	
	return report

def atom_pairs_corr(dataset, name1, name2, q):
	assert((q >= 0) & (q <= 1.0))
	lowq = (1.0 -q)/2
	uppq = 1.0 - lowq
	
	df = dataset[dataset[name1].notnull() & dataset[name2].notnull()]
	df = df[[name1, name2]]
	
	shifts1 = np.concatenate(df[name1].to_list())
	shifts2 = np.concatenate(df[name2].to_list())
	
	assert(shifts1.shape == shifts2.shape)
	
	if shifts1.shape[0] == 0:
		return None
	
	s1 = list()
	s2 = list()
	
	s1_lower = np.quantile(shifts1[np.where(shifts1 != None)], lowq)
	s1_upper = np.quantile(shifts1[np.where(shifts1 != None)], uppq)
	
	s2_lower = np.quantile(shifts2[np.where(shifts2 != None)], lowq)
	s2_upper = np.quantile(shifts2[np.where(shifts2 != None)], uppq)
	
	for x1, x2 in zip(shifts1, shifts2):
		if (x1 is not None) & (x2 is not None):
			if ((x1 >= s1_lower) & (x1 <= s1_upper) & (x2 >= s2_lower) & (x2 <= s2_upper)):
				s1.append(x1)
				s2.append(x2)
	
	if len(s1) == 0: return None
	
	r = np.corrcoef(np.array(s1), np.array(s2))
	
	if np.isnan(np.sum(r)): return None
	else: return r[0,1]

def cs_by_res(df, atom_type):
	shifts = list()
	aminos = list()
	for seq, cs, bool in zip(df['seq'], df[atom_type], df[atom_type].isna()):
		if bool: continue
		
		for i, c in enumerate(cs):
			if c is None: continue
			
			shifts.append(c)
			aminos.append(aa_encoding[seq[i]])
	
	return shifts, aminos

def strong_filter(df, atoms, sigma):
	print(df.shape)
	dfcopy = df.copy()
	dfcopy.set_index('name',inplace=True)
	
	dropped = list()
	for atm in atoms:
		stats = stats_report(clean_atom_set(dfcopy, atm, 1.0),atm)
		mu  = stats['mean']
		std = stats['std']
		
		hi = mu + sigma*std
		lo = mu - sigma*std
		
		for id, shifts, bool in zip(df['name'], df[atm], df[atm].isna()):
			if bool: continue
			
			if id in dropped: continue
			
			#print(id)
			for cs in shifts:
				if cs is None: continue
				
				if cs >= hi:
					dfcopy = dfcopy.drop(id)
					dropped.append(id)
					#print(id)
					break
				
				if cs <= lo:
					dfcopy = dfcopy.drop(id)
					dropped.append(id)
					#print(id)
					break
	
	return dfcopy