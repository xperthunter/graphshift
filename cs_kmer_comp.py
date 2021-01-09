#!/usr/bin/python3

import argparse
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datalib import mean_std

parser = argparse.ArgumentParser(description='Analyze shifts from k-mers')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--kmer','-k', required=True, type=int,
	metavar='<int>', help='number of flanking residues')
parser.add_argument('--atoms', '-a', required=False, nargs='+', 
	help='build kmer vectors for these atomc chemical shifts', 
	default=['N', 'H', 'CA', 'HA', 'C'])

def shift_metric(shift_1, shift_2):
	dis = 0
	n   = 0
	for s1, s2 in zip(shift_1, shift_2):
		if s1 is None or s2 is None: 
			return None
		dis += math.fabs(s1 - s2)
	
	return dis

def norm_pd(df, cols, metrics):
	for indx, row in df.iloc[0:].iterrows():
		for c, na in zip(cols, row[cols].isna()):
			if na: continue
			
			array = np.array(row[c])
			array = np.where(array == None, np.nan, array)
			
			array = array - metrics[c][0]
			array = array / metrics[c][1]
			
			array = np.where(pd.isnull(array), None, array)
			
			row[c] = array.tolist()
			
			df.at[indx,c] = row[c]
	
	return df

def pairs(shifts):
	for i in range(len(shifts)):
		for j in range(i, len(shifts)):
			yield shifts[i], shifts[j]

def fill_nones(row, colnames, length):
	for cn in colnames:
		if type(row[cn]) != list:
			row[cn] = [None for i in range(length)]
	
	return row

arg = parser.parse_args()
k = arg.kmer
atoms = arg.atoms
colnames = ['name', 'seq']
colnames.extend(arg.atoms)


df = pd.read_json(arg.data,compression='xz').reset_index()
#print(df.shape)
bb_df = df[colnames].copy().reset_index()
bb_df = bb_df.dropna().copy().reset_index()

atm_cols = list(bb_df.columns)[4:]

xstd = dict()
for atm in atm_cols:
	mean, sd = mean_std(df, atm)
	xstd[atm] = (mean, sd)

bb_df = norm_pd(bb_df, atm_cols, xstd)

kmer_shifts = dict()
for indx, row in bb_df.iloc[0:].iterrows():
	seq = row['seq']
	
	for i in range(0,len(seq)-k+1):
		kmer = seq[i:i+k]
		
		kshifts = []
		kshifts = row['N'][i:i+k]
#		kshifts.extend(row['H'][i:i+k])
#		kshifts.extend(row['CA'][i:i+k])
#		kshifts.extend(row['HA'][i:i+k])
#		kshifts.extend(row['C'][i:i+k])
	
	if kmer not in kmer_shifts:
		kmer_shifts[kmer] = list()
	
	kmer_shifts[kmer].append(kshifts)

kmer_metrics = dict()
for word in kmer_shifts.keys():
#	print(word, len(kmer_shifts[word]))
	if len(kmer_shifts[word]) == 0: continue
	for l1, l2 in pairs(kmer_shifts[word]):
		score = shift_metric(l1, l2)
		if score is None: continue
#		print(f'{score:.4f}')
		if word not in kmer_metrics:
			kmer_metrics[word] = list()
		kmer_metrics[word].append(score)

kmer_outs = dict()
for iword in kmer_shifts.keys():
	for jword in kmer_shifts.keys():
		if iword == jword: continue
		
		for ishift in kmer_shifts[iword]:
			for jshift in kmer_shifts[jword]:
				score = shift_metric(ishift, jshift)
				if score is None: continue
				
				if iword not in kmer_outs:
					kmer_outs[iword] = []
				
				kmer_outs[iword].append(score)

#print('word num instd outstd inmin?')				
for word in kmer_metrics:
	if len(kmer_metrics[word]) == 1: continue
	#print(word)
	ins  = np.array(kmer_metrics[word])
	outs = np.array(kmer_outs[word])
	
	n      = ins.shape[0]
	stdin  = np.std(ins)
	stdout = np.std(outs)
	
	inmin  = np.min(ins)
	outmin = np.min(outs)
	
	minbool = inmin < outmin
	
	inmu  = np.mean(ins)
	outmu = np.mean(outs)
	
	overlap = np.where(ins > outmin)[0].shape[0]/n
	
	print(f'{word:>{k}s} {n:5d} {stdin:7.4f} {stdout:7.4f} {minbool}', end='')
	print(f'{inmu:7.4f} {outmu:7.4f} {overlap:1.4f}')
	
	"""
	plt.hist(ins)
	plt.show()
	plt.hist(outs)
	plt.show()
	sys.exit()
	"""

sys.exit()
"""

Notes
1. diff in stds
2. % below out min
3. normalize all shifts
4. if outstd > instd
5. plot with all combined
	- in diffs vs out diffs
throw out nones
replace with average
skip them in difference calculations
kmers are in same sequence
chem type encodings

"""