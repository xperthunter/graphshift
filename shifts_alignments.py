#!/usr/bin/python3

import argparse
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datalib import mean_std
import blosum

parser = argparse.ArgumentParser(description='Analyze shifts from k-mers')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--kmer','-k', required=True, type=int,
	metavar='<int>', help='number of flanking residues')
parser.add_argument('--atoms', '-a', required=False, nargs='+', 
	help='build kmer vectors for these atomc chemical shifts', 
	default=['N', 'H', 'CA', 'HA', 'C'])
parser.add_argument('--matrix', '-m', required=True, type=str,
	metavar='<str>', help='file for substitution matrix')

def shift_metric(shift_1, shift_2):
	dis = 0
	n   = 0
	for s1, s2 in zip(shift_1, shift_2):
		if s1 is None or s2 is None: 
			return None
		dis += (s1 - s2)**2
	
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

arg = parser.parse_args()
k = arg.kmer
atoms = arg.atoms
colnames = ['name', 'seq']
colnames.extend(arg.atoms)
print(colnames)
blosum_matrix = blosum.make_blosum(arg.matrix)

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
		
		for atm in atm_cols:
			kshifts.extend(row[atm][i:i+k])
	
	if kmer not in kmer_shifts:
		kmer_shifts[kmer] = list()
	
	kmer_shifts[kmer].append(kshifts)

aligns = list()
scores = list()
comps = dict()
for iword in kmer_shifts.keys():
	for jword in kmer_shifts.keys():
		if (iword, jword) in comps or (jword, iword) in comps: continue
		comps[(iword, jword)] = True
		comps[(jword, iword)] = True
		
		for ishift in kmer_shifts[iword]:
			for jshift in kmer_shifts[jword]:
				score = shift_metric(ishift, jshift)
				if score is None: continue
				
				align_score = blosum.align_score(iword, jword, blosum_matrix)
				
				aligns.append(align_score)
				scores.append(score)

plt.plot(scores, aligns, 'ko')
plt.show()




















