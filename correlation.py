#!/usr/bin/python3

import argparse
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datalib import mean_std
#import blosum

parser = argparse.ArgumentParser(description='Analyze shifts from k-mers')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--atoms', '-a', required=False, nargs='+', 
	help='build kmer vectors for these atomc chemical shifts', 
	default=['N', 'H', 'CA', 'HA', 'C'])

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

def metrics(df, atms):
	xstd = dict()
	for atm in atms:
		mean, sd = mean_std(df, atm)
		xstd[atm] = (mean, sd)
	
	return xstd

arg = parser.parse_args()
atoms = arg.atoms
colnames = ['name', 'seq']
colnames.extend(arg.atoms)

df = pd.read_json(arg.data,compression='xz').reset_index()
#print(df.shape)
bb_df = df[colnames].copy().reset_index()
#bb_df = bb_df.dropna().copy().reset_index()

xstd = metrics(bb_df, atoms)

bb_df = norm_pd(bb_df, atoms, xstd)

chain_shifts = []

for indx, row in bb_df.iloc[0:].iterrows():
	seq   = row['seq']
	chain = []
	
	for i in range(len(seq)):
		for atm in atoms:
			if type(row[atm]) != list:
				chain.append(None)
			else:
				chain.append(row[atm][i])
	
	chain_shifts.append(chain)

# calculate avg

avg   = 0
avgsq = 0
n     = 0
for chain in chain_shifts:
	for shift in chain:
		if shift is None: continue
#		print(f'{shift:.4f}')
		avg   += shift
		avgsq += shift**2
		n     += 1
avg   /= n
avgsq /= n

print(f'<i>: {avg:.4E} <i^2>: {avgsq:.4E} <i>^2: {avg**2:.4E} n: {n}')

seqs = range(1,31)

for s in seqs:
	ijs = 0
	n   = 0
	for chain in chain_shifts:
		if len(chain) < s: continue
		for i in range(0,len(chain)-s):
			if chain[i] is None or chain[i+s] is None: continue
			ij = chain[i]*chain[i+s]
			ijs += ij
			n   += 1
	
	ijs /= n
	corr = (ijs - avg**2)/(avgsq - avg**2)
	print(f'{s} {corr:.4E} {n}')



"""
- average of pairs
- average 1 squared
- average square
N H CA HA C
if at N, 1 is H and CA
if at H, 1 is N
if at CA, 1 is HA C
if at C, 1 is N

for k
for i in shift list

"""
















