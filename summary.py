#!/usr/bin/python3

import argparse
import sys
import math
import numpy as np
import pandas as pd

import sequence_filter

parser = argparse.ArgumentParser(description='Analyze shifts from k-mers')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--blast', '-r', required=False, type=str,
	metavar='<str>', help='Blast report, optional')

def kmer_statistics(sequences, k):
	kmers = dict()
	obs   = 0
	for seq in sequences:
		for i in range(0,len(seq)-k+1):
			word = seq[i:i+k]
			if 'X' in word: continue
			if word not in kmers:
				kmers[word] = True
			obs += 1
	
	uniq_kmers = len(list(kmers.keys()))
	
	return uniq_kmers, obs

def seqs_from_id(df, ids):
	seqs   = list()
	shifts = 0
	for indx, row in df.iloc[0:].iterrows():
		if str(row['name']) in ids:
			seqs.append(row['seq'])
			
			for col in row.keys():
				if col == 'index' or col == 'name' or col == 'seq': continue
				if type(row[col]) == list:
					s  = np.array(row[col])
					cs = np.where(s != None)[0]
					shifts += cs.shape[0]
	
	return seqs, shifts

arg = parser.parse_args()
df = pd.read_json(arg.data,compression='xz').reset_index()

ids = [str(i) for i in df['name'].tolist()]
seqs, shifts = seqs_from_id(df, ids)

all_seqs = dict()
for seq in seqs:
	if seq not in all_seqs:
		all_seqs[seq] = 0
	
	all_seqs[seq] += 1

print(f'Total number of sequences      : {len(seqs):>5d}')
print(f'Total number of chemical shifts: {shifts:>8d}')
print(f'Number of unique sequences     : {len(list(all_seqs.keys())):>5d}')

ksize = range(1,11)

kdata = dict()
for k in ksize:
	uniq, tot = kmer_statistics(seqs, k)
	print(f'k-size: {k:>2d} unique words: {uniq:>7d} obs: {tot:>7d}')

if arg.blast:
	print()
	
	percents = [10, 40, 50, 80, 90]
	
	for p in percents:
		filtered_ids  = sequence_filter.identity_filter(arg.blast, p)
		filtered_seqs, shifts = seqs_from_id(df, filtered_ids)
		n = len(filtered_seqs)
		print(f'Number of sequences after {p}% Filtering: {n:>7d}')
		print(f'Total number of chemical shifts        : {shifts:>7d}')
		
		for k in ksize:
			uniq, tot = kmer_statistics(filtered_seqs, k)
			print(f'k-size: {k:>2d} unique words: {uniq:>7d} obs: {tot:>7d}')
		
		print()
		
"""
Perform same analysis on 10, 40, 50, 80, 90 filters
Number of sequences
Number of shifts
kmers in after filtering
"""




