#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
import sys

import sequence_filter

parser = argparse.ArgumentParser(description='Make shift substitution matrix')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--flank','-f', required=True, type=int,
	metavar='<int>', help='number of flanking residues')
parser.add_argument('--atom','-a', required=True, type=str,
	metavar='<str>')
parser.add_argument('--blast','-b', required=False, type=str,
	metavar='<str>', help='blast tabular report')

arg = parser.parse_args()
k   = arg.flank
atm = arg.atom
alphabet = ['A','C','D','E','F',
			'G','H','I','K','L',
			'M','N','P','Q','R',
			'S','T','V','W','Y']

df = pd.read_json(arg.data,compression='xz').sample(frac=1.0).reset_index()

if arg.blast:
	keep = sequence_filter.identity_filter(arg.blast, 75)
	df = df[df['name'].isin(keep)]

kmers = dict()

for id, seq, shifts, bool in zip(df.name, df.seq, df[atm], df[atm].isna()):
	if bool: continue
	
	for i in range(k,len(seq)-k):
		if shifts[i] is None: continue
		if seq[i] not in alphabet: continue
		
		kmer = seq[i-k:i+k+1]
		
		if kmer not in kmers: kmers[kmer] = list()
		
		kmers[kmer].append(shifts[i])

matrix = dict()
for word in kmers:
	if word[k] not in matrix: matrix[word[k]] = dict()
	w = list(word)
	for aa in alphabet:
		w[k] = aa
		neword = ''.join(w)
		if neword[k] not in matrix[word[k]]: matrix[word[k]][neword[k]] = list()
		if neword == word:
			matrix[word[k]][neword[k]].extend(kmers[word])
			continue
		if neword in kmers:
			for rowobs in kmers[word]:
				for colobs in kmers[neword]:
					matrix[word[k]][neword[k]].append(colobs - rowobs)

print(' ',end='')
for ah in alphabet:
	print(f'{ah:>7s}', end='')
print()
for i, ai in enumerate(alphabet):
	print(ai, end=' ')
	for j, aj in enumerate(alphabet):
		if ai not in matrix:
			print(f'{0:>6d}', end=' ')
			continue
		elif len(matrix[ai][aj]) == 0:
			print(f'{0:>6d}', end=' ')
			continue
		#if j < i:
		#	print(f'{"":8s}',end='  ')
		if ai == aj:
			sd = np.std(np.array(matrix[ai][aj]))
			print(f'{sd:{" "}>+6.2f}',end=' ')
		else:
			avg_diff = np.mean(np.array(matrix[ai][aj]))
			print(f'{avg_diff:{" "}>+6.2f}',end=' ')
	print()
		

	

"""
average change when changing central amino acid from row res to column res
standard deviation of change

matrix = dict()
for word in kmer:
	if word[k] not in matrix: matrix[word[k]] = dict()
	for aa in alphabet:
		w = list(word)
		w[k] = aa
		neword = ''.join(w)
		if neword == word: continue
		if neword in kmer:
			diffs = list()
			for rowobs in kmer[word]:
				for colobs in kmer[neword]:
					diffs.append(colobs - rowobs)
			
			avg_diff = np.mean(np.array(diffs))
			std_diff = np.std(np.array(diffs))
			
			matrix[word[k]][neword[k]] = (avg_diff, std_diff)

for ai in alphabet:
	
	for aj in alphabet:
		avg = matrix[ai][aj][0]
		sd  = matrix[ai][aj][1]
		
		print(f'{}')	

"""