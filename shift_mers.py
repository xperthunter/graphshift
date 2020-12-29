#!/usr/bin/python3

import argparse
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Analyze shifts from k-mers')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
#parser.add_argument('--kmer','-k', required=True, type=int,
#	metavar='<int>', help='number of flanking residues')
parser.add_argument('--res','-r', required=False, type=str,
	metavar='<str>', help='central amino acid')
parser.add_argument('--atom','-a', required=True, type=str,
	metavar='<str>')

arg = parser.parse_args()
#k = arg.kmer
atm = arg.atom

df = pd.read_json(arg.data,compression='xz').reset_index()

#kmer_shifts = dict()
ls = [1,2,3,4]

in_k = list()
un_x = list()

for k in ls:
	kmer_shifts = dict()
	for id, seq, shifts, bool in zip(df.name, df.seq, df[atm], df[atm].isna()):
		if bool: continue
		
		for i in range(k,len(seq)-k):
			if shifts[i] is None: continue
			
			if arg.res:
				if seq[i] == arg.res:
					kmer = seq[i-k:i+k+1]
				else: continue
			else:
				kmer = seq[i-k:i+k+1]
			
			if kmer in kmer_shifts:
				kmer_shifts[kmer].append(shifts[i])
			else:
				kmer_shifts[kmer] = list()
				kmer_shifts[kmer].append(shifts[i])
	
	ms = list()
	sigs = list()
	for k in kmer_shifts.keys():
		n = len(kmer_shifts[k])
		if n > 1:
			mean = np.mean(np.array(kmer_shifts[k]))
			std = np.std(np.array(kmer_shifts[k]))
			if math.isclose(std, 0):
				snr = -1
			else:
				snr = mean/std
#			print(f'{k} {n} {snr:.4f} {mean:.4f} {std:.4f}')
			ms.append(mean)
			sigs.append(std)
#		else:
#			print(f'{k} {n} -1')
		
	un_x.append(np.mean(np.array(sigs)))
	in_k.append(np.std(np.array(ms)))

print(un_x)
print(in_k)

plt.plot(ls, un_x, 'ko-', label='Unexplainded (E[sigma])')
plt.plot(ls, in_k, 'ro-', label='Std of kmer averages')
plt.legend(loc='center right')
plt.title(f'kmer variance analysis, atom: {atm}')
plt.xticks(ls)
#plt.xlabels(['1','2','3','4','5'])
plt.show()