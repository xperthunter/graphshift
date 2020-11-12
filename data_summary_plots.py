import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Summary plots for BMRB JSON Data')
parser.add_argument('--full', required=True, type=str,
	metavar='<str>', help='json db for full BMRB')
parser.add_argument('--filtered', required=True, type=str,
	metavar='<str>', help='json db for filtered BMRB')
#parser.add_argument('--out', required=True, type=str,
#	metavar='<str>', help='directory to save plots')                        

def total_shifts(df):
	total = 0
	for cn in df.columns.values:
		if cn == 'index' or cn == 'name' or cn == 'seq': continue
		
		mask = ~df[cn].isna()
		shifts = np.concatenate(df[cn][mask].to_list())
		shifts = shifts[np.where(shifts != None)]
		
		total += shifts.shape[0]
	
	return total

def seq_sizes(df):
	sizes = list()
	
	for seq in df['seq']:
		sizes.append(len(seq))
	
	return sizes

def shifts_per(df):
	cs_per = list()
	
	for ind, ids in zip(df['index'],df['name']):
		total = 0
		
		for cn in df.columns.values:
			if cn == 'index' or cn == 'name' or cn == 'seq': continue
			
			if isinstance(df.loc[ind, cn], list) is False: continue
			
			shifts = np.array(df.loc[ind, cn])
			shifts = shifts[np.where(shifts != None)]
			
			total += shifts.shape[0]
		
		cs_per.append(total)
	
	return cs_per

def complete_per(df, atom):
	comp_per = list()
	
	for id, shifts, bool in zip(df['name'], df[atom], df[atom].isna()):
		if bool: continue
		
		shifts = np.array(shifts)
		mask = shifts != None
		
#		print(np.sum(mask), shifts.shape[0])
		comp_per.append(np.sum(mask)/shifts.shape[0])
	
	return comp_per

# grabbing all the args                                                          
arg = parser.parse_args()

dfull = pd.read_json(arg.full,compression='xz').reset_index()
dfilt = pd.read_json(arg.filtered,compression='xz').reset_index()


rows_full = dfull.shape[0]
rows_filt = dfilt.shape[0]

ids_full = dfull['name'].to_list()
ids_filt = dfilt['name'].to_list()

ids_full = len(set(ids_full))
ids_filt = len(set(ids_filt))

nfull = total_shifts(dfull)
nfilt = total_shifts(dfilt)

sfull = os.stat(arg.full).st_size / (1024 * 1024)
sfilt = os.stat(arg.filtered).st_size / (1024 * 1024)
sfull = f'{sfull:.1f} MB'
sfilt = f'{sfilt:.1f} MB'

print(f"{'full':>16}{'filtered':>10}")
print(f"{'rows':<8}{rows_full:>8}{rows_filt:>10}")
print(f"{'ids':<8}{ids_full:>8}{ids_filt:>10}")
print(f"{'shifts':<6}{nfull:>10}{nfilt:>10}")
print(f"{'size':<9}{sfull:>7}{sfilt:>10}")
"""
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,5))
fig.suptitle('Protein sequence length distribution')
axs[0].hist(seq_sizes(dfull))
axs[0].set_title('Full BMRB Set')
axs[1].hist(seq_sizes(dfilt))
axs[1].set_title('Filtered BMRB Set')

plt.show()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,5))
fig.suptitle('Chemical shifts per protein sequence')
axs[0].hist(shifts_per(dfull))
axs[0].set_title('Full BMRB Set')
axs[1].hist(shifts_per(dfilt))
axs[1].set_title('Filtered BMRB Set')

plt.show()

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
fig.suptitle('Percent Complete shift assignments for Backbone Atoms')
axs[0,0].hist(complete_per(dfull, 'N'))
axs[0,0].set_title('N shift completion -- Full BMRB')
axs[0,1].hist(complete_per(dfilt, 'N'))
axs[0,1].set_title('N shift completion -- Filtered BMRB')
axs[1,0].hist(complete_per(dfull, 'H'))
axs[1,0].set_title('H shift completion -- Full BMRB')
axs[1,1].hist(complete_per(dfilt, 'H'))
axs[1,1].set_title('H shift completion -- Filtered BMRB')
axs[2,0].hist(complete_per(dfull, 'CA'))
axs[2,0].set_title('CA shift completion -- Full BMRB')
axs[2,1].hist(complete_per(dfilt, 'CA'))
axs[2,1].set_title('CA shift completion -- Filtered BMRB')
axs[3,0].hist(complete_per(dfull, 'HA'))
axs[3,0].set_title('HA shift completion -- Full BMRB')
axs[3,1].hist(complete_per(dfilt, 'HA'))
axs[3,1].set_title('HA shift completion -- Filtered BMRB')
axs[4,0].hist(complete_per(dfull, 'C'))
axs[4,0].set_title('C shift completion -- Full BMRB')
axs[4,1].hist(complete_per(dfilt, 'C'))
axs[4,1].set_title('C shift completion -- Filtered BMRB')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
"""