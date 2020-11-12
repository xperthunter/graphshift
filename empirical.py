#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mean_std(df, name):
	
	mask = ~df[name].isna()
	
	shifts = np.concatenate(df[name][mask].to_list())
	shifts = shifts[np.where(shifts != None)]
	
	return np.mean(shifts), np.std(shifts)

atoms = ['H', 'N', 'CA', 'HA', 'C']

df = pd.read_json(sys.argv[1],compression='xz').reset_index()

ztable = dict()
for atm in atoms:
	mean, std = mean_std(df, atm)
#	print(f'{atm} {mean} {std}')
	for i, (id, seq, bool) in enumerate(zip(df.name, df.seq, df[atm].isna())):
		
		if i in ztable:
			ztable[i][atm] = 0.0
		else:
			ztable[i] = dict()
			ztable[i]['id'] = str(id)
			ztable[i][atm] = 0.0
#			ztable[name]['sum'] = np.nan
		
		if bool: continue
		
		zsum = 0
		size = 0
		for cs in df[atm][i]:
			if cs is None: continue
			size += 1
			zscore = abs((cs - mean)/std)
			zsum += zscore
		
		if zsum == 0 or size == 0: continue
		ztable[i][atm] = zsum / size

for n in ztable.keys():
	vals = list()
	for k in ztable[n].keys():
		if k == 'id': continue
		vals.append(ztable[n][k])
	
	ztable[n]['sum'] = np.nansum(np.array(vals))


zr = pd.DataFrame(ztable).transpose()
#print(zscore_report.head(5))

zr = zr.sort_values('N', na_position='first')
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('precision', 4)
print(zr)

sig = 3.0
mask = (zr['HA'] < sig) & (zr['sum'] != 0) & (zr['H'] < sig) & (zr['N'] < sig) & (zr['CA'] < sig) & (zr['C'] < sig)

ids = zr['id'][mask].to_list()
print(len(ids))
"""
		
df = dataset[['name','HA']].copy().reset_index()
df = df[~df['HA'].isna()].copy().reset_index()

shifts = np.concatenate(df['HA'].to_list())
shifts = shifts[np.where(shifts != None)]

mean = np.mean(shifts)
std  = np.std(shifts)

print(f'mean: {mean} std: {std}')

ztable = dict()

for name, ha in zip(df.name, df.HA):
	zsum = 0
	size = 0
	for cs in ha:
		if cs is None: continue
		size += 1
		zscore = abs((cs - mean)/std)
		zsum += zscore
	
	if zsum == 0 or size == 0:
		continue
#		print(name, ha)
#		sys.exit()
	ztable[name] = zsum / size

for k, v in sorted(ztable.items(), key=lambda item: item[1]):
	print(k,v)

#plt.hist(ztable.values())
#plt.hist([n for n in ztable.values() if (n > -1.38) & (n < 1.38)])
#plt.show()	

do the same for bb atom types
last column is sum
"""