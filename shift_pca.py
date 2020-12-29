#!/usr/bin/python3

import argparse
from math import floor
import sys

import datalib as dl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='PCA of Protein NMR Chemical Shift')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--percent', '-p', required=False, default=0.10,
	type=float, help='Percent of data to perform PCA on')
parser.add_argument('--atoms', '-a', required=False, nargs='+', 
	help='atoms to build PCA from', default=['N', 'H', 'CA', 'HA', 'C'])
parser.add_argument('--comp3','-t', action='store_true',
	help='3D plot or 2D')
parser.add_argument('--res','-r', action='store_true',
	help='Label by residue identity')
parser.add_argument('--chem','-c', action='store_true',
	help='Label by chemical type')

aa_res = {
	'P': 0, 'G': 1, 'A': 2, 'V': 3, 'I': 4, 
	'L': 5, 'C': 6, 'M': 7, 'F': 8, 'Y': 9,
	'W':10, 'S':11, 'T':12, 'N':13, 'Q':14,
	'R':15, 'H':16, 'K':17, 'D':18, 'E':19
}

aa_chem = {
	'P':'S', 'G':'S', 'A':'N', 'V':'N', 'I':'N', 
	'L':'N', 'C':'N', 'M':'N', 'F':'N', 'Y':'N',
	'W':'N', 'S':'P', 'T':'P', 'N':'P', 'Q':'P',
	'R':'C', 'H':'C', 'K':'C', 'D':'C', 'E':'C'
}

arg = parser.parse_args()
if not arg.res and not arg.chem:
	print('Must choose label encoding, --res or --chem')
	sys.exit()

cmap = plt.get_cmap('Reds')
if arg.res:
	colors = cmap(np.linspace(0, 1, 20))
	labels = list(aa_res.keys())
if arg.chem:
	colors = cmap(np.linspace(0, 1, 4))
	labels = ['S', 'N', 'P', 'C']

atoms = arg.atoms
atoms.append('seq')

df = pd.read_json(arg.data,compression='xz').sample(frac=1).reset_index()

dfdown = df[atoms].copy().reset_index()
dfdown = dfdown.dropna().copy().reset_index()

complete = list()
res_labels = list()
end = floor(dfdown.shape[0]*arg.percent)

for indx, row in dfdown.iloc[0:end].iterrows():
	
	shifts = list()
	for atm in atoms:
		if atm == 'seq': continue
		shifts.append(row[atm])
	
	for i, ress in enumerate(zip(*shifts)):
		if None in ress: continue
#		print(list(ress), row['seq'][i])
		complete.append(list(ress))
		if arg.res:
			res_labels.append(row['seq'][i])
		else:
			res_labels.append(aa_chem[row['seq'][i]])
#		sys.exit()

# Normalize
mustd = dict()
for atm in atoms:
	if atm == 'seq': continue
	mustd[atm] = (0. , 0.)
	mustd[atm] = dl.mean_std(dfdown, atm)

pca = PCA(n_components=len(atoms)-1)
X = np.array(complete)

for res in X:
	for i in range(len(res)):
		mu = mustd[atoms[i]][0]
		std = mustd[atoms[i]][1]
		res[i] = (res[i] - mu) / std

pca.fit(X)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

new = pca.fit_transform(X)
fig = plt.figure()
if arg.comp3:
	ax = fig.add_subplot(111, projection='3d')
else:
	ax = fig.add_subplot(111)


for i, (color, label) in enumerate(zip(colors, labels)):
	x = []
	y = []
	z = []
	for j, r in enumerate(res_labels):
		if r == label:
			x.append(new[j,0])
			y.append(new[j,1])
			z.append(new[j,2])
	
	if arg.comp3:
		ax.scatter(x, y, z, color=color, label=label)
	else:
		ax.scatter(x, y, color=color, label=label)

ax.set_xlabel('Comp 1')
ax.set_ylabel('Comp 2')

if arg.comp3:
	ax.set_zlabel('Comp 3')

plt.legend(loc='lower right', ncol=5, fontsize=6)
plt.show()

























