#!/usr/bin/python3

import os
import pickle
import sys

import graphdot
from graphdot import Graph
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

print(rdkit.__version__)
print(graphdot.__version__)

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

with open(sys.argv[1], 'rb') as fp:
	data = pickle.load(fp)
fp.close()

print(len(data))
newdata = []
count=0
converted = 0
nhc = 0
naz = 0
nvrz = 0
nci = 0
alledges = 0
arsz = 0
astz = 0
sanity = 0
for record in data:
	if record['skipped']: 
		print('how did you get here?')
		sys.exit()
	
	if type(record) == dict:
		mol = record['mol']
	else:
		mol = record
	
	san = Chem.SanitizeMol(mol, catchErrors=True)
	if str(san) != 'SANITIZE_NONE':
		print('sanitize failed')
		continue	
	
	Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
	Chem.rdmolops.AssignStereochemistry(mol)
	record['mol'] = mol
	
	try:
		gmol = Graph.from_rdkit(mol,bond_type='type',set_ring_list=False)
		converted += 1
	except:
		print('from rdkit failed')
		continue
	
	inr = []
	for i, j in zip(gmol.edges['!i'], gmol.edges['!j']):
		bond = mol.GetBondBetweenAtoms(int(i), int(j))
		inr.append(int(bond.IsInRing()))
	
	gmol.edges['inring'] = inr
	gmol.edges['aromatic'] = [int(a) for a in gmol.edges['aromatic']]
	gmol.edges['conjugated'] = [int(c) for c in gmol.edges['conjugated']]
	gmol.edges = gmol.edges.drop(['ring_stereo'])
	
# 	ned = len(gmol.edges['aromatic'])
# 	allzero = 0
# 	rsz     = 0
# 	stz     = 0
# 	for ar, cj, rs, st in zip(gmol.edges['aromatic'], gmol.edges['conjugated'], gmol.edges['ring_stereo'], gmol.edges['stereo']):
# 		if ar == cj and cj == rs and rs == st:
# 			if ar == 0:
# 				allzero += 1
# 		if rs == 0: rsz += 1
# 		if st == 0: stz += 1
# 	
# 	if allzero == ned:
# 		alledges += 1
# 	if rsz == ned:
# 		arsz += 1
# 	if stz == ned:
# 		astz += 1
	
	num = len(gmol.nodes['!i'])
	
	feats     = chem_feature_factory.GetFeaturesForMol(mol)
	donors = []
	acceptors = []
	donors    = [0]*mol.GetNumAtoms()
	acceptors = [0]*mol.GetNumAtoms()
	
	for j in range(len(feats)):
		if feats[j].GetFamily() == 'Donor':
			assert(len(feats[j].GetAtomIds()) == 1)
			idx = feats[j].GetAtomIds()[0]
			assert(donors[idx] == 0)
			donors[idx] = 1
			donors.append(feats[j].GetAtomIds()[0])
		if feats[j].GetFamily() == 'Acceptor':
			assert(len(feats[j].GetAtomIds()) == 1)
			idx = feats[j].GetAtomIds()[0]
			assert(acceptors[idx] == 0)
			acceptors[idx] = 1
	
	#print(donors)
	#print(acceptors)
	gmol.nodes['donors'] = donors
	gmol.nodes['acceptors'] = acceptors
	gmol.nodes['hcount'] = [int(atm.GetTotalNumHs(includeNeighbors=True)) for atm in mol.GetAtoms()]
	deg = [int(a.GetDegree()) for a in mol.GetAtoms()]
	gmol.nodes['degree'] = deg
	aromatic = [int(a.GetIsAromatic()) for a in mol.GetAtoms()]
	gmol.nodes['aromatic'] = aromatic
	
	record['graph'] = gmol
	newdata.append(record)
	
# 	n = mol.GetNumAtoms()
# 	az = 0
# 	vrz = 0
# 	cz = 0
# 	for ch, ci, iv, re in zip(gmol.nodes['charge'], gmol.nodes['chiral'], impv, rade):
# 		if ch == ci and ci == iv and iv == re:
# 			if ch == 0:
# 				az += 1
# 		if iv == re and iv == 0:
# 			vrz += 1
# 		if ci == 0:
# 			cz += 1
# 	
# 	if az == n:
# 		print('all zero')
# 		naz += 1
# 	if vrz == n:
# 		nvrz += 1
# 	if cz == n:
# 		nci += 1

df = pd.DataFrame(newdata)
print(df.head(2))
print(df.shape)
print(df.columns)
df['graphs'] = Graph.unify_datatype(df.graph.to_list())
outfile = f'nmrshiftdb_gdot_fromrdkit.pickle'
df.to_pickle(outfile)

print(f'input {len(data)}')
print(f'sanity check fails: {sanity}')
print(f'converted {converted}')
# print(f'all edge features zero: {alledges} (aromatic, chiral, ring stereo, stereo)')
# print(f'all ring stereo zero: {arsz}')
# print(f'all stereo zero: {astz}')
# print(f'all bond features zero {naz} (charge, chiral, implicit valence, radical e-)')
# print(f'valence and radicals zero: {nvrz}')
# print(f'all chiral 0: {nci}')
