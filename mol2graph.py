#!/usr/bin/python3

import argparse
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

parser = argparse.ArgumentParser(description=''.join(('Make dataframes from',
	'Kang 2020 pickled datasets')))
parser.add_argument('--data', '-d', required=True, type=str,
	metavar='<str>', help='13C/1H pickle dataset from Kang2020')
parser.add_argument('--atom', '-a', required=True, type=str,
	metavar='<str>', help='atom type for dataset')

arg = parser.parse_args()

print(rdkit.__version__)
print(graphdot.__version__)

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def make_shiftlist(rdmols, spids, cs):
	new_data = []
	for m, i, v in zip(rdmols, spids, cs):
		dic = {}
		dic = {'rdmol':m, 'graph':None, 'shifts':[]}
		shifts = []
		
		if len(i) != len(v): 
			print('why i and v not the same?')
			sys.exit()
		
		if len(i) > 1:
			print(i)
			print(v)
			print('why more than 1?')
			sys.exit()
		
		sv = v[0]
		
		for a in m.GetAtoms():
			if int(a.GetIdx()) in sv:
				shifts.append(sv[int(a.GetIdx())])
			else: shifts.append(None)
		
		if len(shifts) != m.GetNumAtoms():
			print('missing some shifts')
			sys.exit()
	
		dic['shifts'] = shifts
		new_data.append(dic)
	
	return new_data

def make_graphs(datalist):
	dataset = []
	for record in datalist:
		assert(record['graph'] == None)
		mol = record['rdmol']
		
		san = Chem.SanitizeMol(mol, catchErrors=True)
		if str(san) != 'SANITIZE_NONE': continue
		
		Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
		Chem.rdmolops.AssignStereochemistry(mol)
		record['rdmol'] = mol
		
		try:
			gmol = Graph.from_rdkit(mol,bond_type='type',set_ring_list=False)
			record['graph'] = gmol
		except:
			continue
		
		# complete edge features with in ring bool
		inr = []
		for i, j in zip(gmol.edges['!i'], gmol.edges['!j']):
			bond = mol.GetBondBetweenAtoms(int(i), int(j))
			inr.append(int(bond.IsInRing()))
		
		gmol.edges['inring'] = inr
		gmol.edges['aromatic'] = [int(a) for a in gmol.edges['aromatic']]
		gmol.edges['conjugated'] = [int(c) for c in gmol.edges['conjugated']]
		gmol.edges = gmol.edges.drop(['ring_stereo'])
		
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
			if feats[j].GetFamily() == 'Acceptor':
				assert(len(feats[j].GetAtomIds()) == 1)
				idx = feats[j].GetAtomIds()[0]
				assert(acceptors[idx] == 0)
				acceptors[idx] = 1
		
		gmol.nodes['donors'] = donors
		gmol.nodes['acceptors'] = acceptors
		gmol.nodes['hcount'] = [int(atm.GetTotalNumHs(includeNeighbors=True)) for atm in mol.GetAtoms()]
		deg = [int(a.GetDegree()) for a in mol.GetAtoms()]
		gmol.nodes['degree'] = deg
		aromatic = [int(a.GetIsAromatic()) for a in mol.GetAtoms()]
		gmol.nodes['aromatic'] = aromatic
		
		dic = {}
		dic = {'rdmol':mol, 'graph':gmol, 'shifts':record['shifts']}
		dataset.append(dic)
	return dataset
		
with open(arg.data, 'rb') as fp:
	data = pickle.load(fp)
fp.close()
print(data['train_df'].keys())
print(len(data['train_df']['rdmol']))
print('train set-up')
withshifts = make_shiftlist(data['train_df']['rdmol'], data['train_df']['spectra_ids'], data['train_df']['value'])
complete = make_graphs(withshifts)

df = pd.DataFrame(complete)
print(df.head(2))
print(df.shape)
print(df.columns)
df['graphs'] = Graph.unify_datatype(df.graph.to_list())
outfile = f'kang2020_{arg.atom}_train.pickle'
df.to_pickle(outfile)
print('train completed')
print(len(data['test_df']['rdmol']))
print('test set-up')
withshiftst = make_shiftlist(data['test_df']['rdmol'], data['test_df']['spectra_ids'], data['test_df']['value'])
completet = make_graphs(withshiftst)

dft = pd.DataFrame(completet)
print(dft.head(2))
print(dft.shape)
print(dft.columns)
dft['graphs'] = Graph.unify_datatype(dft.graph.to_list())
outfile = f'kang2020_{arg.atom}_test.pickle'
dft.to_pickle(outfile)
print('test completed')
