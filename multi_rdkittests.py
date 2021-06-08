#!/usr/bin/python3

import itertools
import json
from multiprocessing import Pool
from multiprocessing import sharedctypes
from multiprocessing import Manager
import sys

import graphdot
from graphdot import Graph
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from xyz2mol import xyz2mol, int_atom

print(rdkit.__version__)
print(graphdot.__version__)

df = pd.read_pickle(sys.argv[1],compression='xz')

print(df.head(5))
print(df.shape)
print(df.columns)

def mol_maker(idx):
	atms = df.symbols[idx]
	coords = df.xyz[idx]
	
	atoms  = [int_atom(atm) for atm in atms]
	
	try:
		mol = xyz2mol(atoms, coords,
		      		  charge=0,
			    	  use_graph=True,
				      allow_charged_fragments=True,
				      embed_chiral=True,
				      use_huckel=True)
	except:
		print('xyz2mol exception')
		sanity += 1
		continue
	
	print('xyz2mol complete')
	if mol is None:
		print('mol is None')
		sanity+=1
		continue
	if len(mol) == 0:
		sanity+=1
		print('mol is empty')
		continue
	else:
		mol = mol[0]
	
	san = Chem.SanitizeMol(mol)
	#print('san value',san)
	if san not in sanity_results:
		sanity_results[san] = 0
	sanity_results[san] += 1