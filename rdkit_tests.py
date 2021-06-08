#!/usr/bin/python3

import json
import multiprocessing
import multiprocessing.pool
import pickle
import sys
import time

import graphdot
from graphdot import Graph
import pandas as pd
import rdkit
from rdkit import Chem
import timeout_decorator
from xyz2mol import xyz2mol, int_atom

print(rdkit.__version__)
print(graphdot.__version__)

class NoDaemonProcess(multiprocessing.Process):
	@property
	def daemon(self):
		return False
	@daemon.setter
	def daemon(self, value):
		pass

class MyPool(multiprocessing.pool.Pool):
	def Process(self, *args, **kwargs):
		proc = super(MyPool, self).Process(*args, **kwargs)
		proc.__class__ = NoDaemonProcess
		
		return proc

with open(sys.argv[1], 'rt') as fp:
	records = json.load(fp,encoding='utf-8')
fp.close()
print(len(records))

# class MyTimeException(StopIteration):
# 	def __init__(self):
# 		return None 

@timeout_decorator.timeout(300, use_signals=False, timeout_exception=StopIteration)
def make_mol(rec):
	atoms = [int_atom(atm) for atm in rec['symbols']]
	coords = rec['xyz']
	rec['skipped'] = True
	rec['mol']     = None
	try:
		mol = xyz2mol(atoms, coords,
					  charge=0,
					  use_graph=True,
					  allow_charged_fragments=True,
					  embed_chiral=True,
					  use_huckel=True)
		if mol is None:
			rec['skipped'] = True
			rec['mol']     = None
			return rec
		if len(mol) == 0:
			rec['skipped'] = True
			rec['mol']     = None
			return rec
		else:
			mol = mol[0]
			rec['skipped'] = False
			rec['mol']     = mol
			return rec
	except Exception:
		rec['skipped'] = True
		rec['mol']     = None
		return rec
	except:
		rec['skipped'] = True
		rec['mol']     = None
		return rec

pool = MyPool(32)
res = pool.map(make_mol, records)
pool.close()
pool.join()
sanity = 0

print("going through results")
print(f'len results: {len(res)}')
completed = []
for rec in res:
	#print(rec)
	if rec is None:
		sanity += 1
		continue
	if rec['skipped']:
		if rec['mol'] != None:
			print('how is skipped and mol not none?')
			print(rec)
			sys.exit()
		sanity += 1
		continue
	else:
		completed.append(rec)
		try:
			smiles = rdkit.Chem.MolToSmiles(rec['mol'], isomericSmiles=True)
			print(smiles)
		except:
			print('no smiles')
			continue
		
print(f'len of input: {len(records)}')
print(f'len results: {len(res)}')
print(f'failed sanity check: {sanity}')

with open('completed.pickle', 'wb') as fo:
	pickle.dump(completed, fo)
fo.close()