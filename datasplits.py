#!/usr/bin/python3

import json
import lzma
import math
from math import sqrt
import random
import sys

def distance(xyz1, xyz2):
    d = 0
    for p1, p2 in zip(xyz1, xyz2):
        d += (p1 - p2)**2
    
    return sqrt(d)


data = json.load(lzma.open(sys.argv[1],mode='rt',encoding='utf-8'))
print(len(data))
# Kang 2020 filtering 
allowed = ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'Cl']
for i, row in enumerate(data):
	if len(row['shifts']) > 64:
		data.pop(i)
		print('too big')
		continue
	
	skip = False
	for sym in row['symbols']:
		if sym not in allowed:
			skip = True
			break
	
	if skip:
		data.pop(i)
		print('atom not allowed')
		continue
	
	skip = False
	for shift, sym in zip(row['shifts'], row['symbols']):
		if sym == 'C':
			if shift == None:
				skip = True
				break
	if skip:
		data.pop(i)
		print('incomplete assignment')
		continue

print(len(data))

# Identify graphs with bad coordinates, all on top of each other
# removing them from the list
bad = list()
datafilter = list()
for row in data:
	coords = row['xyz']
	smiles = row['smiles']
	
	num = (len(coords)**2 - len(coords))/2
	same = 0
	for i in range(0, len(coords)):
		for j in range(i+1, len(coords)):
			dis = distance(coords[i], coords[j])
			if math.isclose(dis, 0.0):
				same += 1
	
	if same == num: bad.append(row)
	elif same > 0 and same < num:
		print(row)
		print(row['xyz'])
		print('only some of atoms overlap?')
		sys.exit()
	else:
		datafilter.append(row)

print('input number of records', len(data))
print('number of records with bad coords', len(bad))
print('number of records left after filtering', len(datafilter))

# Find rows with same SMILES string, duplicated molecules
smiles = dict()
smiles_inds = dict()
for i, row in enumerate(datafilter):
	smile = row['smiles']
	if smile not in smiles:
		smiles[smile] = 0
		smiles_inds[smile] = list()
	
	smiles[smile] += 1
	smiles_inds[smile].append(i)

dups  = 0
udups = 0
for k, v in sorted(smiles.items(), key=lambda item: item[1], reverse=True):
	if v == 1: continue
	dups += v
	udups += 1 

print('total rows with same SMILES strings', dups)
print('number of unique molecules duplicated', udups)

# Determine within duplicated molecules are true duplicates
datafilter2 = list()
partial = 0
for smkey, inds in smiles_inds.items():
	if smiles[smkey] == 1:
		datafilter2.append(datafilter[inds[0]])
		continue
	
	coord_set = [datafilter[i]['xyz'] for i in inds]
	paired_dups = dict()
	num_atoms = len(coord_set[0])
	num_comps = (len(coord_set)**2 - len(coord_set)) / 2
	
	for i in range(0, len(coord_set)):
		if j in paired_dups.values(): continue
		same = 0
		for j in range(i+1, len(coord_set)):
			for p1, p2 in zip(coord_set[i], coord_set[j]):
				dis = distance(p1, p2)
				if math.isclose(dis, 0.0): 
					#print(p1, p2)
					same += 1
		
		if same == num_atoms:
			paired_dups[i] = j
		elif same > 0 and same < num_atoms:
			partial += 1
	
	if len(list(paired_dups.keys())) == 0:
		for i in inds:
			datafilter2.append(datafilter[i])
	else:
		for i, dex in enumerate(inds):
			if i in paired_dups.keys():
				datafilter2.append(datafilter[dex])
				continue
			elif i in paired_dups.values():
				continue
			else:
				datafilter2.append(datafilter[dex])

print('number of partial overlaps', partial)
print('rows after second filtering', len(datafilter2))

# Shuffle list
x=0
while x < 100:
	random.shuffle(datafilter2)
	x += 1

# Half the data set
half = math.ceil(len(datafilter2)/2)
print(half)

# with open('first_filter.json', 'w') as f:
# 	print(json.dumps(datafilter2), file=f)
# f.close()

with open('new_training_filter_atom.json', 'w') as fp:
	print(json.dumps(datafilter2[:half]), file=fp)
fp.close()

with open('new_validation_filter_atom.json', 'w') as fp1:
	print(json.dumps(datafilter2[half:]), file=fp1)
fp1.close()