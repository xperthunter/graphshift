#!/usr/bin/python3

import sys
import csv
import numpy as np
import pandas as pd

dataset = pd.read_json(sys.argv[1],compression='xz').reset_index()

def find_outliers(df, atom, min, max, res):
	
	min = float(min)
	max = float(max)
	
	outliers = 0
	good = 0
	total = 0
	
	out_shifts = []
	
	for seq, shifts, bool, id in zip(df['seq'], df[atom], df[atom].isna(), df['name']):
		
		if bool: continue
		
		for i, cs in enumerate(shifts):
			if cs is None: continue
			if seq[i] == res:
				total += 1
				if cs > max: 
					outliers += 1
					out_shifts.append(cs)
#					yield id
				elif cs < min:
					outliers += 1
					out_shifts.append(cs)
#					yield id
				else:
					good += 1
		
	return total, good, outliers, np.mean(np.array(out_shifts)), np.std(np.array(out_shifts))

with open(sys.argv[2], mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	
	for row in csv_reader:
		aa  = row['aa']
		at  = row['atom_type']
		fg  = row['functional_group']
		min = row['min']
		max = row['max']
		
		if min == '' or max == '': continue
		
		df = dataset[['name','seq',at]].copy().reset_index()
		
		total, passed, failed, mean, std = find_outliers(df, at, min, max, aa)
		print(f'{aa}  {at}  {failed}, {min}, {max}, {total}, {mean:.4f}, {std:.4f}')

