#!/usr/bin/python3

import argparse
import sys
import csv
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Find outliers relative to ideal ranges')
parser.add_argument('--data','-d', required=True, type=str,
	metavar='<str>', help='json db')
parser.add_argument('--ranges','-a', required=True, type=str,
	metavar='<str>', help='range table csv')
parser.add_argument('--by_res','-r', action='store_true',
	help='perform analysis by residue')
parser.add_argument('--by_grp','-g', action='store_true',
	help='perform analyssi by functional group')

arg = parser.parse_args()

if not arg.by_res and not arg.by_grp:
	print('Need to choose one of the modes')
	sys.exit()

dataset = pd.read_json(arg.data,compression='xz').reset_index()

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
			if res == '':
				total += 1
				if cs > max:
					outliers += 1
					out_shifts.append(cs)
				elif cs < min:
					outliers += 1
					out_shifts.append(cs)
				else:
					good += 1
					
					
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
		
	return good, out_shifts

if arg.by_grp:
	results = dict()

with open(arg.ranges, mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	
	for row in csv_reader:
		aa  = row['aa']
		at  = row['atom_type']
		fg  = row['functional_group']
		min = row['min']
		max = row['max']
		
		if min == '' or max == '': continue
		
		df = dataset[['name','seq',at]].copy().reset_index()
		
		if arg.by_res:
			passed, outs = find_outliers(df, at, min, max, aa)
			print(f'{aa}, {at}, {len(outs)}, {good+len(out_shifts)}, {len(outs)/(good+len(outs))}')
		if arg.by_grp:
			passed, outs = find_outliers(df, at, min, max, '')
			if fg not in results:
				results[fg] = dict()
				results[fg]['passed'] = passed
				results[fg]['outs'] = outs
			else:
				results[fg]['passed'] += passed
				results[fg]['outs'] += outs

if arg.by_grp:
	for k in results.keys():
		o = len(results[k]['outs'])
		print(f"{k} {o} {o/(o+results[k]['passed']):.4f}")