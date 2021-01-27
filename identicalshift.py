#!/usr/bin/python3

import json
import sys
from statistics import mean, stdev

# need atom
assert(len(sys.argv) > 1)
atom = sys.argv[2]
	
# the data is in json
data = None
with open(sys.argv[1]) as fp:
	data = json.load(fp)

# group by sequence
gbs = {}
for record in data:
	seq = record['seq']
	if seq not in gbs: gbs[seq] = []
	gbs[seq].append(record)

# main
for seq, records in sorted(gbs.items(), key=lambda item: len(item[1]),
		reverse=True):

	# find the records with chemical shifts
	keep = []
	for record in records:
		if atom in record:
			values = False
			for shift in record[atom]:
				if shift != None:
					values = True
					break
			if values: keep.append(record)
	if len(keep) < 3: continue
	
	# report
	print(seq, len(keep), len(records))
	max = len(records[0]['seq'])
	print('id->\t', end='')
	for record in sorted(keep, key=lambda item: item['name']):
		print(record['name'], end='\t')
	print('mean\tstdev')
	
	for i in range(len(keep[0]['seq'])):
		print(keep[0]['seq'][i], end='\t')
		vals = []
		for record in sorted(keep, key=lambda item: item['name']):
			if atom in record:
				print(record[atom][i], end='\t')
				if record[atom][i] != None: vals.append(record[atom][i])
			else:
				print(None, end='\t')
		if len(vals) > 1:
			print(f'{mean(vals):.1f}\t{stdev(vals):.4f}')
		else:
			print('NA\tNA')
	
	print('-------------------------------')
