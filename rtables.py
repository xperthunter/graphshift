#!/usr/bin/python3

import sys
import csv

print('aa,atom_type,functional_group,min,max')
with open(sys.argv[1], mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	
	for row in csv_reader:
#		print(row)
#		sys.exit()
		aa  = row['Amino acid']
		at  = row['atom type']
		fg  = row['functional group']
		min = row['min']
		max = row['max']
		print(f'{aa},{at},{fg},{min},{max}')
#		sys.exit()