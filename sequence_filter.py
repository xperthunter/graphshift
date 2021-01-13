#!/usr/bin/python3

import argparse
import gzip
import sys

def read_blastreport(report, cutoff):
	blast = dict()
	with gzip.open(report, 'rt') as fp:
		for line in fp.readlines():
			f = line.split()
			qid = f[0]
			sid = f[1]
			pct = float(f[10])
			
			if pct < cutoff: continue
			
			if qid not in blast:
				blast[qid] = list()
				
			blast[qid].append(sid)
	
	return blast

def identity_filter(report, cutoff):
	ignore = dict()
	keep   = list()
	
	blast = read_blastreport(report, cutoff)
	
	for k,v in sorted(blast.items(),key=lambda item: len(item[1]),reverse=True):
		if k not in ignore: keep.append(k)
		for sid in v: ignore[sid] = True
		
	return keep

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=''.join(('Filter sequences to an',
		'user defined sequence identity level')))
	parser.add_argument('--table','-t', required=True, type=str,
		metavar='<str>', help='blast report, tabular format')
	parser.add_argument('--cutoff', '-c', required=True, type=int, 
		help='Percent identity cutoff')
	
	arg = parser.parse_args()
	cutoff = arg.cutoff
	
	keepers = identity_filter(arg.table, cutoff)

	for k in keepers:
		print(k)