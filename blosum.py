#!/usr/bin/python3

import sys
import json

def make_blosum(file):

	blosum = dict()
	with open(file) as fp:
		line = fp.readline()
		res = line.split()
		res = res[:20]
		
		for row_res in res:
			line = fp.readline()
			line = line.split()
			
			blosum[row_res] = dict()
			for i in range(1,21):
				column_res = res[i-1]
				blosum[row_res][column_res] = int(line[i])
	
	return blosum

def align_score(seq1, seq2, matrix):
	
	score = 0
	for i, j in zip(seq1, seq2):
		score += matrix[i][j]
	
	return score