#!/usr/bin/python3

import argparse
import json
import time
import subprocess
import sys

parser = argparse.ArgumentParser(description='Track allocation for jobs')
parser.add_argument('--submitted', '-u', required=True, type=str,
	metavar='<str>', help='json of submitted jobs')
# parser.add_argument('--results', '-r', required=True, type=str,
# 	)

arg = parser.parse_args()

with open(arg.submitted, 'r') as fp:
	submitted = json.load(fp)

for c in submitted:
	jobid = c['jobid']
	
	cmd = f"squeue -j {jobid}"
	cmd_list = cmd.split()
	retval = subprocess.check_output(cmd_list)
	result = retval.decode('utf-8')
	print(result,end="")
	
	if 'Invalid job id specified' in result:
		print('completed?')
	
	