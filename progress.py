#!/usr/bin/python3

import argparse
import json
import os
import re
import time
import subprocess
import sys

results_table = {
	"data"        : None,
	"jobid"       : None,
	"name"        : None,
	"atom"        : None,
	"out"         : None,
	"err"         : None,
	"trainum"     : None,
	"val"         : None,
	"train_cs"    : None,
	"test_cs"     : None,
	"train_nodes" : None,
	"test_nodes"  : None, 
	"alpha"       : None,
	"logml"       : None,
	"inMAE"       : None,
	"outMAE"      : None,
	"inRMSE"      : None,
	"outRMSE"     : None,
}

parser = argparse.ArgumentParser(description='Track allocation for jobs')
parser.add_argument('--submitted', '-u', required=True, type=str,
	metavar='<str>', help='json of submitted jobs')
parser.add_argument('--results', '-r', required=True, type=str,
	metavar='<str>', help='json of completed jobs')

def rlist_server(config, rtable):
	rlist = rtable.copy()
	for k, v in config.items():
		if k in rtable:
			rlist[k] = v
	
	return rlist

arg = parser.parse_args()

with open(arg.submitted, 'r') as fp:
	submitted = json.load(fp)
fp.close()

# with open(arg.results, 'r') as fp:
# 	total_results = json.load(fp)
# fp.close()

total_results = []
for i, c in enumerate(submitted):
	#if c['status'] == 'completed': continue
	jobid = c['jobid']
	print(jobid)
	cmd = f"squeue -j {jobid}"
	cmd_list = cmd.split()
	try:
		retval = subprocess.check_output(cmd_list)
		result = retval.decode('utf-8')
	except subprocess.CalledProcessError as e:
		print(f'{jobid} is invalid')
		print(c.keys())
		filepath = os.path.join(os.getcwd(), c['save'], c['name'])
		print(filepath)
		print(os.path.isfile(filepath))
		if not os.path.isfile(filepath):
			print('model is not made')
			tmpfile = os.path.join(os.getcwd(), c['config_file'])
			if os.path.isfile(tmpfile):
				cmd = f'rm -f {tmpfile}'
				os.system(cmd)
			submitted.pop(i)
			continue
		else:
			#assert(c['status'] == 'running')
			print('gather results')
			outfile = os.path.join(os.getcwd(), c['slurm_outs'], c['outfile'])
			errfile = os.path.join(os.getcwd(), c['slurm_outs'], c['errfile'])
			assert(os.path.isfile(outfile) and os.path.isfile(errfile))
			
			with open(outfile, 'r') as fr:
				gather = False
				for line in fr:
					if '### RESULTS ###' in line:
						gather = True
						reslist = rlist_server(c, results_table)
						continue
					if gather:
						if re.search(r'^_', line):
							vals = line.split()
							assert(len(vals) == 3)
							rkey = vals[0]
							rkey = rkey[1:]
							print(rkey, vals[2])
							
							if rkey not in reslist:
								print('un expected value in output')
							reslist[rkey] = vals[2]
			fr.close()
			c['status'] = 'completed'
			if 'datacolumn' not in c: reslist['data'] = 'unify_ase_rdkit_graph'
			else: reslist['data'] = c['datacolumn']
			total_results.append(reslist)
			reslist = {}
			
			continue
	
	if 'JOBID' in result:
		if 'ST ' in result:
			if 'PD ' in result:
				print(f"jobid {jobid} is allocating, {c['name']}")
				c['status'] = 'allocating'
				tmpfile = os.path.join(os.getcwd(), c['config_file'])
				print(tmpfile)
				assert(os.path.isfile(tmpfile))
			elif 'R ' in result:
				print(f"jobid {jobid} is running, {c['name']}")
				c['status'] = 'running'
			else:
				print('re-run?')

with open(arg.submitted, 'w') as fp:
	json.dump(submitted, fp)
fp.close()

with open(arg.results, 'w') as fp:
	json.dump(total_results, fp)
fp.close()
