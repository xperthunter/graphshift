#!/usr/bin/python3

import argparse
import json
import time
import subprocess
import sys

sample = {
	"name"          : None,
	"errfile"       : None,
	"outfile"       : None,
	"alpha"         : None,
	"tol"           : None,
	"trainum"       : None,
	"val"           : None,
	"batchsize"     : None,
	"data"          : None,
	"kernel"        : None,
	"mem"           : None,
	"gpu_mem"       : None,
	"slurm_outs"    : None,
	"save"          : None,
	"shuffle"       : None,
	"h_elm"         : None,
	"h_elm_range"   : None,
	"l_scale"       : None,
	"l_scale_range" : None,
	"starting_prob" : None,
	"starting_range": None,
	"stopping_prob" : None,
	"stopping_range": None,
	"optimize"      : None,
	"atom"          : None
}

skeys = sample.keys()
def keycheck(dic, sample):
	for k in dic.keys():
		if k not in sample:
			print("Non standard key in config")
			print(json.dumps(dic, indent=2))
			sys.exit()
	for k in sample.keys():
		if k not in dic:
			print(f"Missing key {k} in config")
			print("sample\n", json.dumps(sample, indent=2))
			print()
			print("config\n", json.dumps(dic, indent=2))
			sys.exit()
	return True

parser = argparse.ArgumentParser(description=''.join(('Schedule multiple ',
	'jobs for GPR chemical shift predictionn model')))
parser.add_argument('--config', '-c', required=True, type=str,
	metavar='<str>', help='path to json with configurations for models to be trained')
parser.add_argument('--slurm', '-s', required=True, type=str,
	metavar='<str>', help='path to slurm scheduler')
parser.add_argument('--submitted', '-u', required=True, type=str,
	metavar='<str>', help='json of submitted jobs')

arg = parser.parse_args()

if arg.submitted:
	with open(arg.submitted, 'r') as fp:
		submitted = json.load(fp)
	
	names = []
	for c in submitted:
		names.append(c['name'])
	
with open(arg.config) as config:
	configs = json.load(config)
	assert(type(configs) is list)

cmd_1 = f"bash {arg.slurm}"

for c in configs:
	assert(type(c) is dict)
	assert(keycheck(c, sample))
	if c['name'] in names:
		print(json.dumps(c, indent=2))
		print(f"{c['name']} already used, skipping")
		continue
	
	time.sleep(1)
	
	tmp_dic = c.copy() 
	tmp_dic.pop('errfile')
	tmp_dic.pop('outfile')
	tmp_dic.pop('mem')
	tmp_dic.pop('gpu_mem')
	tmp_dic.pop('slurm_outs')
	
	timestr = time.strftime("%Y%m%d_%H%M%S")
	tmp_name = f"tmp.config.{c['name']}.{timestr}.json"
	print(tmp_name)
	with open(tmp_name, 'w') as fp:
		json.dump(tmp_dic, fp)
	
	cmd = ''.join((cmd_1,
				   f" -e {c['errfile']}",
				   f" -o {c['outfile']}",
				   f" -m {c['mem']}",
				   f" -g {c['gpu_mem']}",
				   f" -l {c['slurm_outs']}",
				   f" -j {tmp_name}"))
	
	print(cmd)
	cmd_list = cmd.split()
	retval = subprocess.check_output(cmd_list)
	result = retval.decode('utf-8')
	print(result,end="")
	resplit = result.split()
	if resplit[0] == 'Submitted': 
		jobid = int(resplit[-1])
	else:
		print('job not submitted')
		sys.exit()
	
	tmp_dic['jobid'] = jobid
	submitted.append(tmp_dic)

with open(arg.submitted, 'w') as fp:
	json.dump(submitted, fp)