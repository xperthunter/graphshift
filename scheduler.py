#!/usr/bin/python3

import argparse
import json
import subprocess
import sys

sample = {
	"name"       : None,
	"errfile"    : None,
	"outfile"    : None,
	"alpha"      : None,
	"tol"        : None,
	"trainum"    : None,
	"val"        : None,
	"batchsize"  : None,
	"data"       : None,
	"kernel"     : None,
	"mem"        : None,
	"gpu_mem"    : None,
	"slurm_outs" : None
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
arg = parser.parse_args()

with open(arg.config) as config:
	configs = json.load(config)
	assert(type(configs) is list)

cmd_1 = f"bash {arg.slurm}"

for c in configs:
	assert(type(c) is dict)
	assert(keycheck(c, sample))
	
	cmd = ''.join((cmd_1,
				   f" -e {c['errfile']}",
				   f" -o {c['outfile']}",
				   f" -m {c['mem']}",
				   f" -g {c['gpu_mem']}",
				   f" -l {c['slurm_outs']}"))
	print(cmd)
	cmd_list = cmd.split()
	print(cmd_list)
	retval = subprocess.check_output(cmd_list)
	result = retval.decode('utf-8')
	print(result,end="")

"""
arg        = parser.parse_args()
n          = int(arg.train)
tol        = float(arg.tol)
chunk_size = int(arg.batch)
out_path   = str(arg.out)

df = pd.read_pickle(arg.data,compression='xz')

print(df.shape)

for i in range(0, df.shape[0], n):
	if i+n < df.shape[0]: print(i,i+n)
	else:                 print(i, df.shape[0])
	
df[:i], df[i+n:]

n = 1000
tot 19347
19 diff models
n = 1800
be 18
whatever is left over, make a new file for remainder_n.pickle.xz

bgpr_n_1.model.pickle
bgpr_1000_1.model.pickle
bgpr_1000_2.model.pickle

parser.add_argument('--data', '-d', required=True, type=str,
	metavar='<str>', help='xz zipped pickled pandas dataframe')
parser.add_argument('--train', '-n', required=True,
	metavar='<int>', help='training set size')
parser.add_argument('--batch', '-b', required=True,
	metavar='<int>', help='batch size')
parser.add_argument('--tol', '-t', required=False, default=1e-3,
	metavar='<float>', help='tolerance for optimization')
parser.add_argument('--out', '-o', required=True,
	metavar='<str>', help='directory for saved models')


"""