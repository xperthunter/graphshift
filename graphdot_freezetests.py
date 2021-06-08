#!/usr/bin/python3

import argparse
import json
import os
import pickle
import sys

import graphdot
from graphdot import Graph
from graphdot.linalg.cholesky import CholSolver
from graphdot.linalg.spectral import pinvh
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.fix import Normalization
from graphdot.kernel.marginalized.starting_probability import Uniform
from graphdot.microkernel import (
	TensorProduct,
	KroneckerDelta,
	SquareExponential,
	Constant
)
from graphdot.model.gaussian_process import GaussianProcessRegressor
from graphshift_lib import rmse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=''.join(('Test parameter optimization ',
	'in Graphdot GPR')))
parser.add_argument('--data', '-d', required=False, type=str,
	metavar='<str>', help='xz zipped json database')
parser.add_argument('--atom', '-a', required=False, default='C',
	metavar='<str>', help='atom type to predict on, default is carbon')
parser.add_argument('--trainum', '-n', required=False,
	metavar='<int>', help='training set size')
parser.add_argument('--batchsize', '-b', required=False,
	metavar='<int>', help='batch size')
parser.add_argument('--alpha', '-p', required=False, default=1e-8,
	metavar='<float>', help='alpha level')
parser.add_argument('--shuffle', '-u', required=False,
	action='store_true', help='shuffle or not shuffle dataset')
parser.add_argument('--optimize', '-m', required=False,
	action='store_true', help='optimize or build?')
parser.add_argument('--name', '-f', required=False, default='model.pickle',
	metavar='<str>', help='Name for pickle output file')
parser.add_argument('--save', '-o', required=False,
	metavar='<str>', help='directory for saved models')
parser.add_argument('--config', '-c', required=False,
	metavar='<str>', help='json configuration file')

arg = parser.parse_args()

if arg.config is None:
	print('must provide a config file')
	sys.exit()

with open(arg.config, 'r') as fp:
		config = json.load(fp)

data        = str(config['data'])
atom        = str(config['atom'])
n           = int(config['trainum'])
tol         = float(config['tol'])
alpha       = float(config['alpha'])
nv          = int(config['val'])
batch_size  = int(config['batchsize'])
out_path    = str(config['save'])
out_name    = str(config['name'])
optimize    = bool(config['optimize'])
shuffle     = bool(config['shuffle'])

h_elm          = float(config['h_elm'])
h_elm_range    = [float(b) for b in config['h_elm_range']]
assert(h_elm_range[0] <= h_elm_range[1])
assert(len(h_elm_range) == 2)
l_scale        = float(config['l_scale'])
l_scale_range  = [float(b) for b in config['l_scale_range']]
assert(l_scale_range[0] <= l_scale_range[1])
assert(len(l_scale_range) == 2)
starting_prob  = float(config['starting_prob'])
starting_range = [float(b) for b in config['starting_range']]
assert(starting_range[0] <= starting_range[1])
assert(len(starting_range) == 2)
stopping_prob  = float(config['stopping_prob'])
stopping_range = [float(b) for b in config['stopping_range']]
assert(stopping_range[0] <= stopping_range[1])
assert(len(stopping_range) == 2)

gdot_kernel = Normalization(
				MarginalizedGraphKernel(
					node_kernel=TensorProduct(
						element=KroneckerDelta(h=h_elm, h_bounds=h_elm_range),
					),
					edge_kernel=TensorProduct(
						length=SquareExponential(length_scale=l_scale, length_scale_bounds=l_scale_range)
					),
					p=Uniform(starting_prob, starting_range),
					q=stopping_prob,
					q_bounds=stopping_range
				)
			)

df = pd.read_pickle(data,compression='xz')

os.remove(arg.config)

if shuffle:
	df = df.sample(frac=1.0)

X = df.graphs[:n]
y   = np.concatenate(df.shifts[:n].to_list())
sym = np.concatenate(df.symbols[:n].to_list())
cmask = np.where(sym != atom)
y[cmask] = None
ymask    = np.where(y != None)

if nv == -1:
	Xv = df.graphs[n:]
	yv   = np.concatenate(df.shifts[n:].to_list())
	symv = np.concatenate(df.symbols[n:].to_list())
	cmaskv = np.where(symv != atom)
	yv[cmaskv] = None
	ymaskv     = np.where(yv != None)
else:
	Xv = df.graphs[n:n+nv]
	yv   = np.concatenate(df.shifts[n:n+nv].to_list())
	symv = np.concatenate(df.symbols[n:n+nv].to_list())
	cmaskv = np.where(symv != 'C')
	yv[cmaskv] = None
	ymaskv     = np.where(yv != None)	

train_nodes = [len(g.nodes['!i']) for g in X]
train_nodes = np.sum(np.array(train_nodes))
test_nodes  = [len(g.nodes['!i']) for g in Xv]
test_nodes  = np.sum(np.array(test_nodes))

print(f"""
train graphs: {n}
validation graphs: {nv}
train shifts: {y.shape[0]}
train nodes: {train_nodes}
validation shifts: {yv.shape[0]}
validation nodes: {test_nodes}
""")

gpr = GaussianProcessRegressor(
    kernel=gdor_kernel,
    alpha=alpha,
    optimizer=optimize,
    normalize_y=True,
    kernel_options=dict(nodal=True),
    regularization='*'
)
gpr.fit(X,y,verbose=True,tol=tol,repeat=5)

lml = gpr.log_marginal_likelihood()

inpreds = gpr.predict(X)
outpreds = gpr.predict(Xv)

inMAE = np.mean(np.abs(inpreds[ymask] - y[ymask]))
inRMSE = rmse(inpreds[ymask], y[ymask])

outMAE = np.mean(np.abs(outpreds[ymaskv] - yv[ymaskv]))
outRMSE = rmse(outpreds[ymaskv], yv[ymaskv])

print(gpr.kernel.hyperparameters)
print(f'Log marginal likelihood: {lml:1.4E}')
print(f'Train RMSE: {inRMSE:1.4E}')
print(f'Test RMSE : {outRMSE:1.4E}')
	
print(f'Train MAE: {inMAE:1.4E}')
print(f'Test MAE: {outMAE:1.4E}')