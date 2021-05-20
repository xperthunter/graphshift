#!/usr/bin/python3

import json
import os
import pandas as pd
import sys

from ase import Atoms
import graphdot
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.fix import Normalization
from graphdot.kernel.marginalized.starting_probability import Uniform
from graphdot.microkernel import (
	TensorProduct,
	SquareExponential,
	KroneckerDelta,
	Constant
)
from graphdot.model.gaussian_process import (
	GaussianProcessRegressor,
	LowRankApproximateGPR
)
import networkx as nx
import numpy as np

print(f"Graphdot version: {graphdot.__version__}")

def make_kernel():
	return Normalization(
		MarginalizedGraphKernel(
			node_kernel=TensorProduct(
				element=KroneckerDelta(h=0.5, h_bounds=(0.1, 0.9)),
			),
			edge_kernel=TensorProduct(
				length=SquareExponential(
					length_scale=0.1,
					length_scale_bounds=(0.1, 0.5)
				)
			),
			p=Uniform(10.0, (1.0, 50.0)),
			q=0.01,
			q_bounds=(0.001, 0.5)
		)
	)

df = pd.read_json(sys.argv[1],compression='xz').sample(frac=0.10).reset_index()
print(df.shape)
print(df.columns)

df['atoms']  = df.apply(lambda x: Atoms(symbols=x['symbols'], positions=x['xyz']),axis=1)
df['graph']  = df.atoms.apply(Graph.from_ase)
df['graphs'] = Graph.unify_datatype(df.graph.to_list())

gpr = GaussianProcessRegressor(
	kernel=make_kernel(),
	alpha=1e-4,
	optimizer=True,
	normalize_y=True,
	kernel_options=dict(nodal=True)
)

n = 5
X = df.graphs[:n]
y = np.concatenate(df.shifts[:n].to_list())
y_mask = np.where(y != None)
print(len(X), len(y), y_mask[0].shape)

gpr.fit(X, y, tol=1e-1, verbose=True)

pred = gpr.predict(X)
RMSE = np.std(y[y_mask] - pred[y_mask])
print(RMSE)