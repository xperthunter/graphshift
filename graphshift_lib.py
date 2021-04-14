#!/usr/bin/python3

from ase import Atoms
import graphdot
from graphdot import Graph
from math import sqrt
import numpy as np
import pandas as pd

def make_df(file):
	df = pd.read_json(file,compression='xz').reset_index()
	df['atoms']  = df.apply(lambda x: Atoms(symbols=x['symbols'], positions=x['xyz']),axis=1)
	df['graph']  = df.atoms.apply(Graph.from_ase)
	df['graphs'] = Graph.unify_datatype(df.graph.to_list())
	
	return df

def make_maskshifts(shifts, symbols, focus_atom):
    lens = [len(l) for l in shifts]
    maxlen=max(lens)
    total_mask = np.zeros((len(shifts),maxlen),np.uint0)

    cs = np.zeros((len(shifts),maxlen),np.float64)
    sym = np.zeros((len(symbols),maxlen),dtype=object)
    mask = np.arange(maxlen) < np.array(lens)[:,None]
    cs[mask] = np.concatenate(shifts)
    sym[mask] = np.concatenate(symbols)

    nanmask = np.isnan(cs)
    total_mask[nanmask] = 1

    zeromask = np.where(cs == 0)
    total_mask[zeromask] = 1
    
    elmask = np.where(sym != focus_atom)
    total_mask[elmask] = 1

    return np.ma.masked_array(cs,mask=total_mask)

def rmse(predictions, gt):
	assert(predictions.shape[0] == gt.shape[0])
	rmse = predictions - gt
	rmse = np.power(rmse, 2)
	rmse = rmse / gt.shape[0]
	rmse = np.sum(rmse)
	
	return sqrt(rmse)
