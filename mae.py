#!/usr/bin/python3

import argparse
import os
import pickle
import sys

from batchedGRP import BatchedGaussianProcessRegressor
from graphshift_lib import make_df, make_maskshifts, rmse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=''.join(('Load training models ',
	'and evaluate rmse and mae')))
parser.add_argument('--data', '-d', required=True, type=str,
	metavar='<str>', help='xz zipped pickle file')
parser.add_argument('--size', '-n', required=True, type=str,
	metavar='<str>', help='number of graphs model was trained with')
parser.add_argument('--path', '-p', required=True,
	metavar='<str>', help='path to model')
parser.add_argument('--model', '-m', required=True,
	metavar='<str>', help='model name')

arg = parser.parse_args()

bgpr = BatchedGaussianProcessRegressor(chunk_size=1400)
bgpr.load(arg.path,filename=arg.model)

df = pd.read_pickle(arg.data,compression='xz')

n  = int(arg.size)
assert(n == len(bgpr.xg))
atom = 'C'
X  = df.graphs[:n]
Y  = make_maskshifts(df.shifts[:n].to_list(), df.symbols[:n].to_list(), atom)

Xv = df.graphs[n:]
Yv = make_maskshifts(df.shifts[n:].to_list(), df.symbols[n:].to_list(), atom)

train_preds = bgpr.predict(X, Y)
test_preds  = bgpr.predict(Xv, Yv)

inRMSE  = rmse(train_preds, Y.compressed())
outRMSE = rmse(test_preds, Yv.compressed())

inMAE  = np.mean(np.abs(train_preds - Y.compressed()))
outMAE = np.mean(np.abs(test_preds - Yv.compressed()))

print(f'Train RMSE: {inRMSE:1.4E}')
print(f'Test RMSE : {outRMSE:1.4E}')

print(f'Train MAE: {inMAE:1.4E}')
print(f'Test MAE: {outMAE:1.4E}')

predictions = {'train':dict(), 'test':dict()}
predictions['train']['pred'] = train_preds
predictions['train']['gt']   = Y.compressed()

predictions['test']['pred'] = test_preds
predictions['test']['gt']   = Yv.compressed()

pred_file = os.path.join(arg.path, 'predictions_'+arg.model)
pickle.dump(predictions, open(pred_file, 'wb'), protocol=3)
print(bgpr.theta)