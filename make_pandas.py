#!/usr/bin/python3

import json
import pickle
import sys

import graphshift_lib as gl
import pandas as pd

df_training   = gl.make_df(sys.argv[1])
df_validation = gl.make_df(sys.argv[2])

df_training.to_pickle('nmrshift_training.pickle')
df_validation.to_pickle('nmrshift_validation.pickle') 