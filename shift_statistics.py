#!/usr/bin/python3

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import datalib as dl

pd.set_option('display.max_rows', 75)

df = pd.read_json(sys.argv[1],compression='xz').reset_index()
print(df.shape)

percentile = 1.0
Nshifts  = dl.clean_atom_set(df, 'N',  percentile)
Hshifts  = dl.clean_atom_set(df, 'H',  percentile)
CAshifts = dl.clean_atom_set(df, 'CA', percentile)
HAshifts = dl.clean_atom_set(df, 'HA', percentile)
Cshifts  = dl.clean_atom_set(df, 'C',  percentile)
CBshifts = dl.clean_atom_set(df, 'CB', percentile)

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(15,8))
fig.suptitle(f'Backbone atom chemical shift histograms ({percentile} percentile)')
axs[0,0].hist(Nshifts)
axs[0,0].set_title('N')
axs[0,1].hist(Hshifts)
axs[0,1].set_title('H')
axs[0,2].hist(CAshifts)
axs[0,2].set_title('CA')
axs[1,0].hist(HAshifts)
axs[1,0].set_title('HA')
axs[1,1].hist(Cshifts)
axs[1,1].set_title('C')
axs[1,2].hist(CBshifts)
axs[1,2].set_title('CB')

plt.subplots_adjust(wspace=0.2, hspace=0.2)

plt.show()

quantile = 1.0

stats = list()

for cn in df.columns.values:
    if cn == 'index' or cn == 'name' or cn == 'seq' or cn == 'H2': continue

    stats.append(dl.stats_report(dl.clean_atom_set(df, cn, quantile),cn))

stats_report_df = pd.DataFrame(stats)
print(stats_report_df.head(69))
