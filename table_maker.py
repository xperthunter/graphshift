#!/usr/bin/python3 

import argparse
import json
import sys

from pytablewriter import MarkdownTableWriter

headers = {
#	'name'        : 'name',
	'n'           : 'trainum',
	'nv'          : 'val',
	'data'        : 'data',
	'train_cs'    : 'train_cs',
	'test_cs'     : 'test_cs',
	'in-mae'      : 'inMAE',
	'out-mae'     : 'outMAE',
	'in-rmse'     : 'inRMSE',
	'out-rmse'    : 'outRMSE',
	'logml'       : 'logml',
	'alpha'       : 'alpha'
# 	'train_nodes' : 'train_nodes',
# 	'test_nodes'  : 'test_nodes',
}


parser = argparse.ArgumentParser(description='make markdown tables for training results')
parser.add_argument('--results', '-r', required=True, type=str,
	metavar='<str>', help='json of completed jobs results')

arg = parser.parse_args()

with open(arg.results, 'r') as fp:
	results = json.load(fp)
fp.close()

mres = []
for r in sorted(results, key = lambda i: float(i['outMAE'])):
	sub_res = []
	for v in headers.values():
		sub_res.append(r[v])
	mres.append(sub_res)

writer = MarkdownTableWriter(
	table_name="GPR Training results",
	headers=list(headers.keys()),
	value_matrix=mres,
	margin=1
)

writer.write_table()
