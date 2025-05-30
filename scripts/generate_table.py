import json
import os
import glob
import numpy as np
import argparse

import prettytable
from prettytable import PrettyTable


def load_json(jsonfile):
    name = os.path.split(jsonfile)[-1].split('.')[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)
    return name, data


def read_scores_seqs(results, method, dataset, sequences, measure):
    read_seq_score = lambda x: round(results[method][dataset][x][measure], 8)
    return [read_seq_score(seq) for seq in sequences]


def generate_table(res_files, outfile, loss=None):
    results = [load_json(file) for file in res_files]
    results = {k: v for k, v in results}

    methods = list(results.keys())
    datasets = list(results[methods[0]].keys())

    fp = open(outfile, 'w')
    for dataset in datasets:
        info = f"Results on {dataset}"
        print(info)

        tabel = PrettyTable()
        tabel.header = False
        tabel.title = info
        tabel.hrules = prettytable.ALL
        tabel.max_table_width = 250
        sequences = list(results[methods[0]][dataset].keys())
        tabel.add_column('Sequence', sequences + ['Mean'])

        measures = list(results[methods[0]][dataset][sequences[0]].keys())
        if loss is not None:
            measures = [x for x in measures if x in loss]
        for measure in measures:
            for method in methods:
                data = read_scores_seqs(results, method, dataset, sequences, measure)
                data.append(np.mean(data))
                tabel.add_column(f"{measure}/{method}", data)
        print(tabel)
        fp.writelines(tabel.get_string())
        fp.writelines('\n')
    fp.close()

