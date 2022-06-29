# -*- coding: utf-8 -*-
import os
import pickle
import json
from sklearn.utils import shuffle
import random
import sys
import numpy as np



# main

if len(sys.argv) != 2:
    raise ValueError('Args: <test set path prefix>\n  Data input expected as <prefix>.jsonl and <prefix>-truth.jsonl.')

path_prefix = sys.argv[1]


dir_pairs = path_prefix+".jsonl"
dir_labels = path_prefix+"-truth.jsonl"
# open json files
with open(dir_pairs, 'r') as f:
    lines_pairs = f.readlines()
with open(dir_labels, 'r') as f:
    lines_labels = f.readlines()

docs_L = []
docs_R = []
labels_a = []
labels_c = []
for n in range(len(lines_pairs)):

    pair, label = json.loads(lines_pairs[n].strip()), json.loads(lines_labels[n].strip())

    docs_L.append(pair['pair'][0])
    docs_R.append(pair['pair'][1])
    if label['authors'][0] == label['authors'][1]:
        labels_a.append(1)
    else:
        labels_a.append(0)
    labels_c.append(0)


#####################
# load validation set
#####################
dir_results = os.path.join('..', 'data_preprocessed')
file_results = os.path.join(dir_results, 'results.txt')

open(file_results, 'a').write('-----------------------------------------------\n')


###################
# check re-sampling
###################
# counts
dict_counts = {"SA_SF": 0,
               "SA_DF": 0,
               "DA_SF": 0,
               "DA_DF": 0,
               }
for i in range(len(docs_L)):

    if labels_a[i] == 1 and labels_c[i] == 1:
        dict_counts["SA_SF"] += 1
    if labels_a[i] == 1 and labels_c[i] == 0:
        dict_counts["SA_DF"] += 1
    if labels_a[i] == 0 and labels_c[i] == 1:
        dict_counts["DA_SF"] += 1
    if labels_a[i] == 0 and labels_c[i] == 0:
        dict_counts["DA_DF"] += 1

open(file_results, 'a').write('val: '
                              + ', #pairs: ' + str(len(labels_a))
                              + ', a=0: ' + str(np.sum(np.array(labels_a) == 0))
                              + ', a=1: ' + str(np.sum(np.array(labels_a) == 1))
                              + ', c=0: ' + str(np.sum(np.array(labels_c) == 0))
                              + ', c=1: ' + str(np.sum(np.array(labels_c) == 1))
                              + '\n')
open(file_results, 'a').write('SA_SF: ' + str(dict_counts["SA_SF"])
                              + ', SA_DF: ' + str(dict_counts["SA_DF"])
                              + ', DA_SF: ' + str(dict_counts["DA_SF"])
                              + ', DA_DF: ' + str(dict_counts["DA_DF"])
                              + '\n')

#######
# store
#######
with open(os.path.join(dir_results, 'pairs_val'), 'wb') as f:
    pickle.dump((docs_L, docs_R, labels_a, labels_c), f)
