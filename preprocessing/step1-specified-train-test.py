# -*- coding: utf-8 -*-
import os
import pickle
import json
from sklearn.utils import shuffle
import random
import sys

#################
# parse json file
#################
def parse(path):
    g = open(path, 'r')
    for l in g:
        yield json.loads(l.strip())


##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self):

        # dictionary with all documents
        self.dict_author_fandom_doc = {}

        # sets with unique authors/fandoms/docs
        self.authors = set()
        self.fandoms = set()
        self.unique_docs = set()

        # author/fandom sets for training set (train DML, BFS, UAL)
        self.authors_train = set()
        self.fandoms_train = set()
        # author/fandom sets for calibration set (train O2D2)
        self.authors_cal = set()
        self.fandoms_cal = set()
        # author/fandom sets for validation set (check results)
        self.authors_val = set()
        self.fandoms_val = set()

        # dictionaries for the splits
        self.dict_author_fandom_doc_train = {}
        self.dict_author_fandom_doc_cal = {}
        self.dict_author_fandom_doc_val = {}

        # counts
        self.n_train = 0
        self.n_cal = 0
        self.n_val = 0
        self.n_dropped = 0

    ##################
    # parse train docs
    ##################
    def parse_raw_data(self, path_prefix):

        self.dict_author_fandom_doc = {}

        # sets with unique authors/fandoms/docs
        self.authors = set()
        self.fandoms = set()
        self.unique_docs = set()

        dir_pairs = path_prefix+".jsonl"
        dir_labels = path_prefix+"-truth.jsonl"
        # open json files
        with open(dir_pairs, 'r') as f:
            lines_pairs = f.readlines()
        with open(dir_labels, 'r') as f:
            lines_labels = f.readlines()

        for n in range(len(lines_pairs)):

            pair, label = json.loads(lines_pairs[n].strip()), json.loads(lines_labels[n].strip())

            for i in range(2):

                # get author-ID, fandom, fanfiction
                author = label['authors'][i]
                fandom = pair['fandoms'][i]
                doc = pair['pair'][i]

                # remove "broken" or very short documents
                if doc not in self.unique_docs:

                    self.unique_docs.add(doc)
                    self.authors.add(author)
                    self.fandoms.add(fandom)

                    if author not in self.dict_author_fandom_doc.keys():
                        self.dict_author_fandom_doc[author] = {}
                    if fandom not in self.dict_author_fandom_doc[author].keys():
                        self.dict_author_fandom_doc[author][fandom] = []
                    self.dict_author_fandom_doc[author][fandom].append(doc)


    def assign_data_to(self, dataset_id):

        if dataset_id == 'train':
            self.fandoms_train = self.fandoms
            self.authors_train = self.authors
            self.dict_author_fandom_doc_train = self.dict_author_fandom_doc
            self.n_train = len(self.dict_author_fandom_doc_train)
        elif dataset_id == 'val':
            self.fandoms_val = self.fandoms
            self.authors_val = self.authors
            self.dict_author_fandom_doc_val= self.dict_author_fandom_doc
            self.n_val = len(self.dict_author_fandom_doc_val)
        elif dataset_id == 'cal':
            self.fandoms_cal = self.fandoms           
            self.authors_cal = self.authors
            self.dict_author_fandom_doc_cal = self.dict_author_fandom_doc
            self.n_cal = len(self.dict_author_fandom_doc_cal)
        else:
            raise Error("Invalid dataset_id value, must be 'train', 'val', or 'cal'")


# main

if len(sys.argv) != 3:
    raise ValueError('Args: <train set path prefix> <test set path prefix>\n  Data input expected as <prefix>.jsonl and <prefix>-truth.jsonl.')

train_prefix = sys.argv[1]
test_prefix = sys.argv[2]


#####################################
# create folder for preprocessed data
#####################################
dir_results = os.path.join('..', 'data_preprocessed')
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

##########
# log file
##########
file_results = os.path.join(dir_results, 'results.txt')
if os.path.isfile(file_results):
    os.remove(file_results)

##########################
# create object for Corpus
##########################
corpus = Corpus()

open(file_results, 'a').write('parse training data...' + '\n')
corpus.parse_raw_data(train_prefix)
corpus.assign_data_to('train')
open(file_results, 'a').write('parse test data...' + '\n')
corpus.parse_raw_data(test_prefix)
corpus.assign_data_to('val')

##############################
# store results (binary files)
##############################
with open(os.path.join(dir_results, 'dict_author_fandom_doc_train'), 'wb') as f:
    pickle.dump(corpus.dict_author_fandom_doc_train, f)
with open(os.path.join(dir_results, 'dict_author_fandom_doc_cal'), 'wb') as f:
    pickle.dump(corpus.dict_author_fandom_doc_cal, f)
with open(os.path.join(dir_results, 'dict_author_fandom_doc_val'), 'wb') as f:
    pickle.dump(corpus.dict_author_fandom_doc_val, f)

############
# statistics
############

#open(file_results, 'a').write('# unique docs: ' + str(len(corpus.unique_docs)) + '\n')
#open(file_results, 'a').write('# unique authors: ' + str(len(corpus.authors)) + '\n')
#open(file_results, 'a').write('# unique fandoms: ' + str(len(corpus.fandoms)) + '\n')

#open(file_results, 'a').write('# docs (train): ' + str(corpus.n_train) + '\n')
#open(file_results, 'a').write('# docs (cal): ' + str(corpus.n_cal) + '\n')
#open(file_results, 'a').write('# docs (val): ' + str(corpus.n_val) + '\n')
#open(file_results, 'a').write('# docs (dropped): ' + str(corpus.n_dropped) + '\n')

open(file_results, 'a').write('# authors (train): ' + str(len(corpus.authors_train)) + '\n')
open(file_results, 'a').write('# authors (cal): ' + str(len(corpus.authors_cal)) + '\n')
open(file_results, 'a').write('# authors (val): ' + str(len(corpus.authors_val)) + '\n')

open(file_results, 'a').write('# fandoms (train): ' + str(len(corpus.fandoms_train)) + '\n')
open(file_results, 'a').write('# fandoms (cal): ' + str(len(corpus.fandoms_cal)) + '\n')
open(file_results, 'a').write('# fandoms (val): ' + str(len(corpus.fandoms_val)) + '\n')

open(file_results, 'a').write('intersection authors (train + cal): ' + str(len(corpus.authors_train.intersection(corpus.authors_cal))) + '\n')
open(file_results, 'a').write('intersection authors (train + val): ' + str(len(corpus.authors_train.intersection(corpus.authors_val))) + '\n')
open(file_results, 'a').write('intersection authors (cal + val): ' + str(len(corpus.authors_cal.intersection(corpus.authors_val))) + '\n')

open(file_results, 'a').write('intersection fandoms (train + cal): ' + str(len(corpus.fandoms_train.intersection(corpus.fandoms_cal))) + '\n')
open(file_results, 'a').write('intersection fandoms (train + val): ' + str(len(corpus.fandoms_train.intersection(corpus.fandoms_val))) + '\n')
open(file_results, 'a').write('intersection fandoms (cal + val): ' + str(len(corpus.fandoms_cal.intersection(corpus.fandoms_val))) + '\n')

open(file_results, 'a').write('finished!' + '\n')
