import numpy as np
import json
import pandas as pd
from tqdm import trange
import argparse
from modules.fnn import FNN

folds = 10

for i in trange(1, folds + 1):
    print('model:', i)
    x_train = pd.read_csv('data/subtrain_cat/train_{}.tsv'.format(i), sep='\t')
    x_valid = pd.read_csv('data/subtrain_cat/valid_{}.tsv'.format(i), sep='\t')

    y_train = x_train.target
    x_train = x_train.drop(['record_number', 'target'], axis=1)
    y_valid = x_valid.target
    x_valid = x_valid.drop(['record_number', 'target'], axis=1)