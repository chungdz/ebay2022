import pandas as pd
from tqdm import trange
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--target", default='parsed_train.tsv', type=str)
parser.add_argument("--filename", default='subtrain', type=str)
args = parser.parse_args()

data = pd.read_csv('data/{}'.format(args.target), sep='\t')
folds = 10
split_num = int(data.shape[0] / 10)

split_points = []
for i in range(folds):
    split_points.append(split_num * i)
split_points.append(data.shape[0])

for i in trange(1, folds + 1):
    
    start_index = split_points[i - 1]
    end_index = split_points[i]
    
    pre_subset = data[0: start_index]
    nxt_subset = data[end_index: data.shape[0]]
    
    cur_train = pd.concat([pre_subset, nxt_subset], axis=0)
    cur_valid = data[start_index: end_index]
    
    cur_train.to_csv('data/{}/train_{}.tsv'.format(args.filename, i), index=None, sep='\t')
    cur_valid.to_csv('data/{}/valid_{}.tsv'.format(args.filename, i), index=None, sep='\t')
    print(start_index, end_index, cur_train.shape, cur_valid.shape)

    del cur_train, cur_valid
    gc.collect()