import numpy as np
from catboost import Pool, CatBoostRegressor
import json
import pandas as pd
from tqdm import trange
import argparse

class EbayMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            cur_approx = round(approx[i])
            residual = int(target[i] - cur_approx)
            if residual < 0:
                error_sum += w * -0.6 * residual
            else:
                error_sum += w * 0.4 * residual

        return error_sum, weight_sum

folds = 10
num_rounds = 50
esr = 3

parser = argparse.ArgumentParser()
parser.add_argument("--starti", default=1, type=int)
parser.add_argument("--depth", default=6, type=int)
args = parser.parse_args()

loss_and_output = []
all_log = []
for i in trange(args.starti, folds + 1):
    print('model:', i)
    train_set = pd.read_csv('data/subtrain/train_{}.tsv'.format(i), sep='\t')
    train_set['cross_city'] = train_set['cross_city'].astype('int')
    train_set['cross_state'] = train_set['cross_state'].astype('int')
    valid_set = pd.read_csv('data/subtrain/valid_{}.tsv'.format(i), sep='\t')
    valid_set['cross_city'] = valid_set['cross_city'].astype('int')
    valid_set['cross_state'] = valid_set['cross_state'].astype('int')

    x_train = train_set.drop(['record_number', 'target'],axis=1)
    y_train = train_set.target
    x_valid = valid_set.drop(['record_number', 'target'],axis=1)
    y_valid = valid_set.target

    train_pool = Pool(x_train, 
                  y_train, 
                  cat_features=[0, 4, 7, 8, 12, 13],
                  feature_names=list(x_train.columns))
    test_pool = Pool(x_valid,
                 y_valid,
                 cat_features=[0, 4, 7, 8, 12, 13],
                 feature_names=list(x_valid.columns))

    model = CatBoostRegressor(iterations=num_rounds, 
                          depth=args.depth, 
                          learning_rate=1, 
                          loss_function='RMSE',
                          eval_metric=EbayMetric())
    
    model.fit(train_pool, early_stopping_rounds=3, eval_set=test_pool, use_best_model=True, log_cout=open('result/output.txt', 'w'))
    model.save_model('para/catboost_{}.cbm'.format(i))
    logstr = open('result/output.txt', 'r').readlines()
    all_log.append(logstr)
    print(logstr)
    llen = len(logstr)
    for i in range(llen - 1, -1, -1):
        curl = logstr[i]
        if 'bestTest' in curl:
            loss_and_output.append(float(curl.split()[-1]))
            print(float(curl.split()[-1]))
            break

lao = np.array([1 / x for x in loss_and_output])
lao = lao / lao.sum()

json.dump(list(lao), open('para/catboost_weight.json', 'w'))
json.dump(loss_and_output, open('para/all_log.json', 'w'))