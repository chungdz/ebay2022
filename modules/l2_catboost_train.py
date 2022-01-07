import numpy as np
from catboost import Pool, CatBoostRegressor
import json
import pandas as pd
from tqdm import trange
import argparse
import gc

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

parser = argparse.ArgumentParser()
parser.add_argument("--starti", default=1, type=int)
parser.add_argument("--depth", default=4, type=int)
parser.add_argument("--num_rounds", default=1000, type=int)
parser.add_argument("--border_count", default=254, type=int)
parser.add_argument("--random_strength", default=1, type=float)
parser.add_argument("--esr", default=5, type=int)
parser.add_argument("--l2_leaf", default=3, type=float)
args = parser.parse_args()

loss_and_output = []
all_log = []
total_rc = []
total_preds = []
for i in trange(args.starti, folds + 1):
    print('model:', i)
    train_set = pd.read_csv('data/sl_data/subtrain/train_{}.tsv'.format(i), sep='\t')
    valid_set = pd.read_csv('data/sl_data/subtrain/valid_{}.tsv'.format(i), sep='\t')
    
    x_train = train_set.drop(['record_number', 'target'], axis=1)
    y_train = train_set.target
    x_valid = valid_set.drop(['record_number', 'target'], axis=1)
    y_valid = valid_set.target
    print(x_train.columns)

    train_pool = Pool(x_train, 
                  y_train,
                  feature_names=list(x_train.columns))
    test_pool = Pool(x_valid,
                 y_valid,
                 feature_names=list(x_valid.columns))

    model = CatBoostRegressor(iterations=args.num_rounds, 
                          depth=args.depth,
                          border_count=args.border_count,
                          learning_rate=1, 
                          loss_function='RMSE',
                          random_strength=args.random_strength,
                          one_hot_max_size=8,
                          l2_leaf_reg=args.l2_leaf,
                          grow_policy='SymmetricTree',
                          eval_metric=EbayMetric())
    
    model.fit(train_pool, early_stopping_rounds=args.esr, eval_set=test_pool, use_best_model=True, log_cout=open('result/output.txt', 'w'))
    model.save_model('para/l2_catboost_{}.cbm'.format(i))
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
    
    preds = model.predict(test_pool)
    total_preds += preds.flatten().tolist()
    total_rc += valid_set.record_number.values.tolist()
    del train_set, valid_set, x_train, x_valid, y_train, y_valid, train_pool, test_pool
    gc.collect()

to_save = []
for rnumber, predict_value in zip(total_rc, total_preds):
    to_save.append([rnumber, predict_value])
savedf = pd.DataFrame(to_save, columns=['record_number', 'catboost_predict'])
savedf.to_csv('data/sl_data/l2_catboost_train.tsv', sep='\t', index=None) 

lao = np.array([1 / x for x in loss_and_output])
lao = lao / lao.sum()

json.dump(list(lao), open('para/l2_catboost_weight.json', 'w'))
json.dump(loss_and_output, open('para/l2_all_log.json', 'w'))
print('mean loss:', np.mean(loss_and_output))