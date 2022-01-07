import numpy as np
from catboost import Pool, CatBoostRegressor
import json
import pandas as pd
from tqdm import trange
from datetime import datetime, timedelta

def add_func(row):
    acct = row['acceptance_scan_timestamp']
    dd = row['target']
    cdate = datetime.strptime(acct.split()[0], "%Y-%m-%d")
    cdate = cdate + timedelta(days=dd)
    return cdate.strftime("%Y-%m-%d")

quiz_set = pd.read_csv('data/sl_data/parsed_quiz.tsv', sep='\t')
quiz_set.drop(['record_number'], axis=1, inplace=True)

real_quiz_set = pd.read_csv('data/quiz.tsv', sep='\t')

test_pool = Pool(quiz_set, feature_names=list(quiz_set.columns))

folds = 10
w = json.load(open('para/l2_catboost_weight.json', 'r'))
final_day = np.zeros((quiz_set.shape[0]))
for i in range(1, folds + 1):
    cb = CatBoostRegressor()
    cb.load_model('para/l2_catboost_{}.cbm'.format(i))
    ypred = cb.predict(test_pool)
    final_day = final_day + w[i - 1] * ypred

to_save = []
for rnumber, predict_value in zip(real_quiz_set['record_number'].values, final_day):
    to_save.append([rnumber, predict_value])
savedf = pd.DataFrame(to_save, columns=['record_number', 'catboost_predict'])
savedf.to_csv('data/sl_data/l2_catboost_quiz.tsv', sep='\t', index=None) 

real_quiz_set['target'] = pd.Series(np.round(final_day))
res_set = real_quiz_set[['record_number', 'acceptance_scan_timestamp', 'target']]
res_set['arrive_date'] = res_set.apply(add_func, axis=1)
print("null res:", sum(res_set['arrive_date'].isnull()))
res_set.drop(columns=['acceptance_scan_timestamp', 'target']).to_csv('result/l2_catboost_result.tsv', header=None, index=None, sep='\t')