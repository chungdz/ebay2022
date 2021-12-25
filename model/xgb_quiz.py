import xgboost as xgb
import json
import pandas as pd
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def add_func(row):
    acct = row['acceptance_scan_timestamp']
    dd = row['target']
    cdate = datetime.strptime(acct.split()[0], "%Y-%m-%d")
    cdate = cdate + timedelta(days=dd)
    return cdate.strftime("%Y-%m-%d")

quiz_set = pd.read_csv('data/parsed_quiz.tsv', sep='\t').drop(['record_number'],axis=1)
real_quiz_set = pd.read_csv('data/quiz.tsv', sep='\t')
dtest = xgb.DMatrix(quiz_set)
folds = 10

w = json.load(open('para/xgb_weight.json', 'r'))
final_day = np.zeros((quiz_set.shape[0]))
for i in range(1, folds + 1):
    bst = xgb.Booster()
    bst.load_model('para/xgb_{}.json'.format(i))
    ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
    final_day = final_day + w[i - 1] * ypred


real_quiz_set['target'] = pd.Series(np.round(final_day)) + 1
res_set = real_quiz_set[['record_number', 'acceptance_scan_timestamp', 'target']]
res_set['arrive_date'] = res_set.apply(add_func, axis=1)
print("null res:", sum(res_set['arrive_date'].isnull()))
res_set.drop(columns=['acceptance_scan_timestamp', 'target']).to_csv('result/xgb_result.tsv', header=None, index=None, sep='\t')