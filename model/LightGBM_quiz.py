import json
import pandas as pd
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import lightgbm as lgb

def add_func(row):
    acct = row['acceptance_scan_timestamp']
    dd = row['target']
    cdate = datetime.strptime(acct.split()[0], "%Y-%m-%d")
    cdate = cdate + timedelta(days=dd)
    return cdate.strftime("%Y-%m-%d")

cat_feats = ['shipment_method_id','category_id', 'bt', 'package_size', 'cross_city', 'cross_state']

quiz_set = pd.read_csv('data/parsed_quiz.tsv', sep='\t').drop(['record_number'],axis=1)
real_quiz_set = pd.read_csv('data/quiz.tsv', sep='\t')

dtest = lgb.Dataset(quiz_set, categorical_feature=cat_feats)
folds = 10

w = json.load(open('para/lgbm_weight.json', 'r'))
final_day = np.zeros((quiz_set.shape[0]))
for i in range(1, folds + 1):
    bst = lgb.Booster(model_file='para/lgbm_{}.txt'.format(i))
    ypred = bst.predict(quiz_set, num_iteration=bst.best_iteration)
    final_day = final_day + w[i - 1] * ypred

real_quiz_set['target'] = pd.Series(np.round(final_day))
res_set = real_quiz_set[['record_number', 'acceptance_scan_timestamp', 'target']]
res_set['arrive_date'] = res_set.apply(add_func, axis=1)
print("null res:", sum(res_set['arrive_date'].isnull()))
res_set.drop(columns=['acceptance_scan_timestamp', 'target']).to_csv('result/lgb_result.tsv', header=None, index=None, sep='\t')
