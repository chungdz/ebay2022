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

quiz_set = pd.read_csv('data/parsed_quiz.tsv', sep='\t')
quiz_set['cross_city'] = quiz_set['cross_city'].astype('int')
quiz_set['cross_state'] = quiz_set['cross_state'].astype('int')
quiz_set['sender_state'] = quiz_set['sender_state'].astype('int')
quiz_set['receive_state'] = quiz_set['receive_state'].astype('int')

to_drop = json.load(open('config/to_drop.json', 'r'))
quiz_set.drop(['record_number'] + to_drop, axis=1, inplace=True)

quiz_set_1 = quiz_set[['carrier_max_estimate', 'carrier_min_estimate', 'acc_date', 'acc_hour', 'declared_handling_days']]
quiz_set_2 = quiz_set[['shipment_method_id', 'dis', 'sender_state', 'package_size', 'seller_size', 'category_id', 'receive_state', 'cross_state','shipping_fee','item_price','weight','shipping_units','tz_dis','bt','quantity','cross_city']]
cat_index = []
cat_set = set({"shipment_method_id", "category_id", "bt", "package_size",
"cross_state", "sender_state", "receive_state","cross_city"})
for idx, cn in enumerate(quiz_set_2.columns):
    if cn in cat_set:
        cat_index.append(idx)

test_pool_1 = Pool(quiz_set_1,
                 #cat_features=cat_index,
                 feature_names=list(quiz_set_1.columns))

test_pool_2 = Pool(quiz_set_2,
                 cat_features=cat_index,
                 feature_names=list(quiz_set_2.columns))


real_quiz_set = pd.read_csv('data/quiz.tsv', sep='\t')

# model weights, according to inverse of mean loss
folds = 10
m1_w = np.mean(json.load(open('para/all_log_1.json', 'r')))
m2_w = np.mean(json.load(open('para/all_log_2.json', 'r')))
model_weight = np.array([1/m1_w, 1/m2_w])
model_weight = model_weight/model_weight.sum()

w1 = json.load(open('para/catboost_weight_1.json', 'r'))
w2 = json.load(open('para/catboost_weight_2.json', 'r'))
final_day = np.zeros((quiz_set.shape[0]))
for i in range(1, folds + 1):
    cb1 = CatBoostRegressor()
    cb1.load_model('para/catboost_1_{}.cbm'.format(i))
    ypred1 = cb1.predict(test_pool_1)

    cb2 = CatBoostRegressor()
    cb2.load_model('para/catboost_2_{}.cbm'.format(i))
    ypred2 = cb2.predict(test_pool_2)
    final_day = final_day + w1[i - 1] * ypred1 * model_weight[0] + w2[i - 1] * ypred2 * model_weight[1]

to_save = []
for rnumber, predict_value in zip(real_quiz_set['record_number'].values, final_day):
    to_save.append([rnumber, predict_value])
savedf = pd.DataFrame(to_save, columns=['record_number', 'catboost_predict'])
savedf.to_csv('data/sl_data/catboost_quiz_adj.tsv', sep='\t', index=None)

real_quiz_set['target'] = pd.Series(np.round(final_day))
res_set = real_quiz_set[['record_number', 'acceptance_scan_timestamp', 'target']]
res_set['arrive_date'] = res_set.apply(add_func, axis=1)
print("null res:", sum(res_set['arrive_date'].isnull()))
res_set.drop(columns=['acceptance_scan_timestamp', 'target']).to_csv('result/catboost_result_adj.tsv', header=None, index=None, sep='\t')
