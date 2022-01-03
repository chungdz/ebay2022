#!/usr/bin/env python
# coding: utf-8

import lightgbm as lgb
import json
import pandas as pd
import numpy as np

folds = 10
num_rounds = 130
esr = 3

def custom_asymmetric_eval(preds, train_data):
    labels = train_data.get_label()
    preds = np.round(preds)
    residual = (labels - preds).astype("int")
    loss = np.where(residual < 0, residual * -0.6, residual * 0.4) 
    return "ebay_loss", np.mean(loss), False

params = {
    'boosting_type': 'dart',
    'objective': 'regression',
#     'metric': {'l2', 'l1', custom_asymmetric_eval},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

loss_and_output = []
for i in range(1, folds + 1):
    print('model:', i)
    train_set = pd.read_csv('data/subtrain/train_{}.tsv'.format(i), sep='\t')
    valid_set = pd.read_csv('data/subtrain/valid_{}.tsv'.format(i), sep='\t')

    x_train = train_set.drop(['record_number', 'target'],axis=1)
    y_train = train_set.target
    x_valid = valid_set.drop(['record_number', 'target'],axis=1)
    y_valid = valid_set.target
    
    cat_feats = ['shipment_method_id','category_id', 'bt', 'package_size', 'cross_city', 'cross_state','sender_state', 'receive_state',
                 'isNextDay','isHoliday']
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cat_feats, params={"max_bin": 1000})
    lgb_eval = lgb.Dataset(x_valid, y_valid, categorical_feature=cat_feats, reference=lgb_train)
    
    results = {}
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=num_rounds,
                valid_sets=lgb_eval,
                categorical_feature=cat_feats,
                feval=custom_asymmetric_eval,
                evals_result=results,
                early_stopping_rounds=esr)
    
    gbm.save_model('para/lgbm_{}.txt'.format(i))
    loss_and_output.append(min(results['valid_0']['ebay_loss']))
    

lao = np.array([1 / x for x in loss_and_output])
lao = lao / lao.sum()

json.dump(list(lao), open('para/lgbm_weight.json', 'w'))

