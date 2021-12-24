import xgboost as xgb
import json
import pandas as pd
import numpy as np

folds = 10
num_rounds = 15
esr = 3

params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        # larger -> under fit
        'gamma': 0,
        'max_depth': 10,
        # L2 Reg
        'lambda': 1,
        'subsample': 0.7,
        'colsample_bytree': 1,
        # child node number
        'min_child_weight': 3,
        # shrinkage
        'eta': 0.3,
        'seed': 1000,
        'nthread': 8,
    }

def custom_asymmetric_eval(preds, train_data):
    labels = train_data.get_label()
    preds = np.ceil(preds)
    residual = (labels - preds).astype("int")
    loss = np.where(residual < 0, residual * -0.6, residual * 0.4) 
    return "ebay_loss", np.mean(loss)

loss_and_output = []
for i in range(1, folds + 1):
    print('model:', i)
    train_set = pd.read_csv('data/subtrain/train_{}.tsv'.format(i), sep='\t')
    valid_set = pd.read_csv('data/subtrain/valid_{}.tsv'.format(i), sep='\t')

    x_train = train_set.drop(['record_number', 'target'],axis=1)
    y_train = train_set.target
    x_valid = valid_set.drop(['record_number', 'target'],axis=1)
    y_valid = valid_set.target

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_valid, y_valid)

    results = {}
    bst = xgb.train(params, dtrain, 
                    num_boost_round=num_rounds, 
                    evals=[(dtest, 'test1')], 
                    feval=custom_asymmetric_eval, 
                    evals_result=results,
                    early_stopping_rounds=esr)

    bst.save_model('para/xgb_{}.json'.format(i))
    loss_and_output.append(min(results['test1']['ebay_loss']))

lao = np.array([1 / x for x in loss_and_output])
lao = lao / lao.sum()

json.dump(list(lao), open('para/xgb_weight.json', 'w'))



