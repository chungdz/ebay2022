import json
from unicodedata import category
import pandas as pd
import numpy as np

train_set = pd.read_csv('data/parsed_train.tsv', sep='\t')
quiz_set = pd.read_csv('data/parsed_quiz.tsv', sep='\t')

quiz_set['target'] = 0
train_quiz = pd.concat([train_set, quiz_set], axis=0)

train_quiz['shipping_fee'] = (train_quiz['shipping_fee'] - train_quiz['shipping_fee'].mean()) / train_quiz['shipping_fee'].std()
train_quiz['carrier_min_estimate'] = (train_quiz['carrier_min_estimate'] - train_quiz['carrier_min_estimate'].mean()) / train_quiz['carrier_min_estimate'].std()
train_quiz['carrier_max_estimate'] = (train_quiz['carrier_max_estimate'] - train_quiz['carrier_max_estimate'].mean()) / train_quiz['carrier_max_estimate'].std()
train_quiz['item_price'] = (train_quiz['item_price'] - train_quiz['item_price'].mean()) / train_quiz['item_price'].std()
train_quiz['quantity'] = (train_quiz['quantity'] - train_quiz['quantity'].mean()) / train_quiz['quantity'].std()
train_quiz['weight'] = (train_quiz['weight'] - train_quiz['weight'].mean()) / train_quiz['weight'].std()
train_quiz['tz_dis'] = (train_quiz['tz_dis'] - train_quiz['tz_dis'].mean()) / train_quiz['tz_dis'].std()
train_quiz['dis'] = (train_quiz['dis'] - train_quiz['dis'].mean()) / train_quiz['dis'].std()
train_quiz['acc_hour'] = (train_quiz['acc_hour'] - train_quiz['acc_hour'].mean()) / train_quiz['acc_hour'].std()
train_quiz['pay_hour'] = (train_quiz['pay_hour'] - train_quiz['pay_hour'].mean()) / train_quiz['pay_hour'].std()
train_quiz['acc_date'] = (train_quiz['acc_date'] - train_quiz['acc_date'].mean()) / train_quiz['acc_date'].std()
train_quiz['shipping_units'] = (train_quiz['shipping_units'] - train_quiz['shipping_units'].mean()) / train_quiz['shipping_units'].std()
train_quiz['declared_handling_days'] = (train_quiz['declared_handling_days'] - train_quiz['declared_handling_days'].mean()) / train_quiz['declared_handling_days'].std()
train_quiz['seller_size'] = (train_quiz['seller_size'] - train_quiz['seller_size'].mean()) / train_quiz['seller_size'].std()

to_embed = {'shipment_method_id': train_quiz['shipment_method_id'].max() + 1, 
            'category_id': train_quiz['category_id'].max() + 1, 
            'package_size': train_quiz['package_size'].max() + 1, 
            'state_info': int(max(train_quiz['sender_state'].max(), train_quiz['receive_state'].max()) + 1)}

# c1 = pd.get_dummies(train_quiz.shipment_method_id, prefix='sm')
# c2 = pd.get_dummies(train_quiz.category_id, prefix='ci')
# c3 = pd.get_dummies(train_quiz.package_size, prefix='ps')
c4 = pd.get_dummies(train_quiz.cross_city, prefix='cc')
c5 = pd.get_dummies(train_quiz.cross_state, prefix='cs')
# c6 = pd.get_dummies(train_quiz.sender_state, prefix='ss')
# c7 = pd.get_dummies(train_quiz.receive_state, prefix='rs')
train_quiz = pd.concat([train_quiz.drop(['cross_city', 'cross_state'], axis=1), c4, c5], axis=1)

train_quiz[:train_set.shape[0]].to_csv('data/parsed_train_cat.tsv', index=None, sep='\t')
train_quiz[train_set.shape[0]:].drop(['target'], axis=1).to_csv('data/parsed_quiz_cat.tsv', index=None, sep='\t')

json.dump(to_embed, open('data/category_info.json', 'w'))
