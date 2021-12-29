import json
import pandas as pd
import numpy as np

train_set = pd.read_csv('data/parsed_train.tsv', sep='\t')
quiz_set = pd.read_csv('data/parsed_quiz.tsv', sep='\t')

quiz_set['target'] = 0
train_quiz = pd.concat([train_set, quiz_set], axis=0)

c1 = pd.get_dummies(train_quiz.shipment_method_id, prefix='sm')
c2 = pd.get_dummies(train_quiz.category_id, prefix='ci')
c3 = pd.get_dummies(train_quiz.package_size, prefix='ps')
c4 = pd.get_dummies(train_quiz.cross_city, prefix='cc')
c5 = pd.get_dummies(train_quiz.cross_state, prefix='cs')
train_quiz = pd.concat([train_quiz.drop(['record_number', 'target', 'shipment_method_id', 'category_id', 'package_size', 'cross_city', 'cross_state'], axis=1), 
               c1, c2, c3, c4, c5], axis=1)

train_quiz[:train_set.shape[0]].to_csv('data/parsed_train_cat.tsv', index=None, sep='\t')
train_quiz[train_set.shape[0]:].drop(['target'], axis=1).to_csv('data/parsed_quiz_cat.tsv', index=None, sep='\t')
