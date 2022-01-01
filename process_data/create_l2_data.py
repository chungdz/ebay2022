import pandas as pd
from datetime import datetime
import time
import json
from tqdm import tqdm

train_set = pd.read_csv('data/parsed_train.tsv', sep='\t').filter(items=['record_number', 'target'], axis=1)
quiz_set = pd.read_csv('data/parsed_quiz.tsv', sep='\t').filter(items=['record_number'], axis=1)
quiz_set['target'] = 0
print('origin data:', train_set.shape, quiz_set.shape)

pfnn_train = pd.read_csv('data/sl_data/pfnn_train.tsv', sep='\t')
pfnn_quiz = pd.read_csv('data/sl_data/pfnn_quiz.tsv', sep='\t')
print('pfnn shape:', pfnn_train.shape, pfnn_quiz.shape)

catboost_train = pd.read_csv('data/sl_data/catboost_train.tsv', sep='\t')
catboost_quiz = pd.read_csv('data/sl_data/catboost_quiz.tsv', sep='\t')
print('catboost shape:', catboost_train.shape, catboost_quiz.shape)

train_set = train_set.merge(pfnn_train, how='inner', on='record_number')
quiz_set = quiz_set.merge(pfnn_quiz, how='inner', on='record_number')
print('merge one:', train_set.shape, quiz_set.shape)

train_set = train_set.merge(catboost_train, how='inner', on='record_number')
quiz_set = quiz_set.merge(catboost_quiz, how='inner', on='record_number')
print('merge two:', train_set.shape, quiz_set.shape)

train_quiz = pd.concat([train_set, quiz_set], axis=0)
train_quiz['pFNN_predict'] = (train_quiz['pFNN_predict'] - train_quiz['pFNN_predict'].mean()) / train_quiz['pFNN_predict'].std()
train_quiz['catboost_predict'] = (train_quiz['catboost_predict'] - train_quiz['catboost_predict'].mean()) / train_quiz['catboost_predict'].std()

train_quiz[:train_set.shape[0]].to_csv('data/sl_data/parsed_train.tsv', index=None, sep='\t')
train_quiz[train_set.shape[0]:].drop(['target'], axis=1).to_csv('data/sl_data/parsed_quiz.tsv', index=None, sep='\t')