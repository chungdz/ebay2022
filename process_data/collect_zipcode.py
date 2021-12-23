import pandas as pd
import json

train_set = pd.read_csv('data/train.tsv', sep='\t')
quiz_set = pd.read_csv('data/quiz.tsv', sep='\t')
zip_code = set(train_set['item_zip']) | set(train_set['buyer_zip']) | set(quiz_set['item_zip']) | set(quiz_set['buyer_zip'])
json.dump(list(zip_code), open('zip_code_list.json', 'w', encoding='utf-8'))