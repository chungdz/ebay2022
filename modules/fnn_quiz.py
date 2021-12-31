import numpy as np
import json
import pandas as pd
from tqdm import trange
import argparse
from modules.fnn import FNN
from utils.train_util import set_seed
from torch.utils.data import DataLoader
from config.dl import FNNData
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime, timedelta

def test(cfg, model, valid_data_loader):
    model.eval()  
        
    with torch.no_grad():
        preds = []
        for data in tqdm(valid_data_loader, total=len(valid_data_loader), desc='test'):
            # 1. Forward
            pred = model(data)
            # pred = torch.argmax(pred, dim=1)
            preds.append(pred)

        return np.concatenate(preds, axis=0)

def add_func(row):
    acct = row['acceptance_scan_timestamp']
    dd = row['target']
    cdate = datetime.strptime(acct.split()[0], "%Y-%m-%d")
    cdate = cdate + timedelta(days=dd)
    return cdate.strftime("%Y-%m-%d")

parser = argparse.ArgumentParser()
parser.add_argument("--folds", default=10, type=int)
parser.add_argument("--save_path", default='para', type=str)
parser.add_argument("--batch_size", default=64, type=int)
args = parser.parse_args()

test_dataset = FNNData('data/parsed_quiz_cat.tsv', test_mode=True)
model = FNN(test_dataset.__feature_len__())
w = json.load(open('para/pfnn_weight.json', 'r'))
final_day = np.zeros((test_dataset.__len__(), 128))
for i in range(1, args.folds + 1):
    test_dl = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    saved_model_path = os.path.join(args.save_path, 'pfnn_{}'.format(i))
    # saved_model_path = os.path.join(args.save_path, 'fnn_tmp')
    if not os.path.exists(saved_model_path):
        print("Not Exist: {}".format(saved_model_path))
        exit()
    pretrained_model = torch.load(saved_model_path, map_location='cpu')
    print(model.load_state_dict(pretrained_model, strict=False))
    res = test(args, model, test_dl)
    final_day = final_day + w[i - 1] * res
    
real_quiz_set = pd.read_csv('data/quiz.tsv', sep='\t')
real_quiz_set['target'] = pd.Series(np.argmax(final_day, axis=1))
res_set = real_quiz_set[['record_number', 'acceptance_scan_timestamp', 'target']]
res_set['arrive_date'] = res_set.apply(add_func, axis=1)
print("null res:", sum(res_set['arrive_date'].isnull()))
res_set.drop(columns=['acceptance_scan_timestamp', 'target']).to_csv('result/fnn.tsv', header=None, index=None, sep='\t')


