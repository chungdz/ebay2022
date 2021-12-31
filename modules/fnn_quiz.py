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

def test(model, valid_data_loader):
    model.eval()  
        
    with torch.no_grad():
        preds = []
        for data in valid_data_loader:
            # 1. Forward
            pred = model(data)
            if pred.dim() > 1:
                pred = pred.squeeze()
            try:
                preds += pred.numpy().tolist()
            except:
                preds.append(int(pred.cpu().numpy()))

        preds = np.round(preds)
        return preds

parser = argparse.ArgumentParser()
parser.add_argument("--folds", default=10, type=int)
parser.add_argument("--save_path", default='para', type=str)
args = parser.parse_args()

test_dataset = FNNData('data/parsed_quiz_cat.tsv', test_mode=True)
test_dl = DataLoader(test_dataset, shuffle=False)
model = FNN(test_dataset.__feature_len__())

saved_model_path = os.path.join(args.saved_path, 'pfnn_{}'.format(1))
if not os.path.exists(saved_model_path):
    print("Not Exist: {}".format(saved_model_path))
    exit()
pretrained_model = torch.load(saved_model_path, map_location='cpu')
print(model.load_state_dict(pretrained_model, strict=False))
res = test(model, test_dl)

