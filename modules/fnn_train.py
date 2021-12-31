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

def run(cfg, train_dataset, valid_dataset, fp):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    
    set_seed(7)
    # Build Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Build model.
    model = FNN(train_dataset.__feature_len__())
    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    min_ebay = -1
    # Training and validation
    
    for epoch in range(cfg.epoch):
        # print(model.match_prediction_layer.state_dict()['2.bias'])
        train(cfg, epoch, model, train_data_loader, optimizer, steps_one_epoch)
        eval_return = validate(cfg, model, valid_data_loader)
        print(epoch, eval_return)
        
        if min_ebay == -1 or eval_return < min_ebay:
            min_ebay = eval_return
            torch.save(model.state_dict(), fp)
        
    return min_ebay


def train(cfg, epoch, model, loader, optimizer, steps_one_epoch):
    model.train()
    model.zero_grad()
    enum_dataloader = enumerate(tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch)))

    for i, data in enum_dataloader:
        if i >= steps_one_epoch:
            break
        
        # 1. Forward
        pred = model(data[:, :-1]).squeeze()
        loss = F.mse_loss(pred, data[:, -1])

        # 3.Backward.
        loss.backward()

        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # scheduler.step()
        model.zero_grad()

def validate(cfg, model, valid_data_loader):
    model.eval()  
        
    with torch.no_grad():
        preds, truths = list(), list()
        for data in valid_data_loader:
            # 1. Forward
            pred = model(data[:, :-1])
            if pred.dim() > 1:
                pred = pred.squeeze()
            try:
                preds += pred.numpy().tolist()
            except:
                preds.append(int(pred.cpu().numpy()))
            truths += data[:, -1].numpy().tolist()

        preds = np.round(preds)
        residual = (truths - preds).astype("int")
        loss = np.where(residual < 0, residual * -0.6, residual * 0.4)

        return np.mean(loss)

parser = argparse.ArgumentParser()
parser.add_argument("--folds", default=10, type=int)
parser.add_argument("--epoch", default=3, type=int)
parser.add_argument("--lr", default=0.001, type=int)
parser.add_argument("--save_path", default='para', type=str)
args = parser.parse_args()

loss_and_output = []
for i in range(1, args.folds + 1):
    print('model:', i)
    train_dataset = FNNData('data/subtrain_cat/train_{}.tsv'.format(i))
    valid_dataset = FNNData('data/subtrain_cat/valid_{}.tsv'.format(i))
    cur_loss = run(args, train_dataset, valid_dataset, os.path.join(args.save_path, 'pfnn_{}'.format(i)))
    loss_and_output.append(cur_loss)

lao = np.array([1 / x for x in loss_and_output])
lao = lao / lao.sum()

json.dump(list(lao), open('para/pfnn_weight.json', 'w'))
json.dump(loss_and_output, open('para/all_pfnn_log.json', 'w'))
print('mean loss:', np.mean(loss_and_output))

