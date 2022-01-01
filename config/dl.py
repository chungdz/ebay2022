from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
import pandas as pd
 
class FNNData(Dataset):
    def __init__(self, train_data_path, test_mode=False):
        self.tm = test_mode
        if test_mode:
            x_test = pd.read_csv(train_data_path, sep='\t')
            self.x = torch.FloatTensor(x_test.drop(['record_number'], axis=1).values)
            self.record_number = torch.FloatTensor(x_test.record_number.values).unsqueeze(-1)
        else:
            x_train = pd.read_csv(train_data_path, sep='\t')
            self.x = torch.FloatTensor(x_train.drop(['target', 'record_number'], axis=1).values)
            self.y = torch.LongTensor(x_train.target.values).unsqueeze(-1)
            self.record_number = torch.FloatTensor(x_train.record_number.values).unsqueeze(-1)

    def __getitem__(self, index):
        if self.tm:
            return torch.cat([self.x[index], self.record_number[index]])
        else:
            return torch.cat([self.x[index], self.y[index], self.record_number[index]])
 
    def __len__(self):
        return self.x.size(0)

    def __feature_len__(self):
        return self.x.size(1)