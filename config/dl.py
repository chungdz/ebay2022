from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
import pandas as pd
 
class FNNData(Dataset):
    def __init__(self, train_data_path, test_mode=False):
        self.tm = test_mode
        to_drop_list = ['shipment_method_id', 'category_id', 'package_size', 'sender_state', 'receive_state']

        if test_mode:
            df = pd.read_csv(train_data_path, sep='\t')
            self.x = torch.FloatTensor(df.drop(['record_number'] + to_drop_list, axis=1).values)
        else:
            df = pd.read_csv(train_data_path, sep='\t')
            self.x = torch.FloatTensor(df.drop(['target', 'record_number'] + to_drop_list, axis=1).values)
            self.y = torch.LongTensor(df.target.values).unsqueeze(-1)
        
        self.record_number = torch.FloatTensor(df.record_number.values).unsqueeze(-1)
        self.x2 = torch.LongTensor(df[to_drop_list].values)

    def __getitem__(self, index):
        if self.tm:
            return torch.cat([self.x[index], self.record_number[index]])
        else:
            return torch.cat([self.x[index], self.y[index], self.record_number[index]])
 
    def __len__(self):
        return self.x.size(0)

    def __feature_len__(self):
        return self.x.size(1)