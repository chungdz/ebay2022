import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, flayer, cfg):
        super(FNN, self).__init__()
        self.smi_emb = nn.Embedding(cfg['shipment_method_id'], 30)
        self.ci_emb = nn.Embedding(cfg['category_id'], 30)
        self.ps_emb = nn.Embedding(cfg['package_size'], 15)
        self.si_emb = nn.Embedding(cfg['state_info'], 30)

        self.h1 = nn.Sequential(
            nn.Linear(flayer, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.h2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # self.h3 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        # self.h4 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        self.h5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, data):
        smi = self.smi_emb(data[:, -5].long())
        ci = self.ci_emb(data[:, -4].long())
        ps = self.ps_emb(data[:, -3].long())
        ss = self.si_emb(data[:, -2].long())
        rs = self.si_emb(data[:, -1].long())
        
        x = torch.cat([data[:, :-5], smi, ci, ps, ss, rs], dim=1)

        x = self.h1(x)
        x = self.h2(x)
        # x = self.h3(x)
        # x = self.h4(x)
        x = self.h5(x)

        return x