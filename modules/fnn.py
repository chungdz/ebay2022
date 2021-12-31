import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, flayer):
        super(FNN, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(flayer, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.h2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.h3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.h4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.h5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, data):
        
        x = self.h1(data)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        output = self.h5(x)

        return output