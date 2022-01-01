import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Regre(nn.Module):
    def __init__(self, flayer):
        super(L2Regre, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(flayer, 1),
        )
        
    def forward(self, data):
        x = self.h1(data)
        return x