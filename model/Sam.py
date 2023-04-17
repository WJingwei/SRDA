import os
import torch
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Sam(nn.Module):
    def __init__(self):
        super(Sam, self).__init__()
        self.Sam = SpatialAttentionModul()
    def forward(self, x):
        x,map = self.Sam(x)
        return x,map
class SpatialAttentionModul(nn.Module):
    def __init__(self):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)
        x = Ms * x
        x = x.squeeze(-1)

        return x,Ms

