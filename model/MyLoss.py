import torch
from torch import nn

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss,self).__init__()
        self.SRloss = torch.nn.L1Loss().cuda()
        self.claloss = nn.CrossEntropyLoss().cuda()
        self.maploss = torch.nn.MSELoss().cuda()
    def forward(self, output_img, real_img, feat, x_map, y_map, label):
        SRloss=self.SRloss(output_img, real_img)
        claloss = self.claloss(feat,label)
        maploss = self.maploss(x_map,y_map)
        MyLoss = SRloss + 0.0001 * maploss + 0.0006 * claloss

        return MyLoss