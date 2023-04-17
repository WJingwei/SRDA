from torch import nn
from model.DConvSR import DConvSR


from model.resnetAM import resnet50


class DConv_ResnetSR_map(nn.Module):
    def __init__(self,upscale_factor):
        super(DConv_ResnetSR_map,self).__init__()
        self.DConvSR = DConvSR(upscale_factor).cuda()
        self.ResNet = resnet50(num_classes=3, include_top=True).cuda()

    def forward(self,inputdata):
        sr,x_map= self.DConvSR(inputdata)
        feat, y_map = self.ResNet(sr)
        return sr,feat,x_map,y_map


