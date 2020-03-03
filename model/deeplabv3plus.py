import torch
import torch.nn as nn
from model.atrous_resnet import resnet50_atrous
from model.module import ASPP


class DeeplabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeeplabV3Plus, self).__init__()
        #调用backbone
        self.backbone = resnet50_atrous(pretrained=True, os=cfg.OUTPUT_STRIDE)
        input_channel = 2048
        self.aspp = ASPP(in_chans=input_channel, out_chans=cfg.ASPP_OUTDIM, rate=16//cfg.OUTPUT_STRIDE)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.OUTPUT_STRIDE//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, cfg.SHORTCUT_DIM, cfg.SHORTCUT_KERNEL, 1, padding=cfg.SHORTCUT_KERNEL//2,bias=False),
                nn.BatchNorm2d(cfg.SHORTCUT_DIM),
                nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
                nn.Conv2d(cfg.ASPP_OUTDIM+cfg.SHORTCUT_DIM, cfg.ASPP_OUTDIM, 3, 1, padding=1,bias=False),
                nn.BatchNorm2d(cfg.ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(cfg.ASPP_OUTDIM, cfg.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(cfg.ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.ASPP_OUTDIM, cfg.NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layers = self.backbone(x)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow],1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result
