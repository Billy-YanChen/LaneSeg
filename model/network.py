import torch.nn as nn
import torchvision.models as models
from model.module import Block, Bottleneck, DownBottleneck, Layer

#pytorch Torchvision

class ResNet101v2(nn.Module):
    '''
    ResNet101 model 
    '''
    def __init__(self):
        super(ResNet101v2, self).__init__()
        #下采样2倍
        self.conv1 = Block(3, 64, 7, 3, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2_1 =DownBottleneck(64, 256, stride=1)
        self.conv2_2 =Bottleneck(256, 256)
        self.conv2_3 =Bottleneck(256, 256)

        #下采样2倍 8倍
        self.layer3 = Layer(256, [512]*2, "resnet")

        #下采样2倍 16倍
        self.layer4 = Layer(512, [1024]*23, "resnet")

        #下采样2倍 32倍
        self.layer5 = Layer(1024, [2048]*3, "resnet")

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f2, f3, f4, f5]
