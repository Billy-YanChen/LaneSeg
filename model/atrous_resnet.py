import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

bn_mom = 0.0003
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}


def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chans, out_chans, stride, atrous)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_chans, out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNet_Atrous, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.' % os)

        self.inplanes = 64

        # 下采样2倍
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 下采样2倍
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])

        # 下采样2倍
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])
        # 下采样2倍
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16 // os)
        # 不进行下采样，使用了空洞卷积
        # layer 456 都是3层，3个block，每个block都会有一个3*3 的卷积
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=stride_list[2],
                                       atrous=[item * 16 // os for item in atrous])
        self.layer5 = self._make_layer(block, 2048, 512, layers[3], stride=1,
                                       atrous=[item * 16 // os for item in atrous])
        self.layer6 = self._make_layer(block, 2048, 512, layers[3], stride=1,
                                       atrous=[item * 16 // os for item in atrous])
        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_chans, out_chans, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        if stride != 1 or in_chans != out_chans * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans * block.expansion),
            )

        layers = []
        layers.append(block(in_chans, out_chans, stride=stride, atrous=atrous[0], downsample=downsample))
        in_chans = out_chans * 4
        for i in range(1, blocks):
            layers.append(block(in_chans, out_chans, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        layers_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layers_list.append(x)
        x = self.layer2(x)
        layers_list.append(x)
        x = self.layer3(x)
        layers_list.append(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        layers_list.append(x)

        return layers_list


def resnet50_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        # old_dict = model_zoo.load_url(model_urls['resnet50'])
        old_dict = torch.load('E:/learn/resnet50-19c8e357.pth')
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


# 模型加载的部分，将pre-trained的weight加载进模型
def resnet101_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-101 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        # model_urls是一个加载的dict
        # old_dict 返回了一个字典 parameter：value pretrained weights
        # old_dict = model_zoo.load_url(model_urls['resnet101'])
        old_dict = torch.load('E:/learn/resnet101-5d3b4d8f.pth')
        # 返回model对应的字典
        model_dict = model.state_dict()
        # pretrained weights中的item
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        # 策略：如果old_dict和model 中有相同的key，那么就更新
        model_dict.update(old_dict)
        # 将weights加载进模型中，变成了pretrained weight的model
        model.load_state_dict(model_dict)
    return model
