import math
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    Depth Separable Convolution.
    """

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.depth_wise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding,
                                    dilation, groups=inplanes, bias=bias)
        # self.bn = nn.BatchNorm2d(inplanes)
        self.point_wise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias)

    def forward(self, x):
        x = self.depth_wise(x)
        # x = self.bn(x)
        x = self.point_wise(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.features = nn.Sequential(
            SeparableConv2d(inplanes, planes, 3, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),
            SeparableConv2d(planes, planes, 3, stride=1, dilation=dilation),
            nn.ReLU(inplace=True),
            SeparableConv2d(planes, planes, 3, stride=1, dilation=dilation)
        )

        self.downsample = None
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        x = self.features(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        return x


class AlignedXception(nn.Module):
    """
    Ref:
        Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
    """

    def __init__(self, output_stride=16):
        """
        :param output_stride: Multiples of image down-sampling. The default value is 16(DeepLab v3+) or
        it can be set to 8(DeepLab v3).
        """
        super(AlignedXception, self).__init__()
        if output_stride == 8:
            self.stride = [1, 1]
            self.dilation = [4, 4]
        elif output_stride == 16:
            self.stride = [2, 1]
            self.dilation = [2, 2]
        elif output_stride == 32:
            self.stride = [2, 2]
            self.dilation = [1, 1]
        else:
            raise NotImplementedError

        # Entry flow
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            BasicConv2d(64, 128, 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = BasicConv2d(128, 256, 2)
        self.stage3 = BasicConv2d(256, 728, self.stride[0])

        # Middle flow
        layers = []
        for _ in range(16):
            layers.append(BasicConv2d(728, 728, stride=1, dilation=self.dilation[0]))
        self.stage4 = nn.Sequential(*layers)

        # Exit flow
        self.stage5 = nn.Sequential(
            BasicConv2d(728, 1024, stride=self.stride[1], dilation=self.dilation[1]),
            nn.ReLU(inplace=True),
            SeparableConv2d(1024, 1536, dilation=self.dilation[1]),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 1536, dilation=self.dilation[1]),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, dilation=self.dilation[1]),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        :param x:
        :return:
            result: Output two feature map to skip connect.
        """
        x = self.stem(x)
        x = self.stage1(x)
        low_level_features = x
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        return x, low_level_features
