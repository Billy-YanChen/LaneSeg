"""
参考https://github.com/jfzhang95/pytorch-deeplab-xception
@Author  : yanchen
@Software: PyCharm
@Time    : 2020/3/3 15:42
"""
from model.backbone import resnet, xception, drn, mobilenet,AlignedXception


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
        #return AlignedXception.AlignedXception(output_stride)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
