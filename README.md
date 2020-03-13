"# LaneSeg" 
deeplabv3+
将encoder-decoder和ASPP相结合。encoder-decoder获取更多边界信息，ASPP获取更多特征信息。

encoder 
    backbone：论文中是resnet101 或者 xception
    xception:大量使用Sep Conv（深度可分离卷积）替换max pooling
        论文中提到，通过实验，1*1 Conv后不加relu，收敛速度和效果优于“加relu”，我猜测，浅层feature进行relu会导致损失信息
    aspp:多尺度带洞卷积结构
    1、1*1 Conv
    2、3*3 Conv，rate 6
    3、3*3 Conv，rate 12
    4、3*3 Conv，rate 18
    5、Image Pooling
    将多个卷积concat拼接，然后1*1 Conv，bn，relu，上采样4倍，output_stride由16变为4

decoder
    从encoder中，取对应相同分辨率（output_stride=4）的feature，然后1*1 Conv降通道，之后与encoder结果进行concat，然后3*3 Conv微调特征，最后直接上采样4倍得到分割结果
