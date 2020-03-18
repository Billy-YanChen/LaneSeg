"# LaneSeg" 
本项目训练数据出自[无人车车道线检测挑战赛](http://aistudio.baidu.com/aistudio/#/competition/detail/5),主要训练框架为pytorch。训练模型尝试选用了unet，deeplabv3+等。
### 总体描述
本项目是第一次动手训练，代码参考了模型论文原文及各路大神的实现，在摸索中感受深度学习训练的流程及注意点。
### 代码结构
    |Projects - |data_list - train.csv  训练集数据路径
                           - val.csv  验证集数据路径
                       
                |models 模型代码存放
                        
                |utils  - data_feeder.py  数据读取、生成
                        - image_process.py  数据预处理
                        - make_lists.py   生成数据列表
                        - process_labels.py  label的编解码
                
                |train.py   训练脚本
                
                |predict.py  预测图片并标记
                
                |config.py   训练配置参数

### 学习笔记
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
