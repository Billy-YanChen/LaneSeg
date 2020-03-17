# 对参数的管理，使得代码的简洁度很好
class Config(object):
    # model config
    OUTPUT_STRIDE = 16  # 下采样的次数
    ASPP_OUTDIM = 256  # 输出的channel
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8
    # resnet, xception, mobilenet
    MODEL_BACKBONE = "xception"
    TRAIN_BN_MOM = 0.1
    SYNC_BN = True
    FREEZE_BN = False
    # train config
    EPOCHS = 10
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 0.0006
