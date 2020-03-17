import os
import cv2
import torch
import numpy as np
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config
from utils.image_process import crop_resize_data
from utils.process_labels import decode_color_labels

# 对全局的环境变量进行设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 将设备数量，设备名打印出来
for dvi in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(dvi))

device_id = 0
predict_net = 'deeplabv3p'
# 设置一个字典，通过key值来选取使用的model
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}


def load_model(model_path):
    lane_config = Config()
    # neis做了一个通用化的处理，取出来config的配置文件
    net = nets[predict_net](lane_config)
    # eval（）不会进行反向传播
    net.eval()

    if torch.cuda.is_available():

        # 放到第device_id的GPU上进行运算，将数据从内存上传输到显存中
        net = net.cuda(device=device_id)
        # 加载模型
        map_location = 'cuda:%d' % device_id
    else:
        map_location = 'cpu'

    #
    model_param = torch.load(model_path, map_location=map_location)['state_dict']

    model_param = {k.replace('module.', ''): v for k, v in model_param.items()}
    net.load_state_dict(model_param)
    return net


def img_transform(img):
    img = crop_resize_data(img)
    # 轴
    img = np.transpose(img, (2, 0, 1))

    # batch的封装，添加了一个新的轴，在前面扩充一维，变成了 1*C*H*W,...不管图片后面哪一维，全都要
    img = img[np.newaxis, ...].astype(np.float32)

    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img


def get_color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    # 将channel取得的最大的响应作为标签
    pred = torch.argmax(pred, dim=1)
    # squeeze 将某些维度上的1 去掉
    pred = torch.squeeze(pred)

    pred = pred.detach().cpu().numpy()
    # 转换成color的label
    pred = decode_color_labels(pred)
    # 将通道数返回来
    pred = np.transpose(pred, (1, 2, 0))
    return pred


def main():
    # 预测图片的路径，存储在test_example
    # test_dir = 'test_example'
    # model_path = os.path.join(test_dir, predict_net + '_finalNet.pth.tar')
    model_path = os.path.join('./logs', 'finalNet.pth.tar')
    #model_path = os.path.join('E:/learn/result/resnet101', 'finalNet.pth.tar')
    '''model_dir = 'log'
    test_dir = 'test_example'
    model_path = os.path.join(model_dir,'finalNet.pth.tar'''''

    print('Loading model...')
    net = load_model(model_path)
    print('Done.')

    img_path = os.path.join('E:\\', '171206_025807977_Camera_5.jpg')
    img = cv2.imread(img_path)

    # 对图片进行tansform，可以让网络认识
    img = img_transform(img)

    print('Model infering...')

    # 进行model inference

    # __call__ 使用了forward进行前向传播
    pred = net(img)
    print('Done.')

    # 对预测的结果进行处理，进行了颜色的转换
    color_mask = get_color_mask(pred)
    cv2.imwrite(os.path.join('E:\\', 'color_mask2.jpg'), color_mask)


if __name__ == '__main__':
    main()
