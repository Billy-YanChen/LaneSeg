from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3plus import DeeplabV3Plus
from config import Config


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# device_list = [2, 6]
device_list = [7,0]

def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    #model 转化成训练的状态
    net.train()
    total_mask_loss = 0.0

    #这里是一个dataloader
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:

        #取出来batch_item中的value
        image, mask = batch_item['image'], batch_item['mask']

        #检测环境中是否存在cuda
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])

        #optimizer.zero将每个parameter的梯度清0
        optimizer.zero_grad()
        #输出预测的mask
        out = net(image)

        #计算交叉熵loss
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        total_mask_loss += mask_loss.item()

        #进行后向计算
        mask_loss.backward()

        #optimizer进行更新
        optimizer.step()


        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    #记录数据迭代了多少次
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF, config):
    #将model转化成了eval
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)

    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)

        #detach（）截断梯度的作用，可以不截断，查一下用法
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)

        #计算iou
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    testF.write("Epoch:{} \n".format(epoch))
    #求出每一个类别的iou
    for i in range(8):
        result_string = "{}: {:.4f} \n".format(i, result["TP"][i]/result["TA"][i])
        print(result_string)

        #写入log文件
        testF.write(result_string)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()


def adjust_lr(optimizer, epoch):

    #多机多卡上的 trick：warming up
    if epoch == 0:
        lr = 1e-3
    elif epoch == 2:
        lr = 1e-2
    elif epoch == 100:
        lr = 1e-3
    elif epoch == 150:
        lr = 1e-4
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    #设置model parameters
    lane_config = Config()

    #查看路径是否存在
    if os.path.exists(lane_config.SAVE_PATH):
        #如果存在的话，全部删掉
        shutil.rmtree(lane_config.SAVE_PATH)
    #建立一个新的文件件
    os.makedirs(lane_config.SAVE_PATH, exist_ok=True)

    #打开文件夹，在这两个文件内记录
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "test.csv"), 'w')

    #set up dataset
    # 'pin_memory'意味着生成的Tensor数据最开始是属于内存中的索页，这样的话转到GPU的显存就会很快
    kwargs = {
        #'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    #set up training dataset
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    #set up training dataset 的dataloader
    train_data_batch = DataLoader(train_dataset, batch_size=2*len(device_list), shuffle=True, drop_last=True, **kwargs)

    #set ip validation dataset
    val_dataset = LaneDataset("val.csv", transform=transforms.Compose([ToTensor()]))

    #set up validation dataset's dataloader
    val_data_batch = DataLoader(val_dataset, batch_size=1*len(device_list), shuffle=False, drop_last=False, **kwargs)

    #build model

    net = DeeplabV3Plus(lane_config)
    net.eval()
    #检测一下环境中是否存在GPU，存在的话就转化成cuda的格式
    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])
        net = torch.nn.DataParallel(net, device_ids=device_list)

    #config the optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)

    #查一下weight_decay的作用
    optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)

    #Training and test
    for epoch in range(lane_config.EPOCHS):
        # adjust_lr(optimizer, epoch)
        #在train_epoch中
        train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config)

        test(net, epoch, val_data_batch, testF, lane_config)

        if epoch % 2 == 0:
            torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
    trainF.close()
    testF.close()


    torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth.tar"))


if __name__ == "__main__":
    main()