import os
import cv2
import numpy as np
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
import albumentations as albu
import matplotlib.pyplot as plt
from utils.image_process import crop_resize_data
data = pd.read_csv(os.path.join(os.getcwd(), "data_list", "train.csv"), header=None,
                 names=["image", "label"])
images = data["image"].values[1:]
labels = data["label"].values[1:]
ori_image = cv2.imread(images[0])
ori_mask = cv2.imread(labels[0], cv2.IMREAD_GRAYSCALE)
ori_image, ori_mask = crop_resize_data(ori_image,ori_mask)
seq = iaa.Sequential([iaa.CropAndPad(
            percent=(-0.05, 0.1))])
seg_to = seq.to_deterministic()
for i in range(8):
    seg_to = seq.to_deterministic()
    ori_image = seg_to.augment_image(ori_image)  # 将方法应用在原图像上
    ori_mask = seg_to.augment_image(ori_mask)
    print(ori_mask.dtype)
    print(ori_mask.shape)
    plt.imshow(ori_mask)
    plt.show()