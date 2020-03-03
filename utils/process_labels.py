import numpy as np
import colorsys


def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]))
    # 0
    id_train = {
        0: [0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218, 219, 232, 202, 231, 230, 228, 229, 233, 212, 223],
        1: [200, 204, 209], 2: [201, 203], 3: [217], 4: [210], 5: [214],
        6: [220, 221, 222, 224, 225, 226], 7: [205, 227, 250]}
    for i in range(8):
        for item in id_train[i]:
            encode_mask[color_mask == item] = i

    return encode_mask


def decode_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    deocde_mask[labels == 0] = 0
    # 1
    deocde_mask[labels == 1] = 204
    # 2
    deocde_mask[labels == 2] = 203
    # 3
    deocde_mask[labels == 3] = 217
    # 4
    deocde_mask[labels == 4] = 210
    # 5
    deocde_mask[labels == 5] = 214
    # 6
    deocde_mask[labels == 6] = 224
    # 7
    deocde_mask[labels == 7] = 227

    return deocde_mask


def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    decode_mask[0][labels == 0] = 0
    decode_mask[1][labels == 0] = 0
    decode_mask[2][labels == 0] = 0
    # 1
    decode_mask[0][labels == 1] = 70
    decode_mask[1][labels == 1] = 130
    decode_mask[2][labels == 1] = 180
    # 2
    decode_mask[0][labels == 2] = 0
    decode_mask[1][labels == 2] = 0
    decode_mask[2][labels == 2] = 142
    # 3
    decode_mask[0][labels == 3] = 153
    decode_mask[1][labels == 3] = 153
    decode_mask[2][labels == 3] = 153
    # 4
    decode_mask[0][labels == 4] = 128
    decode_mask[1][labels == 4] = 64
    decode_mask[2][labels == 4] = 128
    # 5
    decode_mask[0][labels == 5] = 190
    decode_mask[1][labels == 5] = 153
    decode_mask[2][labels == 5] = 153
    # 6
    decode_mask[0][labels == 6] = 0
    decode_mask[1][labels == 6] = 0
    decode_mask[2][labels == 6] = 230
    # 7
    decode_mask[0][labels == 7] = 255
    decode_mask[1][labels == 7] = 128
    decode_mask[2][labels == 7] = 0

    return decode_mask


def class_colors(num_classes, bright=True):
    """
    based on the class id to choose a centrial color to show them
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / np.float(num_classes), 1, brightness) for i in range(num_classes)]
    color_map = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    color_map = np.array(color_map)

    return color_map


def verify_labels(labels):
    pixels = [0]
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            pixel = labels[x, y]
            if pixel not in pixels:
                pixels.append(pixel)
    print('The Labels Has Value:', pixels)
