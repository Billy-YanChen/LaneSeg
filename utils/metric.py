import numpy as np


def compute_iou(pred, gt, result):
    """
    pred : [N, H, W]
    gt: [N, H, W]
    """
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(8):
        single_gt = gt==i
        single_pred = pred==i
        temp_tp = np.sum(single_gt * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result
