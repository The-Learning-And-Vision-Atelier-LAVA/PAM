import os
import torch
import numpy as np
from torch.nn import init
import torch.nn.functional as F


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1:
#         init.kaiming_normal_(m.weight.data, a=0.1)


def D1_metric(D_est, D_gt, mask, threshold=3):
    mask = mask.byte()
    error = []
    for i in range(D_gt.size(0)):
        D_est_, D_gt_ = D_est[i,...][mask[i,...]], D_gt[i,...][mask[i,...]]
        if len(D_gt_) > 0:
            E = torch.abs(D_gt_ - D_est_)
            err_mask = (E > threshold) & (E / D_gt_.abs() > 0.05)
            error.append(torch.mean(err_mask.float()).data.cpu())
    return error


def EPE_metric(D_est, D_gt, mask):
    mask = mask.byte()
    error = []
    for i in range(D_gt.size(0)):
        D_est_, D_gt_ = D_est[i,...][mask[i,...]], D_gt[i,...][mask[i,...]]
        if len(D_gt_) > 0:
            error.append(F.l1_loss(D_est_, D_gt_, size_average=True).data.cpu())
    return error



