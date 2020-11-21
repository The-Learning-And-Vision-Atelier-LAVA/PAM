import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from utils import *


def L1Loss(input, target):
    return (input - target).abs().mean()


def loss_disp_unsupervised(img_left, img_right, disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped = warp_disp(img_right, -disp)

    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask

    loss = 0.15 * L1Loss(image_warped * valid_mask, img_left * valid_mask) + \
           0.85 * (valid_mask * (1 - ssim(img_left, image_warped)) / 2).mean()
    return loss


def loss_disp_smoothness(disp, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
            ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss


def loss_pam_photometric(img_left, img_right, att, valid_mask, mask=None):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(img_left.device)

    for idx_scale in range(len(att)):
        scale = img_left.size()[2] // valid_mask[idx_scale][0].size()[2]
        b, c, h, w = valid_mask[idx_scale][0].size()

        att_right2left = att[idx_scale][0]                          # b * h * w * w
        att_left2right = att[idx_scale][1]
        valid_mask_left = valid_mask[idx_scale][0]                  # b * 1 * h * w
        valid_mask_right = valid_mask[idx_scale][1]

        if mask is not None:
            valid_mask_left = valid_mask_left * (nn.AvgPool2d(scale)(mask[0].float()) > 0).float()
            valid_mask_right = valid_mask_right * (nn.AvgPool2d(scale)(mask[1].float()) > 0).float()

        img_left_scale  = F.interpolate(img_left,  scale_factor=1/scale, mode='bilinear')
        img_right_scale = F.interpolate(img_right, scale_factor=1/scale, mode='bilinear')

        img_right_warp = torch.matmul(att_right2left, img_right_scale.permute(0, 2, 3, 1).contiguous())
        img_right_warp = img_right_warp.permute(0, 3, 1, 2)
        img_left_warp = torch.matmul(att_left2right,  img_left_scale.permute(0, 2, 3, 1).contiguous())
        img_left_warp = img_left_warp.permute(0, 3, 1, 2)

        loss_scale = L1Loss(img_left_scale * valid_mask_left, img_right_warp * valid_mask_left) + \
                     L1Loss(img_right_scale * valid_mask_right, img_left_warp * valid_mask_right)

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_cycle(att_cycle, valid_mask):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att_cycle[0][0].device)

    for idx_scale in range(len(att_cycle)):
        b, c, h, w = valid_mask[idx_scale][0].shape
        I = torch.eye(w, w).repeat(b, h, 1, 1).to(att_cycle[0][0].device)

        att_left2right2left = att_cycle[idx_scale][0]
        att_right2left2right = att_cycle[idx_scale][1]
        valid_mask_left = valid_mask[idx_scale][0]
        valid_mask_right = valid_mask[idx_scale][1]

        loss_scale = L1Loss(att_left2right2left * valid_mask_left.permute(0, 2, 3, 1), I * valid_mask_left.permute(0, 2, 3, 1)) + \
                     L1Loss(att_right2left2right * valid_mask_right.permute(0, 2, 3, 1), I * valid_mask_right.permute(0, 2, 3, 1))

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_smoothness(att):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att[0][0].device)

    for idx_scale in range(len(att)):
        att_right2left = att[idx_scale][0]
        att_left2right = att[idx_scale][1]

        loss_scale = L1Loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) + \
                     L1Loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) + \
                     L1Loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) + \
                     L1Loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def warp_disp(img, disp):
    '''
    borrowed from
    '''
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def ssim(img1, img2, window_size=11):
    _, channel, h, w = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)
