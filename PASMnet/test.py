from models.PASMnet import *
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from datasets.kitti_dataset import KITTIDataset
from datasets.sceneflow_dataset import SceneFlowDatset
import argparse
import numpy as np
import time
from utils import EPE_metric, D1_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_disp', type=int, default=0, help='prior maximum disparity, 0 for unavailable')

    parser.add_argument('--dataset', type=str, default='SceneFlow')
    parser.add_argument('--datapath', type=str, default='D:/LongguangWang/Data/SceneFlow')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default='./log/PASMnet_SceneFlow_epoch10.pth')

    return parser.parse_args()


def test(test_loader, cfg):
    net = PASMnet().to(cfg.device)
    if cfg.ckpt.split('.')[-1] == 'tar':
        ckpt = torch.load(cfg.ckpt)['state_dict']
    else:
        ckpt = torch.load(cfg.ckpt)
    net.load_state_dict(ckpt)
    net.eval()
    cudnn.benchmark = True

    EPE = []
    D1 = []
    D3 = []

    with torch.no_grad():
        for iteration, data in enumerate(test_loader):
            img_left, img_right = data['left'].to(cfg.device), data['right'].to(cfg.device)
            disp_gt = data['left_disp'].to(cfg.device).unsqueeze(1)
            mask_0 = (disp_gt > 0)

            disp = net(img_left, img_right, max_disp=cfg.max_disp)

            ## mode 1: all pixels
            # EPE.append(EPE_metric(disp[-1], disp_gt, mask_0).data.cpu())
            # D1.append(D1_metric(disp[-1], disp_gt, mask_0, 1).data.cpu())
            # D3.append(D1_metric(disp[-1], disp_gt, mask_0, 3).data.cpu())

            ## mode 2: exclude pixels with disparites > 192
            mask_192 = (disp_gt < 192) & mask_0
            if float(torch.sum(mask_192)) > 0:
                EPE += EPE_metric(disp, disp_gt, mask_192)
                D1 += D1_metric(disp, disp_gt, mask_192, 1)
                D3 += D1_metric(disp, disp_gt, mask_192, 3)

            print('### Iteration %5d of total %5d --- EPE: %.3f ---D1: %.3f ---D3: %.3f ###' %
                  (iteration+1, len(test_loader.dataset.left_filenames)//cfg.batch_size,
                   float(np.array(EPE).mean()), 100*float(np.array(D1).mean()), 100*float(np.array(D3).mean())))

    print('Mean EPE: %.3f, Mean D1: %.3f, Mean D3: %.3f' %
          (float(np.array(EPE).mean()), 100*float(np.array(D1).mean()), 100*float(np.array(D3).mean())))

def main(cfg):
    if cfg.dataset == 'SceneFlow':
        test_set = SceneFlowDatset(datapath=cfg.datapath, list_filename='filenames/sceneflow_test.txt', training=False)
    if cfg.dataset == 'KITTI2012':
        test_set = KITTIDataset(datapath=cfg.datapath, list_filename='filenames/kitti12_val.txt', training=False)
    if cfg.dataset == 'KITTI2015':
        test_set = KITTIDataset(datapath=cfg.datapath, list_filename='filenames/kitti15_val.txt', training=False)

    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=cfg.batch_size, shuffle=False)
    test(test_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)