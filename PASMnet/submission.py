from models.PASMnet import *
from torch.utils.data import DataLoader
from datasets.kitti_dataset import KITTIDataset
import torch.backends.cudnn as cudnn
import skimage
import argparse
import time
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_disp', type=int, default=0, help='prior maximum disparity, 0 for unavailable')

    parser.add_argument('--dataset', type=str, default='KITTI2015')
    parser.add_argument('--datapath', type=str, default='D:/LongguangWang/Data/KITTI2015')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default='./log/PASMnet_KITTI2015_epoch80.pth')

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

    if not os.path.exists('results/' + cfg.dataset + '_' + str(cfg.max_disp) ):
        os.mkdir('results/' + cfg.dataset + '_' + str(cfg.max_disp) )

    with torch.no_grad():
        for iteration, data in enumerate(test_loader):
            img_left, img_right = data['left'].to(cfg.device), data['right'].to(cfg.device)

            top_pad = int(data['top_pad'].data.cpu())
            right_pad = int(data['right_pad'].data.cpu())

            disp = net(img_left, img_right, max_disp=cfg.max_disp)

            if cfg.dataset == 'KITTI2015' or 'KITTI2012':
                disp = torch.clamp(disp[:, :, top_pad:, :-right_pad].squeeze().data.cpu(), 0).numpy()
                skimage.io.imsave('results/'+cfg.dataset+'_'+str(cfg.max_disp)+'/'+
                                  test_loader.dataset.left_filenames[iteration][-13:], (disp * 256).astype('uint16'))


def main(cfg):
    if cfg.dataset == 'KITTI2012':
        test_set = KITTIDataset(datapath=cfg.datapath, list_filename='filenames/kitti12_test.txt', training=False)
    if cfg.dataset == 'KITTI2015':
        test_set = KITTIDataset(datapath=cfg.datapath, list_filename='filenames/kitti15_test.txt', training=False)

    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test(test_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)