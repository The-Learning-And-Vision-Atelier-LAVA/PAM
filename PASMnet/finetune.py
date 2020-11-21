from models.PASMnet import *
from datasets.kitti_dataset import KITTIDataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
from loss import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_disp', type=int, default=0, help='prior maximum disparity, 0 for unavailable')

    parser.add_argument('--dataset', type=str, default='KITTI2015')
    parser.add_argument('--datapath', default='D:/LongguangWang/Data/KITTI2015', help='data path')
    parser.add_argument('--savepath', default='log/', help='save path')

    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--n_workers', type=int, default=2, help='number of threads in dataloader')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=60, help='number of epochs to update learning rate')
    parser.add_argument('--resume_model', type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=1, help='the frequency of printing losses (epchs)')
    parser.add_argument('--save_freq', type=int, default=40, help='the frequency of saving models (epochs)')

    return parser.parse_args()


def train(train_loader, cfg):
    net = PASMnet().to(cfg.device)
    net = nn.DataParallel(net, device_ids=[0,1])
    net.train()
    cudnn.benchmark = True

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    loss_epoch = []
    loss_list = []
    EPE_epoch = []
    D3_epoch = []
    EPE_list = []

    epoch_start = 0

    if cfg.resume_model is None:
        # load model pre-trained on SceneFlow
        if cfg.max_disp == 0:
            ckpt = torch.load('log/PASMnet_SceneFlow_epoch10.pth.tar')
        else:
            ckpt = torch.load('log/PASMnet_' + str(cfg.max_disp) + '_SceneFlow_epoch10.pth.tar')

        if isinstance(net, nn.DataParallel):
            net.module.load_state_dict(ckpt['state_dict'])
        else:
            net.load_state_dict(ckpt['state_dict'])

    else:
        ckpt = torch.load(cfg.resume_model)

        if isinstance(net, nn.DataParallel):
            net.module.load_state_dict(ckpt['state_dict'])
        else:
            net.load_state_dict(ckpt['state_dict'])

        epoch_start = ckpt['epoch']
        loss_list = ckpt['loss']
        EPE_list = ckpt['EPE']

    for epoch in range(epoch_start, cfg.n_epochs):
        # lr stepwise
        lr = cfg.lr * (cfg.gamma ** -(epoch // cfg.n_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for iteration, data in enumerate(train_loader):
            img_left, img_right = data['left'].to(cfg.device), data['right'].to(cfg.device)
            disp_gt = data['left_disp'].to(cfg.device).unsqueeze(1)

            disp, att, att_cycle, valid_mask = net(img_left, img_right, max_disp=cfg.max_disp)

            # loss-D
            loss_P = loss_disp_unsupervised(img_left, img_right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'))

            # loss-S
            loss_S = loss_disp_smoothness(disp, img_left)

            # loss-PAM
            loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask)
            loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
            loss_PAM_S = loss_pam_smoothness(att)
            loss_PAM = loss_PAM_P + 5 * loss_PAM_S + 5 * loss_PAM_C

            # losses
            loss = loss_P + 0.5 * loss_S + loss_PAM
            loss_epoch.append(loss.data.cpu())

            # metrics
            mask = disp_gt > 0
            EPE_epoch += EPE_metric(disp, disp_gt, mask)
            for i in range(cfg.batch_size):
                D3_epoch += D1_metric(disp[i, :, :, :].unsqueeze(0), disp_gt[i, :, :, :].unsqueeze(0), mask[i, :, :, :].unsqueeze(0), 3)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print
        if (epoch+1) % cfg.print_freq == 0:
            print('Epoch----%5d, loss---%f, EPE---%f, D3---%f' %
                  (epoch + 1,
                   float(np.array(loss_epoch).mean()),
                   float(np.array(EPE_epoch).mean()),
                   float(np.array(D3_epoch).mean())))

        if (epoch+1) % cfg.save_freq == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            EPE_list.append(float(np.array(EPE_epoch).mean()))

            if cfg.max_disp == 0:
                filename = 'PASMnet_' + cfg.dataset + '_epoch' + str(epoch + 1) + '.pth.tar'
            else:
                filename = 'PASMnet_' + str(cfg.max_disp) + '_' + cfg.dataset + '_epoch' + str(epoch + 1) + '.pth.tar'

            save_ckpt({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                'loss': loss_list,
                'EPE': EPE_list
            }, save_path=cfg.savepath, filename=filename)

            loss_epoch = []
            EPE_epoch = []
            D3_epoch = []


def main(cfg):
    if cfg.dataset == 'KITTI2012':
        train_set = KITTIDataset(datapath=cfg.datapath, list_filename='filenames/kitti12_train.txt', training=True)
    if cfg.dataset == 'KITTI2015':
        train_set = KITTIDataset(datapath=cfg.datapath, list_filename='filenames/kitti15_train.txt', training=True)

    train_loader = DataLoader(dataset=train_set, num_workers=cfg.n_workers, batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

