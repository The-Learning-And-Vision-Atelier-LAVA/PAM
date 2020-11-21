# Test on KITTI 2015 for submission
## max_disp = 0
python submission.py --max_disp=0 \
                     --datapath='D:/LongguangWang/Data/KITTI2015'
                     --ckpt='./log/PASMnet_KITTI2015_epoch80.pth'

## max_disp = 192
python submission.py --max_disp=192 \
                     --datapath='D:/LongguangWang/Data/KITTI2015'
                     --ckpt='./log/PASMnet_192_KITTI2015_epoch80.pth'